from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from typing import Any

from core.hapi_optical_depth_manager import (
    HapiCalculationCancelled,
    HapiOpticalDepthManager,
)


def _emit(event_type: str, **payload: Any) -> None:
    print(
        # 进程管道在部分Windows区域设置下不是UTF-8；ASCII转义可避免中文编码失败。
        json.dumps({"type": event_type, **payload}, ensure_ascii=True),
        flush=True,
    )


def _calculate_batch(
    manager: HapiOpticalDepthManager,
    arguments: dict[str, Any],
    profiles: list[dict[str, Any]],
    cancel_path: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    profile_count = len(profiles)
    progress_scale = 1000
    for position, profile_item in enumerate(profiles):
        if cancel_path.exists():
            raise HapiCalculationCancelled("HAPI代表廓线批量计算已取消。")

        profile_arguments = dict(arguments)
        profile_arguments["profile_path"] = str(profile_item["profile_path"])
        profile_arguments["for_index"] = int(profile_item.get("for_index", 0))

        def batch_progress(value: int, maximum: int, message: str) -> None:
            fraction = float(value) / float(maximum) if maximum > 0 else 0.0
            completed = int(round((position + max(0.0, min(fraction, 1.0))) * progress_scale))
            _emit(
                "progress",
                value=completed,
                maximum=max(profile_count * progress_scale, 1),
                message=f"[{position + 1}/{profile_count}] {message}",
            )

        try:
            result = manager.calculate(
                **profile_arguments,
                progress=batch_progress,
                cancelled=cancel_path.exists,
            )
        except HapiCalculationCancelled:
            raise
        except BaseException as exc:  # noqa: BLE001
            failures.append(
                {
                    "batch_index": int(profile_item.get("batch_index", position)),
                    "profile_path": str(profile_item["profile_path"]),
                    "message": str(exc) or type(exc).__name__,
                }
            )
            continue
        result["batch_index"] = int(profile_item.get("batch_index", position))
        result["source_profile_path"] = str(profile_item["profile_path"])
        results.append(result)
        _emit(
            "progress",
            value=(position + 1) * progress_scale,
            maximum=max(profile_count * progress_scale, 1),
            message=f"已完成代表廓线 {position + 1}/{profile_count}。",
        )
    return {
        "profile_count": profile_count,
        "results": results,
        "failures": failures,
    }


def main() -> int:
    if len(sys.argv) != 2:
        _emit("error", message="HAPI worker缺少计算请求文件。")
        return 2
    request_path = Path(sys.argv[1]).resolve()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        manager = HapiOpticalDepthManager(
            hapi_path=request.get("hapi_path"),
            database_dir=request.get("database_dir"),
            table_sources=dict(request.get("table_sources", {})),
        )
        cancel_path = Path(request["cancel_path"])

        def progress(value: int, maximum: int, message: str) -> None:
            _emit(
                "progress",
                value=int(value),
                maximum=int(maximum),
                message=str(message),
            )

        arguments = dict(request["arguments"])
        batch_profiles = list(request.get("batch_profiles", []))
        if batch_profiles:
            batch_result = _calculate_batch(
                manager, arguments, batch_profiles, cancel_path
            )
            if not batch_result["results"]:
                messages = "；".join(
                    str(item.get("message", "未知错误"))
                    for item in batch_result["failures"]
                )
                raise RuntimeError(f"全部代表廓线计算失败：{messages}")
            _emit("batch_result", result=batch_result)
            return 0
        result = manager.calculate(
            **arguments, progress=progress, cancelled=cancel_path.exists
        )
    except HapiCalculationCancelled as exc:
        _emit("cancelled", message=str(exc))
        return 0
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _emit("error", message=str(exc) or type(exc).__name__)
        return 1
    _emit("result", result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
