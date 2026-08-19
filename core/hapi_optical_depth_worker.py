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

        result = manager.calculate(
            **dict(request["arguments"]),
            progress=progress,
            cancelled=cancel_path.exists,
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
