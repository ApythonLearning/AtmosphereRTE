# ARTE Atmosphere

面向地气系统建模与大气辐射传输的独立桌面软件。项目从 STIRS 中拆分并重组了地球环境、大气辐射传输、HAPI 分层吸收光学厚度和卫星光谱验证能力，不包含目标几何、网格、热求解及红外成像模块。

## 主要能力

- 读取 MOD11 陆地温度、海表温度、MOD08 云和 MCD12 地表类型，生成统一经纬度地球环境网格与 NPZ 缓存。
- 可选读取 MERRA-2 M2T1NXAER 的 AOD550 与 Ångström 指数；未提供时按能见度估算气溶胶光学厚度。
- 35 层 Beer–Lambert、层源函数与 δ-Eddington 型散射传输；包含地表、液态水云、OPAC 气溶胶、瑞利散射和 TSIS-1 太阳参考谱。
- 快速地球盘积分光谱与高分辨率目标单柱光谱。
- 从 NUCAPS 廓线调用 HAPI/HITRAN 生成 35 层逐波数总吸收光学厚度。
- 支持本地导入原始 HITRAN `.par` 文件或 HAPI `.data/.header` 文件对；完整文件对直接原地引用、不重复复制，并随项目保存其外部路径。
- 导入 CSV/TXT/DAT、NPZ、NetCDF、HDF5 卫星光谱，计算 RMSE、NRMSE、MAE、Bias、相关系数和光谱夹角。
- 按非重叠波长区间修正总光学厚度并另存，不覆盖原文件。
- 工作区项目保存、退出自动保存及启动自动读取；可恢复输入参数、环境缓存、光学厚度修正和最近光谱。
- 数值框与下拉框全局禁用鼠标滚轮调节，滚动页面时不会意外改变参数。
- HAPI 分层光学厚度与大气光谱验证均直接嵌入主界面页面，无需打开模态弹窗。

原软件中的“开发者模式”限制已移除，分层光学厚度和光谱验证可从主导航直接进入。

## 安装与启动

建议使用 Python 3.11 或 3.12：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

`pyhdf` 仅用于 HDF4 MODIS 产品；若只读取 HDF5/NetCDF 或已有环境缓存，可不安装它。首次 HAPI 计算可能需要联网下载 HITRAN 谱线，下载结果会保存在工作目录。

## 推荐工作流

1. 在“地球环境与大气辐射”页选择四类环境产品并生成缓存，或直接加载已有缓存。
2. 如需高分辨率计算，在“分层吸收光学厚度”页从 NUCAPS 生成 35 层 CSV，或导入已有 CSV。
3. 设置位置、高度、太阳方向和物理选项，执行快速地球盘或高分辨率单柱求解。
4. 在“大气光谱验证”页导入卫星产品进行共同波段匹配与残差分析。

输出与 HAPI 数据默认保存在界面右上角设置的工作目录。仓库内自带 TSIS-1 HSRS v2 太阳参考谱和 HAPI 程序，但不捆绑大体积 HITRAN 数据库、NUCAPS/MODIS/MERRA-2 或卫星观测样例。

项目保存在工作目录的 `arte_atmosphere_project.json`。点击“保存项目”可立即保存；关闭软件时自动保存。再次启动时会读取 `.arte_atmosphere_recent.json` 指向的上次工作区，并自动恢复可用缓存和最近光谱。“打开项目…”可读取其他工作区中的项目 JSON。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖指定位置参数、35 层光学厚度导入/修正和光谱验证指标。
