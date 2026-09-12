# pixelart

**图片 → 2.5D 像素风转换器。** 输入一张图，输出具有深度雾、体积光、光锥、尘埃与分层视差的像素画：
**静态图（PNG）** 或 **4~12 秒无缝循环视频（无损 WebP）**。

---

## 特性

- **深度感知大气**：单目深度驱动的空气透视雾、体积光、可解耦顶点的光锥、屏幕阴影光柱
- **双光源**：副光源独立位置/强度，散射色自动采样，线性加法混合
- **分层视差推拉** 与 **尘埃粒子**（数量/亮度/闪烁/远近衰减）
- **天空替换**（可选）：自适应检测天空区域，替换为程序化渐变 + 星场（夜/暮/昼）
- **细节预算**：低网格下细线与五官不被均值抹掉
- **取色模式**：色板吸附（4~128 色，经典像素画）⇄ 连续色（跳过吸附的高保真后处理）
- **循环完美**：所有动画严格周期，循环闭合逐位验证；全局色板，逐帧零漂移
- **WebGPU 加速**：全渲染链在浏览器 GPU 上执行（与 CPU 逐位一致），失败自动回退
- **调参台**：本地 Web UI，全参数即时预览、参数扫描与存取、调试图（深度/等亮线/体积光）、自由缩放平移画布、中英双语界面
- **出片**：全量出图与拖图出画面均通过浏览器下载交付

---

## 快速开始

**Windows**：双击 **`start.bat`** —— 自动创建虚拟环境、安装依赖、下载深度模型（走 hf-mirror）、启动服务并打开浏览器。

手动方式（任意平台）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux/macOS
.venv/Scripts/python tools/fetch_models.py                # 下载深度模型（≈99 MB）
.venv/Scripts/python tools/m3_server.py                   # 启动调参台
```

打开 <http://127.0.0.1:8770>，选择文件上传或是拖拽文件进入页面， 并在素材下拉中选中，调参满意后点「全量出图」——动画与首帧会自动下载。

命令行出片（不用界面）：

```bash
.venv/Scripts/python tools/m2_render.py --src assets/input/your.jpg --aspect native --seconds 6
```

---

## 项目结构

```
pixelart/
├─ start.bat               # Windows 一键启动（环境自检 + 自动补齐 + 拉起浏览器）
├─ requirements.txt
├─ src/pixelart/           # 管线核心（纯 Python，CPU 可跑）
│  ├─ analyze.py           #   单目深度（Depth Anything V2，ONNX）
│  ├─ resample.py          #   结构感知降采样 + 影调 + 细节预算
│  ├─ compose.py           #   深度雾 / 体积光 / 光锥 / 辉光
│  ├─ sky.py               #   天空检测与替换（渐变 + 星场）
│  ├─ masks.py             #   语义掩膜 / 光源检测
│  ├─ palette.py           #   色板：取色 / 吸附 / 显著度赎回
│  ├─ pixelate.py          #   像素尾巴（取色模式 / 抖动 / 边缘）
│  ├─ webgpu.py            #   WebGPU uniform 打包
│  └─ encode.py            #   输出编码（无损 WebP / APNG / MP4）
├─ tools/                  # 调参台服务 + CLI 出片 + 模型下载 + 诊断工具
│  └─ wgsl/                #   渲染链的 WGSL 移植（与 CPU 逐位比对）
├─ tests/                  # 单元测试
├─ docs/                   # 设计文档 / 先例调研 / spike 记录
├─ scripts/                # 环境脚本
└─ assets/input/           # 放你自己的图片（不入库）
```

---

## 说明

- 深度模型（Depth Anything V2 Small，≈99 MB）首次运行时自动从 hf-mirror 下载，也可用 `HF_ENDPOINT` 覆盖下载源
- 素材目录不入库（可放自己的图片）；`models/`、`out/`、`.venv/` 同样不随仓库分发
- 输出编码：默认无损动画 WebP（色板精确）；MP4 需安装 `imageio-ffmpeg` 并使用 `yuv444p`
- 运行环境：Windows（一键脚本/调参台）；管线本体跨平台，Python ≥ 3.11
- 许可：待定

---

## 调参台一览

- **左轨**：网格、预处理、天空替换、雾、体积光与辉光、像素尾巴、动画幅度、输出——全参数实时预览
- **取景台**：滚轮以光标为中心缩放、中键拖动平移；左键拖图出画面 = 保存当前帧
- **调试图**：深度图 / 等亮线 / 体积光形状，与主画面并排，看清每个阶段在做什么
- **全量出图**：后台全分辨率渲染，完成后动画（无损 WebP）与首帧（PNG）自动经浏览器下载
