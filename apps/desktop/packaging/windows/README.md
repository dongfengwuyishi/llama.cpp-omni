# Comni — Windows Desktop App

这一目录包含 Windows 平台原生桌面应用（`Comni.exe`）的构建配置。

## 架构

```
Comni.exe  (PySide6 GUI, PyInstaller onedir)
  └── 通过 subprocess 拉起:
        python.exe worker.py           (worker 推理，调 llama-server.exe)
        python.exe gateway.py          (FastAPI HTTP 服务)
```

和 macOS 的 `menubar_app.py` 设计保持一致：GUI 自身不做推理，只是启动 / 停止 /
监控后端服务 + 打开浏览器的 Web UI。

## 快速开始（开发者模式）

```powershell
# 1) 确保 miniconda / Python 3.10+ 已安装
#    装依赖：PySide6 + fastapi + uvicorn + …
pwsh apps/start.ps1

# 2) 第一次启动会自动装依赖并尝试定位 llama-server.exe
#    如果还没编译过，先自己 build 一次：
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON   # 有 NVIDIA GPU 时加 CUDA
cmake --build build --config Release --target llama-server -j
```

`apps/start.ps1` 等价于 Mac 的 `apps/start.sh`，只是用 PowerShell。

## 打包成 `Comni.exe`（发布模式）

```powershell
# 在仓库根目录执行。这会：
#   1. 装 PySide6 / Pillow / PyInstaller
#   2. 把 macOS 图标 Comni.png 转为 Comni.ico（9 个尺寸）
#   3. 如果还没有 build/bin/Release/llama-server.exe，自动 cmake build
#      （有 NVIDIA GPU 自动开 CUDA）
#   4. 调 PyInstaller 按 comni.spec 打包
pwsh apps/desktop/packaging/windows/build.ps1

# 产出：
#   dist/Comni/
#     Comni.exe                       <- 双击启动
#     _internal/...                   <- PyInstaller 运行时
#     resources/apps/...              <- server/ frontend/ assets/
#     resources/build/bin/Release/    <- llama-server.exe + *.dll
```

常用开关：

```powershell
pwsh apps/desktop/packaging/windows/build.ps1 -Clean            # 清掉旧的 dist/
pwsh apps/desktop/packaging/windows/build.ps1 -SkipLlamaBuild   # 不重建 llama-server
pwsh apps/desktop/packaging/windows/build.ps1 -SkipInstallDeps  # 跳过 pip install
pwsh apps/desktop/packaging/windows/build.ps1 -Python 'C:\...\python.exe'
```

## 运行时 Python 依赖

`Comni.exe` 自身**不捆绑**重量级推理依赖（fastapi / onnxruntime / huggingface_hub 等）。
它在启动子进程时，会按以下顺序查找 Python 解释器：

1. 环境变量 `COMNI_PYTHON`（优先级最高）
2. `<Comni.exe 同级目录>/python-embed/python.exe`（可选，见下方「完整独立包」）
3. `%USERPROFILE%\miniconda3\python.exe`，`…\anaconda3\python.exe`
4. `PATH` 里的 `python.exe` / `py.exe`

找到 Python 后，`worker.py` / `gateway.py` 需要的依赖必须装在那个 Python 里：

```powershell
<path-to-python>\python.exe -m pip install -r resources\apps\requirements.txt
```

## 完整独立包（可选，不依赖用户自带 Python）

如果希望用户双击 `Comni.exe` 即可运行，无需预装 Python：

1. 下载 [python embeddable package](https://www.python.org/downloads/windows/)
   （`python-3.12.x-embed-amd64.zip`）
2. 解压到 `dist/Comni/python-embed/`
3. 装依赖（解压后 embed 版自带 pip 有点麻烦，用 `get-pip.py` 引导一下）：
   ```powershell
   cd dist/Comni/python-embed
   (Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py).Content | .\python.exe -
   .\python.exe -m pip install -r ..\resources\apps\requirements.txt
   ```
4. 编辑 `python-embed\python312._pth` 取消 `import site` 注释
5. 重新分发 `dist/Comni/` 整个目录

`windows_app.py` 的 `_resolve_python` 会优先使用 `<exe>/python-embed/python.exe`。

## 文件清单

| 文件 | 说明 |
|---|---|
| `comni.spec` | PyInstaller 主配置，详见文件内注释 |
| `make_icon.py` | 把 `macos/Comni.png` 转成多尺寸 `Comni.ico` |
| `build.ps1` | 一键构建脚本（装依赖 → 转图标 → 编译 llama → 打包） |
| `version_info.txt` | 写入 `Comni.exe` 的 Windows 版本信息 |
| `Comni.ico` | 运行 `make_icon.py` 后生成，**不提交到 git** |

## 版本号管理（跨 mac/win 统一）

版本状态文件：`apps/desktop/packaging/VERSION_LOG.json`

```
apps/desktop/packaging/
├── bump_version.py     # 共享 helper（Python，mac/win 均可调用）
├── VERSION_LOG.json    # { current: {macos, windows}, history: [...] }
├── build_dmg.sh        # mac 打包，自动 peek → build → record
└── windows/
    └── make_installer.ps1   # -Version 由人传入，推荐先跑 peek
```

打包 Windows 前，先从 log 里拿下一个版本号：

```powershell
# 1) 预览下一版（只读，不写）
$ver = python apps\desktop\packaging\bump_version.py peek windows
Write-Host "Next version: $ver"

# 2) 正常走 PyInstaller + Inno Setup 打包
pwsh apps\desktop\packaging\windows\build.ps1
pwsh apps\desktop\packaging\windows\make_installer.ps1 -Version $ver

# 3) 构建都成功后，再把这一版写回 log（否则浪费号）
python apps\desktop\packaging\bump_version.py record windows `
    --version $ver `
    --commit (git rev-parse --short=7 HEAD) `
    --arch x64 `
    --note "installer=Comni-Setup-$ver-win64.exe"
```

默认 patch +1（`1.0.1 → 1.0.2`），需要时 `--bump minor|major` 或 `--set 1.2.3`
显式覆盖。`VERSION_LOG.json` 需要提交到 git，mac 侧 `build_dmg.sh`
也读写同一个文件。

## Symlink 说明

`~/.comni/models/` 下的模型通过符号链接/目录 Junction 指向真实模型目录，节省磁盘。
Windows 上 `os.symlink` 需要「开发者模式」或管理员权限：

- 开启开发者模式（推荐）：设置 → 隐私和安全性 → 开发者选项 → 打开「开发人员模式」
- 没开也没关系：`windows_app.py` 会自动降级到 `mklink /J` 目录 Junction（免管理员）

## 常见问题

**Q：双击 `Comni.exe` 闪退，没反应。**
A：打开 `%APPDATA%\Comni\comni_app.log` 查日志。大概率是找不到 Python 或
   `PYTHONPATH` 配错。

**Q：服务能启动但 Web UI 打不开。**
A：检查 `%APPDATA%\Comni\comni_service.log`。常见原因：llama-server.exe 加载模型
   失败（显存不够 / 模型路径错误）。

**Q：杀毒软件把 `Comni.exe` 标为可疑。**
A：PyInstaller onedir 的典型误报。可以加代码签名证书规避（放在 `build.ps1`
   里做 `signtool sign`），暂未默认启用。
