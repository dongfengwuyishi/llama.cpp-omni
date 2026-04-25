# Comni

基于 MiniCPM-o 4.5 + llama.cpp 的本机多模态对话应用。纯本地推理，不上传任何数据。

## 安装

**macOS（M 系列芯片）**

1. 双击打开 `Comni-macOS-arm64-<version>.dmg`
2. 把 `Comni.app` 拖到 `Applications`
3. 首次启动会被 Gatekeeper 拦截：右键 → 打开 → 再次点"打开"

**Windows**

1. 双击 `Comni-Setup-<version>-win64.exe`，按提示安装
2. 从开始菜单启动 `Comni`

## 使用

启动后会出现在菜单栏（Mac）或任务栏（Win），点一下图标有这些操作：

- **Start Server** 启动推理服务（首次启动会自动下载模型，约 2–4 GB，请耐心等待）
- **Open Web UI** 用默认浏览器打开本地网页 UI
- **Show Window** 打开主窗口，里面能看状态、切模型、改后端
- **Stop Server** / **Quit** 停止服务 / 退出

Web UI 首页有四张卡片，按需点击：

- **Turn-based Chat** 文本 / 图片 / 语音输入的多轮对话
- **Omni Full-Duplex** 实时音视频全双工，摄像头 + 麦克风一边看一边聊
- **Audio Full-Duplex** 纯语音全双工，低延迟语音对话
- **Mobile Preview** 移动端 UI 预览

## 模型下载

所有模型都下载到 `~/.comni/models/`，由 app 自己管理，新使用者使用默认Q4_K_M即可。

两种下载方式：

- **自动** 第一次点 Start Server 会弹出下载进度条，下完就直接启动
- **手动** 菜单栏 → Show Window → Manage Models，挑版本点 Download / Resume

下载源按网络情况自动选，按顺序回退：`huggingface.co` → `hf-mirror.com` → `modelscope.cn`（魔搭）。如果某一个不通或者下载太慢，会自动切到下一个，无需手动干预。中途断网 / 关应用都没关系，下次打开点 Resume 接着下。

如果想固定走某一个源，主窗口里有 **Source** 下拉，可选 Auto / HuggingFace / HF Mirror / ModelScope。国内用户大多用默认 Auto 即可；如果在公司网络下三个源都被限制，可手动指定到能访问的那一个。

如果已经有模型文件想直接导入，把整个模型目录扔进 `~/.comni/models/` 即可，Model Manager 会自动识别。

## 手机访问

Mac 版菜单栏 → 主窗口 → "QR" 按钮会弹出二维码，手机扫码即可打开同一局域网内的 Web UI。
由于用的是自签名证书，第一次访问浏览器会警告"连接不安全"，点"仍要访问"即可。

## 常见路径

- 模型和配置：`~/.comni/`（Windows 对应 `%USERPROFILE%\.comni\`）
- macOS 日志：`~/Library/Application Support/Comni/comni_service.log`
- Windows 日志：`%APPDATA%\Comni\comni_service.log`
- 版本号：`.app` / `.exe` 文件名里带版本，或右键"显示简介"（Windows 是"属性 → 详细信息"）

## 常见问题

**模型下载很慢 / 卡住** — 如果开了 VPN 或系统代理，先关掉再试（app 默认自己绕开代理，但有些代理会劫持所有流量）；还是不行就在主窗口的 **Source** 下拉里手动切 `HF Mirror` 或 `ModelScope`，再点 Resume 继续下。三个源走的是完全独立的 CDN，总有一个能用。

**Web UI 打不开、提示 SSL 错误** — 本地服务跑在 HTTPS 自签名证书上，浏览器会拦一次，点"高级 → 继续访问"即可。

**推理很慢** — 打开主窗口的"后端"设置，确认 Vision 后端用的是 CoreML / CUDA 而不是 CPU。

## 反馈

发现问题或想提需求，欢迎在 GitHub Issue 反馈。
