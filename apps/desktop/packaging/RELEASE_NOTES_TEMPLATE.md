# GitHub Release Notes 模板

每次发新 release 都按本模板填写 release body，保持各版本格式一致。

整体的版本号约定见 [`VERSIONING.md`](./VERSIONING.md)：Comni 使用单一版本号同时覆盖 macOS 与 Windows，每次 release 把同一个数字 +1。

---

## 模板

```markdown
## Comni v{VER}

**Omni inference in C/C++**, built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

### Download

| 平台 | 链接 |
|------|------|
| macOS (Apple Silicon) | [`Comni-macOS-arm64.dmg`](https://github.com/tc-mb/llama.cpp-omni/releases/latest/download/Comni-macOS-arm64.dmg) |
| Windows (x64) | [`Comni-Setup-win64.exe`](https://github.com/tc-mb/llama.cpp-omni/releases/latest/download/Comni-Setup-win64.exe) |

### What's new

- {ITEM_1}
- {ITEM_2}

### Install

#### macOS

1. 下载 `.dmg`，双击挂载
2. 把 `Comni.app` 拖到 `/Applications`
3. 首次启动如果被 Gatekeeper 拦：**Finder 里右键点击 Comni.app → 打开**，再点一次 *打开* 即可

#### Windows

1. 下载 `.exe`，双击运行安装向导
2. 按提示完成安装
3. 从开始菜单或桌面启动 Comni

### System requirements

| 平台 | 要求 |
|------|------|
| macOS | macOS 12 (Monterey)+，Apple Silicon (M1+) 推荐 |
| Windows | Windows 10/11 x64 |
| 内存 | ≥ 16 GB（运行 Q4 模型） |
| 磁盘 | 8–10 GB（首次启动会下载模型） |
```

---

## 占位字段

| 占位 | 含义 | 示例 |
|------|------|------|
| `{VER}` | 本次 release 的版本号；对 macOS 和 Windows 使用同一个值 | `1.0.19` |
| `{ITEM_x}` | 用户感知到的变化，3-6 条短 bullet | `优化端侧 UI` |

## 字段约定

- **H2 标题**：固定写 `## Comni v{VER}`，包含品牌词与版本号；不要分平台单独写两份。
- **副标语**：固定写 `**Omni inference in C/C++**, built on [llama.cpp](...)`。与 README / About 弹窗 / 仓库 description 一致。
- **Download 表**：表头固定 `| 平台 | 链接 |`；链接用 markdown 内联格式 `` [`文件名`](URL) ``；URL 始终指向 `releases/latest/download/<不带版本号的别名>`。
- **What's new**：只写用户感知到的变化，3-6 条短 bullet。
- **Install / System requirements**：原样保留，即使本次只更新单平台，也保留双平台对称。
- **release name**（`gh release` 的 `--title`）：固定写 `Comni — macOS + Windows`，不要在 name 里写版本号，避免在 release 列表被截断。

## 资产命名约定

每个平台同时上传两个同内容文件，版本号都用本次 release 的统一 `{VER}`：

| 文件名 | 用途 |
|--------|------|
| `Comni-macOS-arm64-{VER}.dmg` | 带版本号，归档 / 历史回溯 |
| `Comni-macOS-arm64.dmg` | 不带版本号，作为永久最新链 |
| `Comni-Setup-{VER}-win64.exe` | 带版本号，归档 / 历史回溯 |
| `Comni-Setup-win64.exe` | 不带版本号，作为永久最新链 |

不带版本号的别名是公众号 / 官网外链固定地址，每次发新 release 时把它一起更新。

如果某次 release 只更新了一个平台，另一平台的 `.dmg` / `.exe` 仍然以同一个 `{VER}` 重新发布；其内容（SHA-256）与上一版完全一致，仅是为了让 `releases/latest/download/...` 永远指向最新 release。
