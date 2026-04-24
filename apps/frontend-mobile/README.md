# Mobile Frontend

`apps/frontend-mobile/` 是这次移动端适配的独立子前端，技术栈默认是：

- `React 19`
- `TypeScript 6`
- `Vite 8`
- `bun` 作为默认包管理器

## 推荐用法

先安装依赖：

```bash
bun install
```

本机如果要直接跑最新 `vite`，推荐显式带上 `--bun`：

```bash
bun run --bun dev
bun run --bun build
bun run --bun build:static
bun run --bun preview
bun run --bun lint
```

如果要把构建产物发布到 gateway 提供的 `/mobile/` 入口，用：

```bash
bun run --bun build:static
```

执行后会把 `dist/` 同步到仓库根的 `apps/frontend/mobile/`，`gateway.py` 会通过 `/mobile/` 路由 serve 它。
同时 `bun run --bun build:widget` 也会把 React 版设置浮层编译到 `apps/frontend/mobile-omni/`，覆盖手写壳里的 `settings-widget.js/css`。

## 本地联调代理

开发预览默认会把以下接口代理到：

```text
https://127.0.0.1:8006
```

包括：

- `/api/*`
- `/status`
- `/health`
- `/workers`
- `/ws/*`
- `/s/*`

如果你要切到别的 gateway（例如 HTTP 模式的 `http://127.0.0.1:8006`），可以在启动前覆盖：

```bash
VITE_PROXY_TARGET=http://127.0.0.1:8006 bun run --bun dev
```

之所以保留这层 dev proxy，是为了让 `frontend-mobile` 独立跑在 `5173/8032` 时也能直接连真实 gateway；而真正 build 到 `apps/frontend/mobile/` 后又可以走同源路径，不需要改业务代码。

## npm 兼容路径

脚本本身仍然保持标准的 `vite` / `tsc` 形式，没有为了 bun 做很重的魔改。
如果后续有人更习惯 `npm`，也可以继续走常规命令：

```bash
npm install
npm run dev
npm run build
npm run build:static
npm run preview
npm run lint
```

但 `npm` 路径需要本机 `Node >= 20.19` 或 `>= 22.12`，因为 `Vite 8` 的官方要求就是这个版本线。

## 当前已知问题

这次已经遇到两个和 bun 相关、值得明确记录的问题：

1. `bun create vite@latest` 在当前环境下会因为 `node:util` 的 `styleText` 导出报错，不能顺利完成脚手架初始化。
2. 在这台机器上，直接执行 `bun run dev` 仍可能让 `vite` 落回系统 `Node 16`，从而触发版本不满足。显式使用 `bun run --bun <script>` 可以绕开这个问题。

所以当前策略是：

- 依赖管理和本地开发优先使用 `bun`
- 保留 `npm` 兼容脚本，避免项目结构过度偏离常见前端习惯
- 对 bun 的已知兼容问题直接记录，不把它藏在工程细节里
