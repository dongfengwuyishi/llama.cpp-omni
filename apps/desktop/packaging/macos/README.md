# Comni — macOS Desktop App 打包

这一目录包含 macOS 平台 `.app` + `.dmg` 的打包配置、签名与公证流程。

## 快速打包

```bash
# 仓库根目录
bash apps/desktop/packaging/build_dmg.sh
```

脚本会自动检测：

- 本机 keychain 里有没有 **Developer ID Application** 证书 → 决定是 Developer ID 签名还是 ad-hoc fallback
- 有没有 **notarytool keychain profile** → 决定是否提交公证

然后产出 `apps/desktop/packaging/dist/Comni-macOS-<arch>-<version>.dmg`。

## 三种产出模式

| 模式 | 触发条件 | 用户体验 |
|---|---|---|
| **Notarized**（推荐） | 有 Developer ID 证书 + 有 notary profile | 双击直接打开，零警告 |
| **Signed-only** | 有 Developer ID 证书但没 profile | Gatekeeper 仍警告，需走系统设置放行 |
| **Ad-hoc** | 没有 Developer ID 证书 | M 系列能跑起来但 Gatekeeper 拦截，必须走系统设置或 `xattr -cr` 放行 |

build banner 会清楚告诉你当前是哪种模式：

```
==================================================
  Building Comni.app
==================================================
  Repo:    /Users/.../llama.cpp-omni
  ...
  Sign:    Developer ID Application: Tianchi Cai (VGXUYC2C5K) (team VGXUYC2C5K)
  Notary:  enabled (profile=comni-notarytool)
==================================================
```

## 首次设置（一次性，约 10 分钟）

### 1. 申请 Developer ID Application 证书

1. 进 https://developer.apple.com/account/resources/certificates/add
2. 在 **Software** 分组下选 **Developer ID Application**（不是 Apple Development、不是 Apple Distribution）
3. **Profile Type 选 G2 Sub-CA**（Previous Sub-CA 2027 年就过期）
4. 上传 CSR（已有的就复用，没有就 keychain 访问 → 证书助理 → 从证书颁发机构请求证书）
5. 下载 `.cer`，**双击导入 keychain**（让它和私钥配对）

验证：

```bash
security find-identity -p codesigning -v | grep "Developer ID Application"
# 应输出：
#   N) <hash> "Developer ID Application: <你的名字> (<TEAM_ID>)"
```

### 2. 申请 App-specific Password

1. 进 https://appleid.apple.com → 登录与安全 → 应用专用密码
2. 生成一个，标签写 `comni-notarytool`，得到 `xxxx-xxxx-xxxx-xxxx`，**保存好**

### 3. 把凭据存进 keychain

```bash
bash apps/desktop/packaging/macos/setup_notary.sh \
    --apple-id you@example.com \
    --app-password xxxx-xxxx-xxxx-xxxx
```

> Team ID 默认从证书自动读取。如果 keychain 里有多个 Developer ID 证书，可加 `--team-id VGXUYC2C5K` 显式指定。

完成后再运行 `build_dmg.sh`，banner 里会显示 `Notary: enabled`，公证会自动执行。

## 公证流程（脚本内部做了什么）

`build_dmg.sh` 在 DMG 转换为 UDZO 之后会执行：

```
1. codesign --force --timestamp --sign <Developer ID> Comni-*.dmg
2. xcrun notarytool submit Comni-*.dmg \
       --keychain-profile comni-notarytool --wait
3. xcrun stapler staple Comni-*.dmg          # 把票据打进 DMG
4. spctl -a -t open ...                       # 验证 Gatekeeper 通过
```

整个过程通常 1–3 分钟（Apple 服务器负载决定）。失败时会保留 `notarytool-<version>.log`、`notarytool-<version>-detail.json` 方便排查。

## 公证常见失败

| 错误 | 原因 | 解决 |
|---|---|---|
| `code object is not signed at all` | 某个 nested binary 没签到 | 看 detail.json 找出哪个文件，加进 `build_dmg.sh` 的签名循环 |
| `The binary uses an SDK older than the 10.9 SDK` | 某个 .dylib 太老 | 通常是 Python wheel 里的预编译 .so，更新依赖版本 |
| `The signature does not include a secure timestamp` | 用了 `--timestamp=none` 或网络挂了 | 检查能访问 `timestamp.apple.com` |
| `Invalid value for hardened-runtime` | entitlements.plist 写错 | 用 `plutil -lint entitlements.plist` 验证 |
| `Invalid Hardened Runtime` | 某可执行文件没加 `--options runtime` | 检查 `build_dmg.sh` 签名循环逻辑 |

## 文件清单

| 文件 | 说明 |
|---|---|
| `Info.plist` | bundle metadata，版本号脚本动态写入 |
| `Comni.icns` | App 图标 |
| `Comni.png` | 源图（Win 那边转 .ico 用） |
| `launcher_wrapper.sh` | bundle 主可执行，启动 menubar_app.py |
| **`entitlements.plist`** | hardened runtime 豁免：JIT、unsigned exec memory、library validation、麦克风/摄像头/网络 |
| **`setup_notary.sh`** | 一次性把 App-specific Password 存进 keychain profile |

## 跳过签名 / 公证（仅调试用）

```bash
# 只做 ad-hoc 签名（最快，但 Gatekeeper 拦截）
COMNI_SKIP_SIGN=1 bash apps/desktop/packaging/build_dmg.sh

# Developer ID 签名但跳过公证（适合本机快速迭代）
COMNI_SKIP_NOTARIZE=1 bash apps/desktop/packaging/build_dmg.sh

# 用其他签名身份（多账号场景）
COMNI_SIGN_IDENTITY="Developer ID Application: Other Name (XXXXXXXXXX)" \
COMNI_NOTARIZE_PROFILE="other-notarytool" \
    bash apps/desktop/packaging/build_dmg.sh
```

## 版本号管理

参考 [`../windows/README.md`](../windows/README.md) 的同名章节。mac 与 win 共用 `../VERSION_LOG.json`，`bump_version.py` 是 cross-platform helper。
