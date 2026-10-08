# ComfyUI-RelayImage

在 ComfyUI 里用**你自己中转站**（one-api / new-api / 自建 OpenAI 兼容网关）的 API 出图。

## 为什么需要它

ComfyUI 官方的 `OpenAIGPTImageNodeV2` / `OpenAIGPTImage1` 属于 **partner API 节点**：

- 隐藏输入是 `auth_token_comfy_org` / `api_key_comfy_org`
- **硬编码走 `api.comfy.org` 官方代理**，节点上**没有 `base_url` 字段**
- 所以无法指向自建中转站，也只能用 comfy.org 的余额计费

本节点补上这个缺口：只要你有一个 OpenAI 兼容的图像接口，就能在 ComfyUI 里直接调用。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<你的账号>/ComfyUI-RelayImage.git
# 重启 ComfyUI
```

依赖：`requests`、`numpy`、`torch`、`Pillow`（ComfyUI 环境里通常都有）。
如缺失：`pip install -r requirements.txt`

## 节点

### Relay Image（中转站/自定义 OpenAI 图像接口）

| 参数 | 说明 |
|---|---|
| `base_url` | 中转站根地址，**要带 `/v1`**，如 `http://127.0.0.1:8000/v1` |
| `api_key` | 中转站令牌，如 `sk-relay-xxxxxx` |
| `model` | 网关支持的模型别名，如 `gpt-image-2` / `nano-banana-2` / `nano-banana-pro` |
| `prompt` | 提示词 |
| `api_mode` | `auto` / `generations` / `edits` / `chat`（见下） |
| `size` | `auto` / `1024x1024` / `1536x1024` / `1024x1536` / `2048x2048` / `2048x1152` / `1152x2048` / `3840x2160` / `2160x3840` |
| `quality` | `auto` / `low` / `medium` / `high` |
| `background` | `auto` / `opaque` / `transparent` |
| `n` | 一次要几张（多数网关只返回第 1 张） |
| `timeout` | 单次请求超时秒数（长任务调大） |
| `image`（可选） | 参考图；**传 batch 即为多图**（编辑 / 融合） |
| `mask`（可选） | 局部重绘掩码，**白色区域会被替换**（仅 `edits` 模式） |
| `verify_ssl`（可选） | 自签 https 证书时关掉 |

输出：`image`（IMAGE）、`info`（STRING，调用详情与错误信息，便于排查）

### Relay Model List（列出网关模型）

拉取 `{base_url}/models`，把中转站支持的模型别名打印出来，省得猜模型名。

### Relay 通道路由（填了中转 key 就走中转，否则走官方）

一个节点决定整条工作流走哪条路，**免去手拨开关**：

| 参数 | 说明 |
|---|---|
| `mode` | `自动`（默认）/ `中转站` / `官方API` / `本地Qwen` |
| `relay_api_key` | 中转站令牌。**填了 → 走中转站；留空 → 走官方** |
| `relay_base_url` | 中转站地址，如 `https://xianai.cc/v1` |
| `relay_model` | 模型名，如 `gpt-image-2` |

输出（接到开关和 Relay Image 节点上）：

| 输出 | 接到哪 |
|---|---|
| `use_online` (BOOLEAN) | 「在线 / 本地」总开关的 `switch` |
| `use_relay` (BOOLEAN) | 「官方 / 中转站」通道开关的 `switch` |
| `api_key` / `base_url` / `model` (STRING) | Relay Image 节点的对应输入 |

自动模式的行为：

| 你填的 | 实际走的 |
|---|---|
| `relay_api_key` 有值 | 中转站 |
| `relay_api_key` 留空 | 官方 API |

运行日志会打印 `[RelayChannelRouter] mode=自动 有key=True -> 走 中转站(...)`，方便确认。


## 三种调用方式

| `api_mode` | 请求 | 用途 |
|---|---|---|
| `generations` | `POST {base_url}/images/generations` | 纯文生图 |
| `edits` | `POST {base_url}/images/edits`（multipart） | 图生图 / 多图融合 / 局部重绘 |
| `chat` | `POST {base_url}/chat/completions`（多模态） | 对话式出图，nano-banana 这类模型常用 |

`auto`（默认）：**无参考图走 `generations`；有参考图先走 `edits`，失败自动降级 `chat`**。
被 `edits` 和 `chat` 的差异坑过的话，就手动锁定模式。

响应解析兼容多种返回形态：

- `{"data":[{"b64_json": "..."}]}`
- `{"data":[{"url": "https://..."}]}`
- `{"data":[{"image": "..."}]}`
- chat 风格：从 `choices[0].message.content` 里提取 markdown 图片、直达链接或 `data:image/...;base64,...`

## 用法示例（对接本地中转站）

```
base_url : http://127.0.0.1:8000/v1
api_key  : sk-relay-your-token
model    : gpt-image-2
```

工作流最简结构：

```
[Relay Image] ─ image ─→ [SaveImage]
```

带参考图（编辑/融合）：

```
[LoadImage ×N] ─→ [ImageBatch 或 ImpactMakeImageBatch] ─ image ─→ [Relay Image] ─→ [SaveImage]
```

局部重绘：

```
[LoadImage] ─ image ─┐
                     ├─→ [Relay Image] (api_mode=edits) ─→ [SaveImage]
[Mask] ───────── mask┘
```

## 注意

- `api_key` 会明文保存在工作流 JSON 里。别把带 key 的工作流分享出去；本仓库也不会收集任何信息。
- 本节点直连你填的 `base_url`，**不经过 comfy.org**，计费按你中转站的价格走。
- 多数 OpenAI 兼容网关的 `n>1` 只会返回一张图，多图请改成多次运行或换网关。

## 许可

MIT
