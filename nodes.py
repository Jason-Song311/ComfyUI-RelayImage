"""
Relay Image — 用「自定义 base_url / api_key」调用 OpenAI 兼容的图像接口。

为什么需要它：
    ComfyUI 官方的 OpenAIGPTImageNodeV2 / OpenAIGPTImage1 是 partner API 节点，
    硬编码走 comfy.org 官方代理（隐藏输入 auth_token_comfy_org / api_key_comfy_org），
    节点上没有 base_url 字段，因此无法指向自建中转站（one-api / new-api / 自研网关）。
    本节点补上这个缺口：只要你有一个 OpenAI 兼容的图像接口，就能在 ComfyUI 里直接出图。

支持三种调用方式（自动选择，也可手动指定）：
    generations : POST {base_url}/images/generations        纯文生图
    edits       : POST {base_url}/images/edits              multipart 图生图 / 多图融合 / 局部重绘
    chat        : POST {base_url}/chat/completions          多模态对话式出图（nano-banana 类常用）

作者: Jason-Song311
许可: MIT
"""

import base64
import io
import json
import time
import urllib.parse

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

try:  # ComfyUI 运行时提供；单独 import 本文件时降级为 no-op
    import comfy.utils  # noqa: F401
except Exception:  # pragma: no cover
    comfy = None


CATEGORY = "api node/image/Relay"

SIZE_CHOICES = [
    "auto",
    "1024x1024", "1536x1024", "1024x1536",
    "2048x2048", "2048x1152", "1152x2048",
    "3840x2160", "2160x3840",
]
QUALITY_CHOICES = ["auto", "low", "medium", "high"]
BACKGROUND_CHOICES = ["auto", "opaque", "transparent"]
MODE_CHOICES = ["auto", "generations", "edits", "chat"]
PROVIDER_CHOICES = ["自动", "官方", "中转站"]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _tensor_to_png_bytes(tensor):
    """ComfyUI IMAGE tensor (H,W,C) float 0-1 -> PNG bytes"""
    arr = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr[..., :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


def _mask_to_png_bytes(mask):
    """ComfyUI MASK (H,W) float 0-1 -> PNG bytes（白色=重绘区域）"""
    arr = (mask.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[0]
    img = Image.fromarray(arr, mode="L").convert("RGBA")
    width, height = img.size
    px = img.load()
    for y in range(height):
        for x in range(width):
            v = px[x, y][0]
            px[x, y] = (v, v, v, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_tensor(data):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _parse_error(resp, body_text):
    try:
        j = json.loads(body_text)
        msg = j.get("error")
        if isinstance(msg, dict):
            msg = msg.get("message") or json.dumps(msg, ensure_ascii=False)
        if not msg:
            msg = j.get("message") or j.get("detail")
    except Exception:
        msg = None
    return f"HTTP {resp.status_code}: {msg or body_text[:400]}"


def _extract_images_from_response(payload, session, timeout):
    """从多种 OpenAI 兼容响应里提取图片 bytes。

    支持:
        {"data": [{"b64_json": ...}, {"url": ...}]}
        {"data": [{"image": "...b64 或 http..."}]}
        chat 风格: content 里的 markdown 图片 / data URL
    返回: (list[bytes], 说明文本)
    """
    out, notes = [], []
    seen_urls = set()
    seen_b64 = set()

    def add_b64(s, where):
        s = s.strip()
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        if s in seen_b64:
            return
        try:
            out.append(base64.b64decode(s))
            seen_b64.add(s)
            notes.append(f"{where}:b64")
        except Exception as e:
            notes.append(f"{where}:b64 解码失败({e})")

    def add_url(u, where):
        if u in seen_urls:
            return
        seen_urls.add(u)
        try:
            r = session.get(u, timeout=timeout)
            r.raise_for_status()
            out.append(r.content)
            notes.append(f"{where}:url")
        except Exception as e:
            notes.append(f"{where}:url 下载失败({e})")

    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None and "images" in payload:
            data = payload["images"]
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    if item.get("b64_json"):
                        add_b64(item["b64_json"], f"data[{i}].b64_json")
                    elif item.get("image"):
                        v = item["image"]
                        add_url(v, f"data[{i}].image") if str(v).startswith("http") else add_b64(str(v), f"data[{i}].image")
                    elif item.get("url"):
                        add_url(item["url"], f"data[{i}].url")
                elif isinstance(item, str):
                    add_url(item, f"data[{i}]") if item.startswith("http") else add_b64(item, f"data[{i}]")

        # chat/completions 风格
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text"):
                            text_parts.append(part["text"])
                        elif part.get("type") == "image_url":
                            u = (part.get("image_url") or {}).get("url", "")
                            if u.startswith("http"):
                                add_url(u, "content.image_url")
                            elif u.startswith("data:"):
                                add_b64(u, "content.image_url")
            blob = "\n".join(text_parts)
            if blob:
                notes.append("chat:text")
                # markdown 图片
                import re
                for u in re.findall(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", blob):
                    add_url(u, "chat.md")
                for u in re.findall(r"\((https?://[^)\s]+\.(?:png|jpg|jpeg|webp))\)", blob, re.I):
                    add_url(u, "chat.link")
                for s in re.findall(r"data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=]+)", blob):
                    add_b64(s, "chat.dataurl")
                for s in re.findall(r"!\[[^\]]*\]\((data:image/[^)]+)\)", blob):
                    add_b64(s, "chat.mddata")
                if not out:
                    payload["_relay_text"] = blob
    return out, ", ".join(notes) if notes else "无图片"


# --------------------------------------------------------------------------- #
# 主节点
# --------------------------------------------------------------------------- #
class RelayImageNode:
    """统一图像节点：同一个节点里可以填「官方 comfy.org API Key」或「中转站 Key」，
    按「填了哪个」自动决定走官方代理还是走中转站。

    官方通道 = https://api.comfy.org/proxy/openai/images/{generations,edits}（X-API-KEY 认证）
    中转通道 = {relay_base_url}/images/{generations,edits} 或 {relay_base_url}/chat/completions
    """

    OFFICIAL_BASE = "https://api.comfy.org"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (PROVIDER_CHOICES, {
                    "default": "自动",
                    "tooltip": "自动：填了官方 Key 走官方、填了中转 Key 走中转站；也可强制指定",
                }),
                "official_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "ComfyUI 平台(platform.comfy.org)的 API Key。填了就走官方通道",
                }),
                "relay_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "中转站令牌(sk-xxx)。填了就走中转站",
                }),
                "model": ("STRING", {
                    "default": "gpt-image-2", "multiline": False,
                    "tooltip": "模型名，如 gpt-image-2 / gpt-image-2.5-flare / nano-banana-2",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "api_mode": (MODE_CHOICES, {
                    "default": "auto",
                    "tooltip": "auto：无参考图走文生图、有图走图生图；chat 仅中转站支持",
                }),
                "size": (SIZE_CHOICES, {"default": "auto"}),
                "quality": (QUALITY_CHOICES, {"default": "auto"}),
                "background": (BACKGROUND_CHOICES, {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 10}),
            },
            "optional": {
                "relay_base_url": ("STRING", {
                    "default": "https://xianai.cc/v1", "multiline": False,
                    "tooltip": "中转站地址（走中转站时用），要带 /v1",
                }),
                "image": ("IMAGE", {"tooltip": "参考图；传 batch 即为多图（编辑 / 融合）"}),
                "mask": ("MASK", {"tooltip": "局部重绘掩码，白色区域会被替换"}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    # ------------------------------------------------------------------ #
    def run(self, provider, official_api_key, relay_api_key, model, prompt, api_mode,
            size, quality, background, n, timeout,
            relay_base_url="https://xianai.cc/v1", image=None, mask=None, verify_ssl=True):

        off_key = (official_api_key or "").strip()
        rel_key = (relay_api_key or "").strip()
        base = (relay_base_url or "").strip().rstrip("/")

        if provider == "官方":
            use_official = True
        elif provider == "中转站":
            use_official = False
        else:  # 自动
            if rel_key:
                use_official = False
            elif off_key:
                use_official = True
            else:
                raise ValueError(
                    "两个 Key 都没填：请在 official_api_key 填 comfy.org 的 Key，"
                    "或在 relay_api_key 填中转站令牌")

        if not use_official and not rel_key:
            raise ValueError("走中转站但 relay_api_key 是空的")
        if use_official and not off_key:
            raise ValueError("走官方通道但 official_api_key 是空的")
        if not prompt and image is None:
            raise ValueError("prompt 与 image 至少要有一个")

        session = requests.Session()
        session.verify = bool(verify_ssl)

        # 参考图 -> png bytes
        ref_imgs = []
        if image is not None:
            for i in range(image.shape[0]):
                ref_imgs.append(_tensor_to_png_bytes(image[i])[0])
        mask_bytes = _mask_to_png_bytes(mask) if mask is not None else None

        if use_official:
            return self._run_official(session, off_key, model, prompt, api_mode, size,
                                      quality, background, n, timeout, ref_imgs, mask_bytes)
        return self._run_relay(session, rel_key, base, model, prompt, api_mode, size,
                               quality, background, n, timeout, ref_imgs, mask_bytes)

    # ------------------------------------------------------------------ #
    def _run_official(self, session, api_key, model, prompt, api_mode, size, quality,
                      background, n, timeout, ref_imgs, mask_bytes):
        """官方 comfy.org 代理——接口与官方 OpenAIGPTImage* 节点完全一致。"""
        headers = {"X-API-KEY": api_key, "User-Agent": "ComfyUI-RelayImage/1.0"}
        common = {"model": model, "prompt": prompt, "n": n, "moderation": "low"}
        if quality != "auto":
            common["quality"] = quality
        if background != "auto":
            common["background"] = background
        if size != "auto":
            common["size"] = size

        errors = []
        want_edits = bool(ref_imgs) and api_mode in ("auto", "edits")

        if want_edits:
            url = f"{self.OFFICIAL_BASE}/proxy/openai/images/edits"
            data = {k: str(v) for k, v in common.items()}
            files = []
            for i, b in enumerate(ref_imgs):
                key = "image" if len(ref_imgs) == 1 else "image[]"
                files.append((key, (f"image_{i}.png", b, "image/png")))
            if mask_bytes:
                files.append(("mask", ("mask.png", mask_bytes, "image/png")))
            try:
                r = session.post(url, headers=headers, data=data, files=files, timeout=timeout)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, timeout)
                if imgs:
                    return self._pack(imgs, f"官方API | edits OK | {note} | {len(ref_imgs)} 张参考图")
                errors.append(f"官方 edits 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"官方 edits 失败: {e}")
            if api_mode == "edits":
                raise RuntimeError("官方 edits 调用失败 —— " + " ; ".join(errors))

        # 文生图
        url = f"{self.OFFICIAL_BASE}/proxy/openai/images/generations"
        try:
            r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                             data=json.dumps(common, ensure_ascii=False).encode("utf-8"),
                             timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(_parse_error(r, r.text))
            imgs, note = _extract_images_from_response(r.json(), session, timeout)
            if imgs:
                return self._pack(imgs, f"官方API | generations OK | {note}")
            errors.append(f"官方 generations 未返回图片: {r.text[:200]}")
        except Exception as e:
            errors.append(f"官方 generations 失败: {e}")

        raise RuntimeError("官方通道调用失败 —— " + " ; ".join(errors)
                           + "\n（检查 official_api_key 是否正确、平台是否有余额）")

    # ------------------------------------------------------------------ #
    def _run_relay(self, session, api_key, base, model, prompt, api_mode, size, quality,
                   background, n, timeout, ref_imgs, mask_bytes):
        if not base:
            raise ValueError("走中转站时 relay_base_url 不能为空，例如 https://xianai.cc/v1")

        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "ComfyUI-RelayImage/1.0"}
        if api_key:
            headers["x-api-key"] = api_key

        mode = api_mode
        if mode == "auto":
            mode = "generations" if not ref_imgs else "edits"

        errors = []

        # 1) edits（multipart）
        if mode == "edits":
            if not ref_imgs:
                raise ValueError("api_mode=edits 需要接一张参考图")
            url = f"{base}/images/edits"
            data = {"model": model, "prompt": prompt, "n": str(n), "response_format": "b64_json"}
            if size != "auto":
                data["size"] = size
            if quality != "auto":
                data["quality"] = quality
            if background != "auto":
                data["background"] = background
            files = []
            for idx, b in enumerate(ref_imgs):
                fname = f"image_{idx}.png"
                if len(ref_imgs) == 1:
                    files.append(("image", (fname, b, "image/png")))
                else:
                    files.append(("image[]", (fname, b, "image/png")))
            if mask_bytes:
                files.append(("mask", ("mask.png", mask_bytes, "image/png")))
            try:
                r = session.post(url, headers=headers, data=data, files=files, timeout=timeout)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, timeout)
                if imgs:
                    return self._pack(imgs, f"中转站 | edits OK | {note} | {len(ref_imgs)} 张参考图")
                errors.append(f"edits 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"edits 失败: {e}")
            if api_mode == "edits":
                raise RuntimeError("edits 调用失败 —— " + " ; ".join(errors))

        # 2) generations
        if mode in ("generations", "auto"):
            url = f"{base}/images/generations"
            body = {"model": model, "prompt": prompt, "n": n, "response_format": "b64_json"}
            if size != "auto":
                body["size"] = size
            if quality != "auto":
                body["quality"] = quality
            if background != "auto":
                body["background"] = background
            try:
                r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=timeout)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, timeout)
                if imgs:
                    return self._pack(imgs, f"中转站 | generations OK | {note}")
                errors.append(f"generations 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"generations 失败: {e}")
            if api_mode == "generations" or (api_mode == "auto" and not ref_imgs):
                raise RuntimeError("generations 调用失败 —— " + " ; ".join(errors))

        # 3) chat/completions（多模态，nano-banana 类）
        url = f"{base}/chat/completions"
        content = [{"type": "text", "text": prompt}]
        for b in ref_imgs:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}})
        body = {"model": model, "messages": [{"role": "user", "content": content}], "stream": False}
        try:
            r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                             data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(_parse_error(r, r.text))
            payload = r.json()
            imgs, note = _extract_images_from_response(payload, session, timeout)
            if imgs:
                return self._pack(imgs, f"中转站 | chat OK | {note} | {len(ref_imgs)} 张参考图")
            txt = payload.get("_relay_text", "")
            errors.append("chat 未返回图片" + (f"，模型文本回复: {txt[:200]}" if txt else ""))
        except Exception as e:
            errors.append(f"chat 失败: {e}")

        raise RuntimeError("中转站三种调用方式都没拿到图片 —— " + " ; ".join(errors))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack(img_bytes_list, info):
        tensors = [_bytes_to_tensor(b) for b in img_bytes_list]
        # 尺寸不一致时统一到第一张
        h, w = tensors[0].shape[1], tensors[0].shape[2]
        fixed = []
        for t in tensors:
            if t.shape[1] != h or t.shape[2] != w:
                arr = t[0].cpu().numpy()
                img = Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
                t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None, ...]
            fixed.append(t)
        return (torch.cat(fixed, dim=0), info)


# --------------------------------------------------------------------------- #
# 辅助节点：列出网关支持的模型
# --------------------------------------------------------------------------- #
class RelayModelList:
    """拉取 {base_url}/models，方便确认中转站的模型别名。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://127.0.0.1:8000/v1", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 600, "step": 5}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, base_url, api_key, verify_ssl=True, timeout=30):
        base = (base_url or "").strip().rstrip("/")
        r = requests.get(f"{base}/models",
                         headers={"Authorization": f"Bearer {api_key}", "User-Agent": "ComfyUI-RelayImage/1.0"},
                         verify=bool(verify_ssl), timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(_parse_error(r, r.text))
        data = r.json().get("data", [])
        ids = sorted({d.get("id") for d in data if isinstance(d, dict) and d.get("id")})
        text = "\n".join(ids)
        print(f"[RelayImage] {base}/models 返回 {len(ids)} 个模型:\n{text}")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "RelayImageNode": RelayImageNode,
    "RelayModelList": RelayModelList,
    "RelayChannelRouter": None,   # 由下方定义后回填
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RelayImageNode": "Relay Image (中转站/自定义 OpenAI 图像接口)",
    "RelayModelList": "Relay Model List (列出网关模型)",
    "RelayChannelRouter": "Relay 通道路由 (填了中转key就走中转，否则走官方)",
}


# --------------------------------------------------------------------------- #
# 通道路由：一个节点决定整条工作流走哪条路
# --------------------------------------------------------------------------- #
CHANNEL_CHOICES = ["自动", "官方", "中转站", "本地Qwen"]


class RelayChannelRouter:
    """配置中心：在一处填官方 Key / 中转 Key，其余 7 个统一图像节点自动复用。

    输出：
        use_online        : 接「在线 / 本地」总开关的 switch
        official_api_key  : 接 Relay Image 节点的 official_api_key
        relay_api_key     : 接 Relay Image 节点的 relay_api_key
        relay_base_url    : 接 Relay Image 节点的 relay_base_url
        model             : 接 Relay Image 节点的 model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CHANNEL_CHOICES, {
                    "default": "自动",
                    "tooltip": "自动：填了中转 key 走中转站、填了官方 key 走官方；也可强制指定",
                }),
                "official_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "ComfyUI 平台(platform.comfy.org)的 API Key —— 走官方通道时用",
                }),
                "relay_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "中转站令牌 sk-xxx（如 xianai.cc）—— 走中转站时用",
                }),
                "relay_base_url": ("STRING", {
                    "default": "https://xianai.cc/v1", "multiline": False,
                    "tooltip": "中转站地址，要带 /v1",
                }),
                "model": ("STRING", {
                    "default": "gpt-image-2", "multiline": False,
                    "tooltip": "模型名：官方用 gpt-image-2 / gpt-image-2.5-flare 等；中转站用其支持的别名",
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("use_online", "official_api_key", "relay_api_key", "relay_base_url", "model")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, mode, official_api_key, relay_api_key, relay_base_url, model):
        off = (official_api_key or "").strip()
        rel = (relay_api_key or "").strip()
        base = (relay_base_url or "").strip().rstrip("/")
        mdl = (model or "").strip()

        if mode == "官方":
            use_online, off_out, rel_out = True, off, ""
            where = "官方API(comfy.org)"
            if not off:
                where = "官方API 但未填 Key ⚠"
        elif mode == "中转站":
            use_online, off_out, rel_out = True, "", rel
            where = f"中转站({base})"
            if not rel:
                where = "中转站 但未填 Key ⚠"
        elif mode == "本地Qwen":
            use_online, off_out, rel_out = False, "", ""
            where = "本地Qwen2.1"
        else:  # 自动
            if rel:
                off_out, rel_out = "", rel
                where = f"中转站({base})"
            elif off:
                off_out, rel_out = off, ""
                where = "官方API(comfy.org)"
            else:
                off_out, rel_out = "", ""
                where = "未填任何 Key ⚠（请填 official_api_key 或 relay_api_key）"
            use_online = True

        print(f"[RelayChannelRouter] mode={mode} 官方key={bool(off)} 中转key={bool(rel)} -> 走 {where}")
        return (use_online, off_out, rel_out, base, mdl)


NODE_CLASS_MAPPINGS["RelayChannelRouter"] = RelayChannelRouter

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
