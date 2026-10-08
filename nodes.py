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
    """调用 OpenAI 兼容图像接口（中转站 / one-api / new-api / 自建网关）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8000/v1",
                    "multiline": False,
                    "tooltip": "中转站的 OpenAI 兼容根地址，要带 /v1。例如 http://127.0.0.1:8000/v1",
                }),
                "api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "中转站令牌，例如 sk-relay-xxxxxx。只存在本机工作流里，不会上传。",
                }),
                "model": ("STRING", {
                    "default": "gpt-image-2", "multiline": False,
                    "tooltip": "中转站支持的模型别名，如 gpt-image-2 / nano-banana-2 / nano-banana-pro",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "api_mode": (MODE_CHOICES, {
                    "default": "auto",
                    "tooltip": "auto：无参考图走 generations、有图走 edits（失败自动降级 chat）；也可手动锁定",
                }),
                "size": (SIZE_CHOICES, {"default": "auto"}),
                "quality": (QUALITY_CHOICES, {"default": "auto"}),
                "background": (BACKGROUND_CHOICES, {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1,
                              "tooltip": "生成几张。注意多数网关只返回第 1 张"}),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 10,
                                    "tooltip": "单次请求超时秒数"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "参考图；传 batch 即为多图（编辑 / 融合）"}),
                "mask": ("MASK", {"tooltip": "局部重绘掩码，白色区域会被替换（仅 edits 模式）"}),
                "verify_ssl": ("BOOLEAN", {"default": True, "tooltip": "本地 http 网关可不管；https 自签证书时关掉"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    # ------------------------------------------------------------------ #
    def run(self, base_url, api_key, model, prompt, api_mode, size, quality,
            background, n, timeout, image=None, mask=None, verify_ssl=True):

        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("base_url 不能为空，例如 http://127.0.0.1:8000/v1")
        if not prompt and image is None:
            raise ValueError("prompt 与 image 至少要有一个")

        session = requests.Session()
        session.verify = bool(verify_ssl)
        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "ComfyUI-RelayImage/1.0"}
        if api_key:
            headers["x-api-key"] = api_key  # 有些网关用这个头

        # 参考图 -> png bytes 列表
        ref_imgs = []
        if image is not None:
            for i in range(image.shape[0]):
                ref_imgs.append(_tensor_to_png_bytes(image[i])[0])
        mask_bytes = _mask_to_png_bytes(mask) if mask is not None else None

        mode = api_mode
        if mode == "auto":
            mode = "generations" if not ref_imgs else "edits"

        errors = []

        # ---------------- 1) edits（multipart） ---------------- #
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
                    return self._pack(imgs, f"edits OK | {note} | {len(ref_imgs)} 张参考图")
                errors.append(f"edits 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"edits 失败: {e}")
            if api_mode == "edits":
                raise RuntimeError("edits 调用失败 —— " + " ; ".join(errors))

        # ---------------- 2) generations（json 文生图） ---------------- #
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
                    return self._pack(imgs, f"generations OK | {note}")
                errors.append(f"generations 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"generations 失败: {e}")
            if api_mode == "generations" or (api_mode == "auto" and not ref_imgs):
                raise RuntimeError("generations 调用失败 —— " + " ; ".join(errors))

        # ---------------- 3) chat/completions（多模态，nano-banana 类） ---------------- #
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
                return self._pack(imgs, f"chat OK | {note} | {len(ref_imgs)} 张参考图")
            txt = payload.get("_relay_text", "")
            errors.append("chat 未返回图片" + (f"，模型文本回复: {txt[:200]}" if txt else ""))
        except Exception as e:
            errors.append(f"chat 失败: {e}")

        raise RuntimeError("三种调用方式都没拿到图片 —— " + " ; ".join(errors))

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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RelayImageNode": "Relay Image (中转站/自定义 OpenAI 图像接口)",
    "RelayModelList": "Relay Model List (列出网关模型)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
