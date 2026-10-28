"""
Relay Image — 用「自定义 base_url / api_key」调用 OpenAI 兼容的图像接口。

为什么需要它：
    ComfyUI 官方的 OpenAIGPTImageNodeV2 / OpenAIGPTImage1 是 partner API 节点，
    硬编码走 comfy.org 官方代理（隐藏输入 auth_token_comfy_org / api_key_comfy_org），
    节点上没有 base_url 字段，因此无法指向自建API网站（one-api / new-api / 自研网关）。
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
import math
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

# 与官方 partner 节点 Nano Banana Pro / Gemini 完全对齐
ASPECT_RATIO_CHOICES = [
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]
RESOLUTION_CHOICES = ["1K", "2K", "4K"]
# 与官方一致：IMAGE = 只出图；IMAGE+TEXT = 同时返回文字
RESPONSE_MODALITY_CHOICES = ["IMAGE", "IMAGE+TEXT"]
SEED_MAX = 0xFFFFFFFFFFFFFFFF
# 1K/2K/4K -> 基准边长（按「面积恒定」换算，与 Gemini 的档位语义一致）
_RESOLUTION_BASE = {"1K": 1024.0, "2K": 2048.0, "4K": 4096.0}

# 模型下拉框的选项 —— 出图节点与「★ 配置中心」共用同一份，想加模型改这里一处即可
MODEL_CHOICES = [
    "nano-banana-2",         # 默认：API网站实测可用，中文指令跟随最强
    "nano-banana-pro",       # 更高质量档（单张约 175 秒，偏慢）
    "gpt-image-2",           # 备选通道，当前最便宜（¥0.074/张）
    "gpt-image-2.5",         # 备选通道
]
# 2026-10-22 移除下面两个：上游已下架（监控报告 + 实测均为 400「该模型正在修复」）。
# 上游恢复后把对应行加回列表即可。
#   "nano-banana-2-lite",  # 最便宜档（原 ¥0.054/张）
#   "nano-banana-fast",    # 低延迟档（原 ¥0.054/张）

# ---- 以下三项已从节点 UI 上移除，固定为内部默认值（想改行为改这里） ----
# 接口模式：实测 nano-banana 全系只有 chat/completions 能出图，故固定走 chat。
# 代码里 generations / edits 两条路径仍保留（将来上游修好了可把这里改回 "auto"）。
FIXED_API_MODE = "chat"
FIXED_TIMEOUT = 300                            # 秒
DEFAULT_RELAY_BASE = "https://xianai.cc/v1"    # 没连配置中心时的兜底API网站地址


def ratio_to_wh(aspect_ratio, resolution, multiple=8):
    """「宽高比 + 分辨率」-> 'WxH'。auto 或解析失败返回 None。

    换算规则：保持面积恒定（w*h ≈ base²），宽高比 = a:b。
    例：16:9 @1K -> 1368x768；4:3 @1K -> 1184x888；21:9 @2K -> 3136x1344。
    """
    if not aspect_ratio or aspect_ratio == "auto":
        return None
    try:
        a_s, b_s = str(aspect_ratio).split(":")
        a, b = float(a_s), float(b_s)
    except Exception:
        return None
    if a <= 0 or b <= 0:
        return None
    base = _RESOLUTION_BASE.get(str(resolution).upper(), 1024.0)
    w = base * math.sqrt(a / b)
    h = base * math.sqrt(b / a)
    w = max(multiple, int(round(w / multiple)) * multiple)
    h = max(multiple, int(round(h / multiple)) * multiple)
    return f"{w}x{h}"


def size_hint_text(aspect_ratio, resolution):
    """chat 模式不认 size 参数，把尺寸意图写成一句自然语言兜底。

    只在用户显式偏离默认（auto / 1K）时才返回内容，避免默认值污染 prompt。
    """
    ar_on = bool(aspect_ratio) and aspect_ratio != "auto"
    res_on = bool(resolution) and resolution != "1K"
    if not ar_on and not res_on:
        return ""
    bits = []
    if ar_on:
        bits.append(f"画面宽高比 {aspect_ratio}")
    if resolution:
        bits.append(f"分辨率 {resolution}")
    return "、".join(bits)



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
    """统一图像节点：同一个节点里可以填「官方 comfy.org API Key」或「API网站 Key」，
    按「填了哪个」自动决定走官方代理还是走API网站。

    官方通道 = https://api.comfy.org/proxy/openai/images/{generations,edits}（X-API-KEY 认证）
    API网站通道 = {relay_base_url}/images/{generations,edits} 或 {relay_base_url}/chat/completions
    """

    OFFICIAL_BASE = "https://api.comfy.org"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ---------- 前 7 项与官方 Nano Banana Pro 节点逐字对齐 ----------
                "prompt": ("STRING", {
                    "multiline": True, "default": "", "dynamicPrompts": True,
                    "tooltip": "用文字描述要生成的画面或要做的修改。把约束、风格、细节都写进去。",
                }),
                "model": (MODEL_CHOICES, {
                    "default": "nano-banana-2",
                    "tooltip": "模型名（下拉选择，与官方节点同款）。官方通道用 gpt-image-2 等；"
                               "API网站用其别名。要加模型改 nodes.py 里的 MODEL_CHOICES",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": SEED_MAX,
                    "control_after_generate": True,
                    "tooltip": "固定值时会尽量复现同一结果，但不保证完全确定；换模型或改参数仍会变。",
                }),
                "aspect_ratio": (ASPECT_RATIO_CHOICES, {
                    "default": "auto",
                    "tooltip": "输出宽高比。自动：跟随输入图比例；没有输入图时通常出 16:9。",
                }),
                "resolution": (RESOLUTION_CHOICES, {
                    "default": "1K",
                    "tooltip": "目标分辨率 1K/2K/4K，按「面积恒定」换算成具体像素。",
                }),
                "response_modalities": (RESPONSE_MODALITY_CHOICES, {
                    "default": "IMAGE", "advanced": True,
                    "tooltip": "图像：只出图；图像+文本：同时返回模型的文字回复（走 chat 时生效）。",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "", "advanced": True,
                    "tooltip": "决定 AI 行为的基本指令，会作为 system 消息发出去。",
                }),
                "n": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "一次要几张。走 chat 路径时上游通常只回 1 张。",
                }),
            },
            "optional": {
                # 下面三项不显示输入框，只留连线点 —— 由「★ 配置中心」统一喂进来，
                # 这样换 Key / 换地址只需要改配置中心一处。
                "official_api_key": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "官方 comfy.org 的 Key，从「★ 配置中心」连过来",
                }),
                "relay_api_key": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "API网站令牌 sk-xxx，从「★ 配置中心」连过来",
                }),
                "relay_base_url": ("STRING", {
                    "default": DEFAULT_RELAY_BASE, "forceInput": True,
                    "tooltip": "API网站地址（带 /v1），从「★ 配置中心」连过来",
                }),
                "image": ("IMAGE", {"tooltip": "参考图；传 batch 即为多图（编辑 / 融合）"}),
                "mask": ("MASK", {"tooltip": "局部重绘掩码，白色区域会被替换"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    # ------------------------------------------------------------------ #
    def run(self, prompt, model, seed, aspect_ratio, resolution, response_modalities,
            system_prompt, n,
            official_api_key=None, relay_api_key=None, relay_base_url=None,
            image=None, mask=None):

        off_key = (official_api_key or "").strip()
        rel_key = (relay_api_key or "").strip()
        base = (relay_base_url or "").strip().rstrip("/") or DEFAULT_RELAY_BASE

        # 走哪条通道：按「拿到哪个 Key」自动判断（API网站 Key 优先，与配置中心默认 mode 一致）
        if rel_key:
            use_official = False
        elif off_key:
            use_official = True
        else:
            raise ValueError(
                "没有拿到 API Key —— 请把「★ 配置中心」的 relay_api_key（或 official_api_key）"
                "连到本节点左侧对应的输入点")

        if not prompt and image is None:
            raise ValueError("prompt 与 image 至少要有一个")

        # 「宽高比 + 分辨率」-> 具体 size（auto 时返回 None，即不发送）
        size_spec = ratio_to_wh(aspect_ratio, resolution)

        session = requests.Session()
        session.verify = True

        # 参考图 -> png bytes
        ref_imgs = []
        if image is not None:
            for i in range(image.shape[0]):
                ref_imgs.append(_tensor_to_png_bytes(image[i])[0])
        mask_bytes = _mask_to_png_bytes(mask) if mask is not None else None

        if use_official:
            return self._run_official(session, off_key, model, prompt, size_spec,
                                      n, ref_imgs, mask_bytes,
                                      seed, response_modalities, system_prompt)
        return self._run_relay(session, rel_key, base, model, prompt, size_spec,
                               n, ref_imgs, mask_bytes,
                               aspect_ratio, resolution, seed, response_modalities, system_prompt)

    # ------------------------------------------------------------------ #
    def _run_official(self, session, api_key, model, prompt, size, n, ref_imgs, mask_bytes,
                      seed=42, response_modalities="IMAGE", system_prompt=""):
        """官方 comfy.org 代理——接口与官方 OpenAIGPTImage* 节点完全一致。

        注意：system_prompt / response_modalities 是 Gemini 侧概念，
        官方 OpenAI 代理不认，这里不发送（只在走API网站 chat 时生效）。
        """
        headers = {"X-API-KEY": api_key, "User-Agent": "ComfyUI-RelayImage/1.0"}
        common = {"model": model, "prompt": prompt, "n": n, "moderation": "low", "seed": seed}
        if size:
            common["size"] = size

        errors = []
        # 官方代理只有 generations / edits 两条路：有参考图就走 edits
        want_edits = bool(ref_imgs)

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
                r = session.post(url, headers=headers, data=data, files=files, timeout=FIXED_TIMEOUT)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, FIXED_TIMEOUT)
                if imgs:
                    return self._pack(imgs, f"官方API | edits OK | {note} | {len(ref_imgs)} 张参考图")
                errors.append(f"官方 edits 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"官方 edits 失败: {e}")
            if want_edits:
                raise RuntimeError("官方 edits 调用失败 —— " + " ; ".join(errors))

        # 文生图
        url = f"{self.OFFICIAL_BASE}/proxy/openai/images/generations"
        try:
            r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                             data=json.dumps(common, ensure_ascii=False).encode("utf-8"),
                             timeout=FIXED_TIMEOUT)
            if r.status_code >= 400:
                raise RuntimeError(_parse_error(r, r.text))
            imgs, note = _extract_images_from_response(r.json(), session, FIXED_TIMEOUT)
            if imgs:
                return self._pack(imgs, f"官方API | generations OK | {note}")
            errors.append(f"官方 generations 未返回图片: {r.text[:200]}")
        except Exception as e:
            errors.append(f"官方 generations 失败: {e}")

        raise RuntimeError("官方通道调用失败 —— " + " ; ".join(errors)
                           + "\n（检查 official_api_key 是否正确、平台是否有余额）")

    # ------------------------------------------------------------------ #
    def _run_relay(self, session, api_key, base, model, prompt, size, n, ref_imgs, mask_bytes,
                   aspect_ratio="auto", resolution="1K", seed=42,
                   response_modalities="IMAGE", system_prompt=""):
        if not base:
            raise ValueError("走API网站时 relay_base_url 不能为空，例如 https://xianai.cc/v1")

        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "ComfyUI-RelayImage/1.0"}
        if api_key:
            headers["x-api-key"] = api_key

        # 接口模式已固定（见文件顶部 FIXED_API_MODE）；下面三条路径的代码保留，
        # 将来上游修好 generations / edits 时把 FIXED_API_MODE 改回 "auto" 即可复活。
        mode = FIXED_API_MODE
        if mode == "auto":
            mode = "generations" if not ref_imgs else "edits"

        errors = []

        # 1) edits（multipart）
        if mode == "edits":
            if not ref_imgs:
                raise ValueError("接口模式=edits 需要接一张参考图")
            url = f"{base}/images/edits"
            data = {"model": model, "prompt": prompt, "n": str(n), "response_format": "b64_json"}
            if size:
                data["size"] = size
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
                r = session.post(url, headers=headers, data=data, files=files, timeout=FIXED_TIMEOUT)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, FIXED_TIMEOUT)
                if imgs:
                    return self._pack(imgs, f"API网站 | edits OK | {note} | {len(ref_imgs)} 张参考图")
                errors.append(f"edits 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"edits 失败: {e}")
            if mode == "edits":
                raise RuntimeError("edits 调用失败 —— " + " ; ".join(errors))

        # 2) generations
        if mode in ("generations", "auto"):
            url = f"{base}/images/generations"
            body = {"model": model, "prompt": prompt, "n": n, "response_format": "b64_json"}
            if size:
                body["size"] = size
            try:
                r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=FIXED_TIMEOUT)
                if r.status_code >= 400:
                    raise RuntimeError(_parse_error(r, r.text))
                imgs, note = _extract_images_from_response(r.json(), session, FIXED_TIMEOUT)
                if imgs:
                    return self._pack(imgs, f"API网站 | generations OK | {note}")
                errors.append(f"generations 未返回图片: {r.text[:200]}")
            except Exception as e:
                errors.append(f"generations 失败: {e}")
            if mode == "generations" or (mode == "auto" and not ref_imgs):
                raise RuntimeError("generations 调用失败 —— " + " ; ".join(errors))

        # 3) chat/completions（多模态，nano-banana 类）
        #    实测 chat 路径不吃 size 字段，所以这里两手都做：
        #    a) 把宽高比/分辨率/响应模态当字段一起发（部分网关会透传给上游）
        #    b) 把尺寸意图写成一句自然语言附在 prompt 末尾（chat 下唯一确定有影响的手段）
        url = f"{base}/chat/completions"
        hint = size_hint_text(aspect_ratio, resolution)
        chat_prompt = f"{prompt}\n\n【输出规格】{hint}。" if hint else prompt
        content = [{"type": "text", "text": chat_prompt}]
        for b in ref_imgs:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}})
        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        body = {"model": model, "messages": messages, "stream": False, "seed": seed}
        # 只在用户显式偏离默认时才附带这些字段，降低上游不认识它们的风险
        if aspect_ratio and aspect_ratio != "auto":
            body["aspect_ratio"] = aspect_ratio
        if resolution and resolution != "1K":
            body["resolution"] = resolution
        if response_modalities and response_modalities != "IMAGE":
            body["response_modalities"] = response_modalities

        extras = [k for k in ("seed", "aspect_ratio", "resolution", "response_modalities") if k in body]
        degraded = ""
        last_err = "chat 未返回图片"
        for attempt in (0, 1):
            try:
                r = session.post(url, headers={**headers, "Content-Type": "application/json"},
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=FIXED_TIMEOUT)
                if r.status_code >= 400:
                    last_err = _parse_error(r, r.text)
                    # 第一次带扩展字段被拒 -> 去掉它们重试一次（宁可少参数，也别让整条链路挂掉）
                    if attempt == 0 and extras:
                        degraded = f" | 已降级：上游拒绝 {','.join(extras)}"
                        for k in extras:
                            body.pop(k, None)
                        extras = []
                        continue
                    break
                payload = r.json()
                imgs, note = _extract_images_from_response(payload, session, FIXED_TIMEOUT)
                if imgs:
                    spec = f" | 规格 {aspect_ratio}@{resolution}" if hint else ""
                    return self._pack(imgs, f"API网站 | chat OK | {note} | {len(ref_imgs)} 张参考图{spec}{degraded}")
                txt = payload.get("_relay_text", "")
                last_err = "chat 未返回图片" + (f"，模型文本回复: {txt[:200]}" if txt else "")
                break
            except Exception as e:
                last_err = f"chat 失败: {e}"
                break
        errors.append(last_err)

        raise RuntimeError("API网站三种调用方式都没拿到图片 —— " + " ; ".join(errors))

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
    """拉取 {base_url}/models，方便确认API网站的模型别名。"""

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
    "RelayImageNode": "Relay Image (API网站/自定义 OpenAI 图像接口)",
    "RelayModelList": "Relay Model List (列出网关模型)",
    "RelayChannelRouter": "Relay 通道路由 (填了API网站key就走API网站，否则走官方)",
}


# --------------------------------------------------------------------------- #
# 通道路由：一个节点决定整条工作流走哪条路
# --------------------------------------------------------------------------- #
CHANNEL_CHOICES = ["自动", "官方", "API网站", "本地Qwen"]


class RelayChannelRouter:
    """配置中心：在一处填官方 Key / API网站 Key，其余 7 个统一图像节点自动复用。

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
                    "tooltip": "自动：填了API网站 key 走API网站、填了官方 key 走官方；也可强制指定",
                }),
                "official_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "ComfyUI 平台(platform.comfy.org)的 API Key —— 走官方通道时用",
                }),
                "relay_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "API网站令牌 sk-xxx（如 xianai.cc）—— 走API网站时用",
                }),
                "relay_base_url": ("STRING", {
                    "default": "https://xianai.cc/v1", "multiline": False,
                    "tooltip": "API网站地址，要带 /v1",
                }),
                "model": (MODEL_CHOICES, {
                    "default": "nano-banana-2",
                    "tooltip": "模型名（下拉选择，与出图节点共用同一份列表）",
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "COMBO")
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
        elif mode == "API网站":
            use_online, off_out, rel_out = True, "", rel
            where = f"API网站({base})"
            if not rel:
                where = "API网站 但未填 Key ⚠"
        elif mode == "本地Qwen":
            use_online, off_out, rel_out = False, "", ""
            where = "本地Qwen2.1"
        else:  # 自动
            if rel:
                off_out, rel_out = "", rel
                where = f"API网站({base})"
            elif off:
                off_out, rel_out = off, ""
                where = "官方API(comfy.org)"
            else:
                off_out, rel_out = "", ""
                where = "未填任何 Key ⚠（请填 official_api_key 或 relay_api_key）"
            use_online = True

        print(f"[RelayChannelRouter] mode={mode} 官方key={bool(off)} API网站key={bool(rel)} -> 走 {where}")
        return (use_online, off_out, rel_out, base, mdl)


NODE_CLASS_MAPPINGS["RelayChannelRouter"] = RelayChannelRouter

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
