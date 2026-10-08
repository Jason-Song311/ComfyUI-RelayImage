"""ComfyUI-RelayImage —— 用自定义 base_url / api_key 调用 OpenAI 兼容图像接口。"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
