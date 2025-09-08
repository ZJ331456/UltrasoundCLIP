from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import json
path = "/media/ps/data-ssd/DolphinV1/dolphinV1_3B"
# default: Load the  model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    path, torch_dtype=torch.float16, device_map="cuda"
)

processor = AutoProcessor.from_pretrained(path)


# 用 processor 处理，得到 pixel_values 和 grid_thw
# 图片绝对路径
img_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/models/qwen-2.5-vl/甲状腺超声图.jpg"

# 生成 PIL.Image 列表
pil_images = [Image.open(img_path).convert("RGB")]
dummy_text = "<|vision_start|><|image_pad|><|vision_end|>"
inputs = processor(
    text=[dummy_text] * len(pil_images),
    images=pil_images,
    return_tensors="pt"
).to(model.device, torch.float16)

# 4. 直接过视觉编码器
vision_encoder = model.model.visual

# print(f"inputs:{inputs}\ninputs.input_ids.shape:{inputs.input_ids.shape}\ninputs.attention_mask.shape:{inputs.attention_mask.shape}\ninputs.pixel_values.shape:{inputs.pixel_values.shape}\ninputs.image_grid_thw.shape:{inputs.image_grid_thw.shape}\n")

# print("已保存 inputs.txt，可以完整查看 BatchEncoding 内容")
"""
在 inputs 中，有这几个值
'input_ids'
'attention_mask'
'pixel_values'
'image_grid_thw

'"""

# 提取图像特征
with torch.no_grad():
    image_features = vision_encoder(inputs.pixel_values, inputs.image_grid_thw)
print(f"image_features:{image_features}\nimage_features.shape:{image_features.shape}\n")

        