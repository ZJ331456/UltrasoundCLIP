from transformers import Qwen2_5_VLForConditionalGeneration, AutoImageProcessor
import torch
from PIL import Image
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import json
model_path = "/media/ps/data-ssd/DolphinV1/dolphinV1_3B"
img_path1 = "/media/ps/data-ssd/Pascal/frame/mixture0001/case000001.png"
img_path2 = "/media/ps/data-ssd/Pascal/frame/mixture0012/case000078.png"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="cuda",
    trust_remote_code=True
)
def test1():
    # 1. 手动创建视觉配置（或从预训练模型中加载）
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_path,
    #     torch_dtype="auto",
    #     trust_remote_code=True
    #     )

    # 2. 加载视觉模型
    vision_model = model.visual

    # 3. 加载图像处理器
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 4. 加载并预处理图像
    image = Image.open(img_path1).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
    pixel_values = inputs["pixel_values"]  # [B, C, H, W]
    image_grid_thw = inputs["image_grid_thw"]  # [B, 3]

    # print("输入 shape:", inputs["pixel_values"].shape)
    # # 5. 获取图像特征
    with torch.no_grad():
        # 注意：vision_model 的 forward 需要 grid_thw 参数
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        # print("a_image_features:", image_features)
        ####将同一批次里面的所有图片都转化成[1,2048]的形式
        image_embeds = []
        for feat in image_features:
            embed = feat.mean(dim=0, keepdim=True)  # [1, 2048]
            image_embeds.append(embed)

    #     image_embeds = torch.cat(image_embeds, dim=0)  # [16, 2048]
    #     ###一次性全部投影
    #     vision_proj = nn.Linear(2048, 768).to(image_embeds.device, image_embeds.dtype)
    #     image_embeds_768 = vision_proj(image_embeds)  # [16, 768]

    # # image_features 是一个 list，每个元素是 [num_patches, dim]
    # print(f"图像数量：{len(image_features)}")  # 输出图像的数量
    # print("图像特征 shape:", image_features[0].shape)
    # print("图像特征:", image_embeds[0].shape)
    # print("线性转化之后",image_embeds_768[0].shape)
    # print("图像特征维度:", image_features)
    return image_features, image_embeds
def test2():
    # default: Load the  model on the available device(s)

    processor = AutoProcessor.from_pretrained(model_path)


    # 用 processor 处理，得到 pixel_values 和 grid_thw
    # 图片绝对路径


    # 生成 PIL.Image 列表
    pil_images = [Image.open(img_path2).convert("RGB")]
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
        #  1. 拿到整批的patch特征
        all_feats = vision_encoder(inputs.pixel_values, inputs.image_grid_thw)
         # 2. 计算每张图的 patch 数量并拆分
        patch_nums = (
            inputs.image_grid_thw.prod(dim=1) // (vision_encoder.spatial_merge_size ** 2)
        ).tolist()
        image_features = torch.split(all_feats, patch_nums, dim=0)  # list[Tensor]

        ####将同一批次里面的所有图片都转化成[1,2048]的形式
        image_embeds = []
        for feat in image_features:
            embed = feat.mean(dim=0, keepdim=True)  # [1, 2048]
            image_embeds.append(embed)
    # print(f"image_features:{image_features}\nimage_features.shape:{image_features.shape}\n")
    return image_features, image_embeds 

from scipy.spatial.distance import cosine

# # 假设 feature_a 和 feature_b 是从两种方法中提取的特征向量
a, feature_a = test1()
b, feature_b = test2()
# feature_a = feature_a[0].to(torch.float32).cpu().numpy()
# feature_b = feature_b[0].to(torch.float32).cpu().numpy()
# 先拉成一维
vec_a = feature_a[0].to(torch.float32).cpu().numpy().squeeze()   # shape: (2048,)
vec_b = feature_b[0].to(torch.float32).cpu().numpy().squeeze()   # shape: (2048,)

# 再计算
sim = 1 - cosine(vec_a, vec_b)
print("Cosine Similarity:", sim)
# print(f"a:{len(a[0].shape)}\nb:{len(b[0].shape)}\n")
# # print(f"feature_a.shape:{feature_a[0].shape}\nfeature_b.shape:{feature_b[0].shape}\n")
# print("a:", a)
# print("feature_b:", feature_b)
# print(f"a:{len(a)}\nb:{len(feature_b)}\n")
# print("feature_a:", feature_a[0][:10])  # 打印前 10 个值
# print("feature_b:", feature_b[0][:10])
# print("similarity:", cosine(feature_a, feature_b))
# similarity = 1 - cosine(feature_a[0], feature_b[0])
# print(f"Cosine Similarity: {similarity}")
