import torch
import numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from transformers import (
    AutoImageProcessor,
    ConvNextModel,
    RobertaTokenizer,
    RobertaModel,
)
import json
import os
import sys
import argparse
from typing import List, Dict, Any


import torch.nn.functional as F
from PIL import Image

# 确保可以从项目根目录导入 `models`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.convnext_model import build_convnext_clip_from_config


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def load_image_as_pixel_values(model, image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.preprocess_images(image)
    return pixel_values


def compute_standard_clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """标准 CLIP InfoNCE 损失：image->text 与 text->image 交叉熵平均。

    适用于任意 batch size（包括 1）。
    """
    batch_size = logits_per_image.shape[0]
    device = logits_per_image.device
    labels = torch.arange(batch_size, device=device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Run ConvNeXtCLIP single-image test")
    parser.add_argument("--with-negs", action="store_true", help="加入若干负样本文本以得到非零的 CLIP 损失")
    args = parser.parse_args()
    # 路径配置（按用户要求）
    config_path = \
        "/media/ps/data-ssd/UltrasoundRAG/CLIP/config/convnext/convnext_config.json"
    image_path = \
        "/media/ps/data-ssd/UltrasoundRAG/CLIP/models/qwen-2.5-vl/甲状腺超声图.jpg"
    texts: List[str] = ["甲状腺超声文本"]
    if args.with_negs:
        # 添加若干明显无关的负样本文本，形成 N>1 的对比空间
        texts += [
            "随机无关文本",
            "天气晴朗，适合户外运动",
            "股票市场今日上涨",
        ]

    # 载入配置与模型
    config = load_config(config_path)
    model = build_convnext_clip_from_config(config)

    # 预处理图像与文本，执行前向
    pixel_values = load_image_as_pixel_values(model, image_path)
    outputs = model(pixel_values=pixel_values, texts=texts)

    # 取出结果
    logits_per_image = outputs["logits_per_image"]  # [B, N]
    logits_per_text = outputs["logits_per_text"]    # [N, B]
    image_features = outputs["image_features"]      # [B, D]
    text_features = outputs["text_features"]        # [N, D]
    # 从模型中读取温度参数以便参考
    try:
        # 避免显式依赖内部实现细节，这里通过输出与反推温度不可靠，直接访问参数
        from models.convnext_model import ConvNeXtCLIPModel  # type: ignore
        if isinstance(model, ConvNeXtCLIPModel):
            logit_scale = model.logit_scale.exp().item()
        else:
            logit_scale = float("nan")
    except Exception:
        logit_scale = float("nan")

    # 相似度（对齐到 [-1,1] 的余弦相似度，不带温度）仅用于参考打印
    cosine_sim = (image_features @ text_features.T).squeeze().item()

    # 标准 CLIP 损失
    loss = compute_standard_clip_loss(logits_per_image, logits_per_text)

    # 打印结果
    print("==== ConvNeXtCLIPModel 单样本测试 ====")
    print(f"图像路径: {image_path}")
    print(f"文本: {texts[0]}")
    print(f"特征维度: image={image_features.shape}, text={text_features.shape}")
    print(f"logits_per_image 形状: {logits_per_image.shape}")
    print(f"logits_per_text 形状: {logits_per_text.shape}")
    print(f"logit_scale(温度的倒数): {logit_scale:.4f}")
    print("logits_per_image:")
    print(logits_per_image.detach().cpu())
    print(f"余弦相似度(未加温度): {cosine_sim:.4f}")
    print(f"CLIP 损失: {loss.item():.6f}")


if __name__ == "__main__":
    main()


