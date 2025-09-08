#!/usr/bin/env python3
import json
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Any


def load_coco_captions(ann_path: str) -> Dict[str, Any]:
    with open(ann_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_imageid_to_filename(coco: Dict[str, Any]) -> Dict[int, str]:
    mapping = {}
    for img in coco.get('images', []):
        mapping[int(img['id'])] = img['file_name']
    return mapping


def build_imageid_to_captions(coco: Dict[str, Any]) -> Dict[int, List[str]]:
    caps = defaultdict(list)
    for ann in coco.get('annotations', []):
        img_id = int(ann['image_id'])
        text = ann.get('caption', '').strip()
        if text:
            caps[img_id].append(text)
    return caps


def convert(
    ann_path: str,
    image_root: str,
    out_path: str,
    mode: str = 'all',  # 'first' | 'all'
    limit: int = 0,
) -> int:
    """
    将 COCO captions 转为当前框架的标准JSON结构（与 demo-test.json 兼容）：
    {
      "DataInfo": {
        "coco_<image_id>[_<idx>]": {
          "data_path": "/abs/path/to/image.jpg",
          "refined_caption": "..."
        },
        ...
      }
    }
    - 绝对路径由 image_root + file_name 构建
    - caption 字段映射为 refined_caption
    - 若 mode=all，则每张图的所有caption均导出；否则仅首条
    """
    coco = load_coco_captions(ann_path)
    id2name = build_imageid_to_filename(coco)
    id2caps = build_imageid_to_captions(coco)

    data_info: Dict[str, Dict[str, Any]] = {}
    num_written = 0

    for img_id, file_name in id2name.items():
        captions = id2caps.get(img_id, [])
        if not captions:
            continue
        abs_path = os.path.join(image_root, file_name)
        if mode == 'first':
            key = f"coco_{img_id}"
            data_info[key] = {
                'data_path': abs_path,
                'refined_caption': captions[0]
            }
            num_written += 1
        else:
            for idx, cap in enumerate(captions):
                key = f"coco_{img_id}_{idx}"
                data_info[key] = {
                    'data_path': abs_path,
                    'refined_caption': cap
                }
                num_written += 1
                if limit > 0 and num_written >= limit:
                    break
        if limit > 0 and num_written >= limit:
            break

    output_obj = { 'DataInfo': data_info }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print(f"写入 {num_written} 条到 {out_path}")
    return num_written


def main():
    parser = argparse.ArgumentParser(description='Convert COCO captions to demo-test.json compatible format for this CLIP framework')
    parser.add_argument('--ann', required=True, help='COCO captions json (e.g., captions_val2017.json)')
    parser.add_argument('--image-root', required=True, help='Image root dir (e.g., .../coco2017/val2017)')
    parser.add_argument('--out', required=True, help='Output json path')
    parser.add_argument('--mode', choices=['first', 'all'], default='first', help='Use first caption per image or all captions')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of output samples (0 for no limit)')
    args = parser.parse_args()

    convert(args.ann, args.image_root, args.out, args.mode, args.limit)


if __name__ == '__main__':
    main()


# python /media/ps/data-ssd/UltrasoundRAG/CLIP/tools/convert_coco_captions.py \
#   --ann /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/annotations/captions_val2017.json \
#   --image-root /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val2017 \
#   --out /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val_demo_format.json \
#   --mode all