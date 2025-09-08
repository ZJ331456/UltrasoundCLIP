#!/usr/bin/env python3
import json
import os
import argparse
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def load_datainfo_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'DataInfo' not in data or not isinstance(data['DataInfo'], dict):
        raise ValueError('输入 JSON 必须为 {"DataInfo": {...}} 结构')
    return data['DataInfo']


def group_by_image_path(data_info: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    groups = defaultdict(list)
    for key, item in data_info.items():
        image_path = item.get('data_path')
        if not image_path:
            # 跳过无效项
            continue
        groups[image_path].append((key, item))
    return groups


def split_indices(n: int, ratios: Tuple[float, float, float], seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    r_train, r_val, r_test = ratios
    t_end = int(n * r_train)
    v_end = t_end + int(n * r_val)
    train_idx = idx[:t_end]
    val_idx = idx[t_end:v_end]
    test_idx = idx[v_end:]
    return train_idx, val_idx, test_idx


def build_datainfo(subset_items: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    # 直接保留原 key，确保唯一性和可追踪
    return {key: item for key, item in subset_items}


def main():
    parser = argparse.ArgumentParser(description='按 8:1:1 划分 {"DataInfo": {...}} JSON，避免同图像跨集泄漏（按 data_path 分组）')
    parser.add_argument('--input', required=True, help='输入 JSON（包含 DataInfo）')
    parser.add_argument('--out-dir', required=True, help='输出目录（将创建 val_split 子目录）')
    parser.add_argument('--train-name', default='coco_val_train.json', help='训练集文件名')
    parser.add_argument('--val-name', default='coco_val_val.json', help='验证集文件名')
    parser.add_argument('--test-name', default='coco_val_test.json', help='测试集文件名')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--ratios', default='0.8,0.1,0.1', help='划分比例，格式: train,val,test')
    args = parser.parse_args()

    r = tuple(float(x) for x in args.ratios.split(','))
    assert abs(sum(r) - 1.0) < 1e-6, 'ratios 之和必须为 1.0'

    data_info = load_datainfo_json(args.input)
    groups = group_by_image_path(data_info)
    image_paths = list(groups.keys())
    n = len(image_paths)
    train_idx, val_idx, test_idx = split_indices(n, r, seed=args.seed)

    def collect(indices: List[int]) -> List[Tuple[str, Dict[str, Any]]]:
        items: List[Tuple[str, Dict[str, Any]]] = []
        for i in indices:
            items.extend(groups[image_paths[i]])
        return items

    train_items = collect(train_idx)
    val_items = collect(val_idx)
    test_items = collect(test_idx)

    out_root = os.path.join(args.out_dir, 'val_split')
    os.makedirs(out_root, exist_ok=True)

    outputs = [
        (args.train_name, train_items),
        (args.val_name, val_items),
        (args.test_name, test_items),
    ]

    for name, items in outputs:
        outp = os.path.join(out_root, name)
        obj = {'DataInfo': build_datainfo(items)}
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f'写入 {len(items)} 条 -> {outp}')

    print(f'总图像数(去重): {n}  |  训练/验证/测试(样本条目): {len(train_items)}/{len(val_items)}/{len(test_items)}')


if __name__ == '__main__':
    main()



# (clip) zhoujun@ds-4xA6k:/media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017$ python /media/ps/data-ssd/UltrasoundRAG/CLIP/tools/split_json_by_ratio.py \
#   --input /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val_demo_format.json \
#   --out-dir /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017
# 写入 20011 条 -> /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val_split/coco_val_train.json
# 写入 2502 条 -> /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val_split/coco_val_val.json
# 写入 2501 条 -> /media/ps/data-ssd/UltrasoundRAG/CLIP/dataset/coco2017/val_split/coco_val_test.json
# 总图像数(去重): 5000  |  训练/验证/测试(样本条目): 20011/2502/2501


