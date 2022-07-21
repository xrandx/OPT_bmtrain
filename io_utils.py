# -*- encoding:utf-8 -*-
import torch
import os
import json

def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model


def create_dir(dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_text(src):
    with open(src, "r", encoding="utf-8") as f:
        dst = f.read()
    return dst


def read_json(src):
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, dst):
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def save_jsonl(obj: list, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        obj = "\n".join(obj)
        fh.write(obj)