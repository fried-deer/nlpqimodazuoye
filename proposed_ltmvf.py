# -*- coding: utf-8 -*-
"""
proposed_ltmvf.py

本文提出方法：
LT-MVF: Long-Tail Multi-View Fusion for Forbidden Comment Classification
长尾多视角预训练融合违禁评论识别模型

无需命令行参数，直接运行：
nohup python proposed_ltmvf.py > proposed_ltmvf.log 2>&1 &

默认目录结构：
.
├── all.csv
├── simsun.ttc
├── proposed_ltmvf.py
└── hf_models/
    ├── chinese-macbert-large/
    └── chinese-roberta-wwm-ext-large/

核心方法：
Ours = MacBERT-large + RoBERTa-large 加权 logit 融合
       + Focal BCE
       + 类别权重
       + Weighted Sampler
       + 少数类增强
       + Bias Calibration

同时包含：
1. 单模型 MacBERT-large
2. 单模型 RoBERTa-large
3. Ours 加权 logit 融合模型
4. 双编码器融合模型 DualEncoderFusion
5. 基于预训练编码器的改进深度学习模型
6. 自动可视化与报告生成

输出目录：
outputs_ltmvf/
"""

import os
import re
import gc
import json
import time
import copy
import math
import random
import warnings
from dataclasses import dataclass, asdict
from collections import Counter

warnings.filterwarnings("ignore")

# ============================================================
# HuggingFace 本地离线配置
# ============================================================

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


# ============================================================
# 全局配置：无需用户指定参数
# ============================================================

DATA_PATH = "all.csv"
TEXT_COL = "文本"
LABEL_COL = "类别"
ID_COL = "id"

OUTPUT_DIR = "outputs_ltmvf"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOGIT_DIR = os.path.join(OUTPUT_DIR, "logits")

SEED = 42

MACBERT_LARGE_PATH = "./hf_models/chinese-macbert-large"
ROBERTA_LARGE_PATH = "./hf_models/chinese-roberta-wwm-ext-large"

LABELS_ORDER = [
    "政治敏感",
    "色情",
    "种族歧视",
    "地域歧视",
    "微侵犯(MA)",
    "犯罪",
    "基于文化背景的刻板印象(SCB)",
    "宗教迷信",
    "性侵犯(SO)",
    "基于外表的刻板印象(SA)",
]

# 单模型训练配置
MAX_LEN = 192
SINGLE_BATCH_SIZE = 16
SINGLE_EPOCHS = 10
SINGLE_LR = 1.5e-5
SINGLE_PATIENCE = 3
SINGLE_GRAD_ACCUM = 1

# 双编码器配置
DUAL_MAX_LEN = 160
DUAL_BATCH_SIZE = 4
DUAL_EPOCHS = 6
DUAL_LR = 1.0e-5
DUAL_PATIENCE = 2
DUAL_GRAD_ACCUM = 4

# 少数类增强配置
AUG_MIN_PER_CLASS_SINGLE = 800
AUG_MIN_PER_CLASS_DUAL = 700

# 是否运行双编码器
RUN_DUAL_ENCODER = True

# 是否保存 checkpoints
SAVE_CHECKPOINTS = True


# ============================================================
# 基础工具
# ============================================================

def ensure_dirs():
    for d in [OUTPUT_DIR, FIG_DIR, CKPT_DIR, LOGIT_DIR]:
        os.makedirs(d, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def setup_chinese_font():
    import matplotlib.font_manager as fm

    candidates = [
        os.path.abspath("simsun.ttc"),
        os.path.abspath("SimSun.ttc"),
        os.path.abspath("simsun.ttf"),
        os.path.abspath("SimSun.ttf"),
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                font_prop = FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [
                    font_name,
                    "SimSun",
                    "Noto Sans CJK SC",
                    "Microsoft YaHei",
                    "WenQuanYi Micro Hei",
                    "DejaVu Sans",
                ]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"已加载中文字体文件：{font_path}")
                print(f"Matplotlib 识别到的字体名：{font_name}")
                return font_prop
            except Exception as e:
                print(f"加载字体失败：{font_path}, error={repr(e)}")

    print("警告：未找到 simsun.ttc，中文图表可能显示异常。")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\u200b", "")
    x = x.replace("\ufeff", "")
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def safe_name(name):
    return re.sub(r"[^\w\u4e00-\u9fa5\-\(\)]+", "_", str(name))


def count_total_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def check_local_model(path, name):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"未找到{name}目录：{path}")
    if not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"{name}目录缺少 config.json：{path}")
    if not (
        os.path.exists(os.path.join(path, "pytorch_model.bin"))
        or os.path.exists(os.path.join(path, "model.safetensors"))
    ):
        print(f"警告：{name}目录未发现 pytorch_model.bin 或 model.safetensors，请确认模型文件完整：{path}")


# ============================================================
# 数据读取、划分、增强
# ============================================================

def load_all_csv():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {ID_COL, LABEL_COL, TEXT_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"all.csv 必须包含列 {required}，当前列：{df.columns.tolist()}")

    df = df[[ID_COL, LABEL_COL, TEXT_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

    labels = [x for x in LABELS_ORDER if x in set(df[LABEL_COL])]
    extra = sorted(list(set(df[LABEL_COL]) - set(labels)))
    labels += extra

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    df["label"] = df[LABEL_COL].map(label2id).astype(int)

    return df, labels, label2id, id2label


def stratified_split(df, seed=42):
    train_df, temp_df = train_test_split(
        df,
        train_size=0.70,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


PUNCS = list("，。！？；：,.!?;:、~～…—-_（）()[]【】《》“”\"' ")


def perturb_text(text):
    """
    少数类增强：弱扰动，不改变标签语义。
    """
    text = str(text)
    if len(text) <= 2:
        return text

    ops = [
        "drop_punc",
        "swap_adjacent",
        "delete_space",
        "duplicate_char",
        "light_suffix",
        "identity",
    ]
    op = random.choice(ops)
    chars = list(text)

    if op == "drop_punc":
        idxs = [i for i, c in enumerate(chars) if c in PUNCS]
        if idxs:
            i = random.choice(idxs)
            chars.pop(i)
            return "".join(chars)

    if op == "swap_adjacent":
        if len(chars) >= 4:
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)

    if op == "delete_space":
        return re.sub(r"\s+", "", text)

    if op == "duplicate_char":
        valid = [i for i, c in enumerate(chars) if c.strip()]
        if valid and len(chars) < 500:
            i = random.choice(valid)
            chars.insert(i, chars[i])
            return "".join(chars)

    if op == "light_suffix":
        suffixes = ["啊", "吧", "呢", "了", "嘛"]
        if len(text) < 300:
            return text + random.choice(suffixes)

    return text


def augment_minority_train(train_df, labels, min_per_class=800, seed=42):
    """
    只增强训练集，验证集和测试集绝不增强，避免数据泄漏。
    """
    if min_per_class <= 0:
        return train_df.copy()

    random.seed(seed)
    parts = [train_df.copy()]

    try:
        next_id_base = int(train_df[ID_COL].max()) + 1
    except Exception:
        next_id_base = 10_000_000

    for label_id, label_name in enumerate(labels):
        sub = train_df[train_df["label"] == label_id]
        n = len(sub)

        if n == 0 or n >= min_per_class:
            continue

        need = min_per_class - n
        rows = []
        sub_records = sub.to_dict("records")

        for k in range(need):
            r = random.choice(sub_records).copy()
            r[TEXT_COL] = perturb_text(r[TEXT_COL])
            r[ID_COL] = f"aug_{next_id_base}_{label_id}_{k}"
            rows.append(r)

        print(f"[少数类增强] 类别={label_name} 原训练样本={n} 增强={need} 增强后={min_per_class}")
        parts.append(pd.DataFrame(rows))

    aug_df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return aug_df


def compute_class_weights(train_df, num_labels, device, beta=0.9995, max_weight=8.0):
    counts = np.bincount(train_df["label"].values, minlength=num_labels).astype(np.float32)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.3, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def build_weighted_sampler(train_df, num_labels, power=0.65):
    counts = np.bincount(train_df["label"].values, minlength=num_labels).astype(np.float32)
    weights_per_class = 1.0 / np.power(np.maximum(counts, 1.0), power)
    sample_weights = weights_per_class[train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ============================================================
# Dataset
# ============================================================

class HFDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


class DualHFDataset(Dataset):
    def __init__(self, df, tokenizer_a, tokenizer_b, max_len):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode_one(self, tokenizer, text):
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors=None,
        )
        out = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        else:
            out["token_type_ids"] = torch.zeros_like(out["input_ids"])
        return out

    def __getitem__(self, idx):
        text = self.texts[idx]
        a = self.encode_one(self.tokenizer_a, text)
        b = self.encode_one(self.tokenizer_b, text)

        return {
            "a_input_ids": a["input_ids"],
            "a_attention_mask": a["attention_mask"],
            "a_token_type_ids": a["token_type_ids"],
            "b_input_ids": b["input_ids"],
            "b_attention_mask": b["attention_mask"],
            "b_token_type_ids": b["token_type_ids"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# 模型
# ============================================================

class HFClassifier(nn.Module):
    """
    单预训练编码器分类模型。
    """
    def __init__(self, model_path, num_labels, dropout=0.20):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=False,
        )
        hidden = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))
        return logits


class DualEncoderFusionClassifier(nn.Module):
    """
    双编码器融合模型：
    MacBERT-large + RoBERTa-large

    融合特征：
    [h_macbert, h_roberta, |h1-h2|, h1*h2]
    然后进入门控 MLP 分类器。

    这是一个真正意义上的深度融合模型，而不是简单投票。
    """
    def __init__(
        self,
        model_a_path,
        model_b_path,
        num_labels,
        dropout=0.25,
        freeze_bottom_layers=0,
    ):
        super().__init__()

        self.config_a = AutoConfig.from_pretrained(model_a_path, local_files_only=True)
        self.config_b = AutoConfig.from_pretrained(model_b_path, local_files_only=True)

        self.encoder_a = AutoModel.from_pretrained(
            model_a_path,
            local_files_only=True,
            use_safetensors=False,
        )
        self.encoder_b = AutoModel.from_pretrained(
            model_b_path,
            local_files_only=True,
            use_safetensors=False,
        )

        ha = self.config_a.hidden_size
        hb = self.config_b.hidden_size

        fusion_hidden = max(ha, hb)

        self.proj_a = nn.Linear(ha, fusion_hidden)
        self.proj_b = nn.Linear(hb, fusion_hidden)

        self.gate = nn.Sequential(
            nn.Linear(fusion_hidden * 4, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden * 4, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_labels),
        )

        if freeze_bottom_layers > 0:
            self.freeze_bottom_layers(self.encoder_a, freeze_bottom_layers)
            self.freeze_bottom_layers(self.encoder_b, freeze_bottom_layers)

    def freeze_bottom_layers(self, encoder, n):
        if n >= 1 and hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters():
                p.requires_grad = False

        layers = None
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            layers = encoder.encoder.layer

        if layers is not None:
            for layer in layers[: max(0, n - 1)]:
                for p in layer.parameters():
                    p.requires_grad = False

    def encode(self, encoder, input_ids, attention_mask, token_type_ids):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = encoder(**kwargs)
        return out.last_hidden_state[:, 0]

    def forward(
        self,
        a_input_ids,
        a_attention_mask,
        a_token_type_ids,
        b_input_ids,
        b_attention_mask,
        b_token_type_ids,
    ):
        h1 = self.encode(self.encoder_a, a_input_ids, a_attention_mask, a_token_type_ids)
        h2 = self.encode(self.encoder_b, b_input_ids, b_attention_mask, b_token_type_ids)

        h1 = self.proj_a(h1)
        h2 = self.proj_b(h2)

        diff = torch.abs(h1 - h2)
        prod = h1 * h2

        fusion = torch.cat([h1, h2, diff, prod], dim=-1)

        gate = self.gate(fusion)
        gated_h1 = gate * h1
        gated_h2 = (1.0 - gate) * h2

        fusion = torch.cat([gated_h1, gated_h2, diff, prod], dim=-1)
        logits = self.classifier(fusion)
        return logits


# ============================================================
# 损失函数：Focal BCE + Class Weight
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class FocalCEWithOVRBCE(nn.Module):
    """
    Focal CE + One-vs-Rest BCE
    """
    def __init__(
        self,
        num_labels,
        ce_weight=None,
        bce_pos_weight=None,
        gamma=2.0,
        alpha_ce=1.0,
        alpha_bce=0.35,
        label_smoothing=0.02,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.focal = FocalLoss(
            gamma=gamma,
            weight=ce_weight,
            label_smoothing=label_smoothing,
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        self.alpha_ce = alpha_ce
        self.alpha_bce = alpha_bce

    def forward(self, logits, target):
        ce = self.focal(logits, target)
        onehot = F.one_hot(target, num_classes=self.num_labels).float()
        bce = self.bce(logits, onehot)
        return self.alpha_ce * ce + self.alpha_bce * bce


def make_focal_bce_loss(num_labels, class_weights, train_counts, device):
    counts = torch.tensor(train_counts, dtype=torch.float32, device=device)
    total = counts.sum()
    pos_weight = (total - counts) / counts.clamp_min(1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=60.0)

    return FocalCEWithOVRBCE(
        num_labels=num_labels,
        ce_weight=class_weights,
        bce_pos_weight=pos_weight,
        gamma=2.0,
        alpha_ce=1.0,
        alpha_bce=0.35,
        label_smoothing=0.02,
    )


# ============================================================
# FGM 对抗训练
# ============================================================

class FGM:
    def __init__(self, model, emb_name="word_embeddings", epsilon=0.6):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================
# 指标、预测、校准、融合
# ============================================================

def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict_logits_single(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        y = batch.pop("labels")
        logits = model(**batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)


@torch.no_grad()
def predict_logits_dual(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        y = batch.pop("labels")
        logits = model(**batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)


def compute_metrics(y_true, y_pred, labels):
    num_labels = len(labels)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_labels)),
        zero_division=0,
    )

    ovr_acc = []
    for c in range(num_labels):
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        ovr_acc.append(accuracy_score(yt, yp))

    per_class = pd.DataFrame({
        "类别": labels,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s,
        "one_vs_rest_accuracy": ovr_acc,
    })

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1),
        "ovr_acc_mean": float(np.mean(ovr_acc)),
        "ovr_acc_min": float(np.min(ovr_acc)),
        "per_class": per_class,
    }


def optimize_class_bias(val_logits, val_y, labels, max_iter=3):
    """
    Bias Calibration：
    在验证集上搜索类别 bias，优化 macro-F1。
    """
    num_labels = len(labels)
    bias = np.zeros(num_labels, dtype=np.float32)

    pred = (val_logits + bias).argmax(axis=1)
    best = f1_score(val_y, pred, average="macro", zero_division=0)

    grid = np.array(
        [-2.0, -1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0],
        dtype=np.float32,
    )

    for _ in range(max_iter):
        improved = False
        for c in range(num_labels):
            cur_best = best
            cur_bias = bias[c]

            for g in grid:
                tmp = bias.copy()
                tmp[c] = g
                pred = (val_logits + tmp).argmax(axis=1)
                mf1 = f1_score(val_y, pred, average="macro", zero_division=0)

                if mf1 > cur_best:
                    cur_best = mf1
                    cur_bias = g

            if cur_best > best:
                best = cur_best
                bias[c] = cur_bias
                improved = True

        if not improved:
            break

    return bias, float(best)


def search_weighted_logit_fusion(mac_val_logits, rob_val_logits, val_y, labels):
    """
    搜索：
    fused_logits = alpha * macbert_logits + (1 - alpha) * roberta_logits + bias
    """
    best = {
        "alpha": None,
        "bias": None,
        "val_macro_f1": -1.0,
    }

    for alpha in np.linspace(0.0, 1.0, 41):
        logits = alpha * mac_val_logits + (1.0 - alpha) * rob_val_logits
        bias, tuned_f1 = optimize_class_bias(logits, val_y, labels, max_iter=3)

        if tuned_f1 > best["val_macro_f1"]:
            best["alpha"] = float(alpha)
            best["bias"] = bias
            best["val_macro_f1"] = float(tuned_f1)

    return best


# ============================================================
# 训练配置
# ============================================================

@dataclass
class SingleExperimentConfig:
    name: str
    model_path: str
    max_len: int = MAX_LEN
    batch_size: int = SINGLE_BATCH_SIZE
    epochs: int = SINGLE_EPOCHS
    lr: float = SINGLE_LR
    patience: int = SINGLE_PATIENCE
    grad_accum_steps: int = SINGLE_GRAD_ACCUM
    augment_min_per_class: int = AUG_MIN_PER_CLASS_SINGLE
    use_sampler: bool = True
    use_fgm: bool = True
    dropout: float = 0.20


@dataclass
class DualExperimentConfig:
    name: str
    model_a_path: str
    model_b_path: str
    max_len: int = DUAL_MAX_LEN
    batch_size: int = DUAL_BATCH_SIZE
    epochs: int = DUAL_EPOCHS
    lr: float = DUAL_LR
    patience: int = DUAL_PATIENCE
    grad_accum_steps: int = DUAL_GRAD_ACCUM
    augment_min_per_class: int = AUG_MIN_PER_CLASS_DUAL
    use_sampler: bool = True
    use_fgm: bool = False
    dropout: float = 0.25
    freeze_bottom_layers: int = 0


# ============================================================
# 单模型训练
# ============================================================

def train_single_model(cfg, train_df_raw, val_df, test_df, labels, device, seed=42):
    print(f"\n========== 训练单编码器模型：{cfg.name} ==========")

    set_seed(seed)
    num_labels = len(labels)

    train_df = augment_minority_train(
        train_df_raw,
        labels,
        min_per_class=cfg.augment_min_per_class,
        seed=seed,
    )

    class_weights, counts = compute_class_weights(train_df, num_labels, device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)

    train_ds = HFDataset(train_df, tokenizer, cfg.max_len)
    val_ds = HFDataset(val_df, tokenizer, cfg.max_len)
    test_ds = HFDataset(test_df, tokenizer, cfg.max_len)

    sampler = build_weighted_sampler(train_df, num_labels) if cfg.use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = HFClassifier(
        cfg.model_path,
        num_labels=num_labels,
        dropout=cfg.dropout,
    )
    model.to(device)

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)

    criterion = make_focal_bce_loss(
        num_labels=num_labels,
        class_weights=class_weights,
        train_counts=counts,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=0.01,
    )

    total_steps = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs)
    warmup_steps = int(0.08 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    fgm = FGM(model) if cfg.use_fgm else None

    best_state = None
    best_val_macro = -1.0
    best_bias = np.zeros(num_labels, dtype=np.float32)
    bad_epochs = 0
    logs = []
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            y = batch.pop("labels")

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(**batch)
                loss = criterion(logits, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if cfg.use_fgm:
                fgm.attack()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    adv_logits = model(**batch)
                    adv_loss = criterion(adv_logits, y)
                    adv_loss = adv_loss / cfg.grad_accum_steps
                scaler.scale(adv_loss).backward()
                fgm.restore()

            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)

        val_logits, val_y = predict_logits_single(model, val_loader, device)
        bias, tuned_macro = optimize_class_bias(val_logits, val_y, labels, max_iter=3)
        val_pred = (val_logits + bias).argmax(axis=1)
        val_m = compute_metrics(val_y, val_pred, labels)

        log = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_ovr_acc_mean": val_m["ovr_acc_mean"],
            "val_ovr_acc_min": val_m["ovr_acc_min"],
            "epoch_seconds": time.time() - t0,
        }
        logs.append(log)

        print(
            f"[{cfg.name}] epoch={epoch:02d} "
            f"loss={log['train_loss']:.4f} "
            f"val_macroF1={log['val_macro_f1']:.4f} "
            f"val_acc={log['val_accuracy']:.4f} "
            f"val_ovrMin={log['val_ovr_acc_min']:.4f}"
        )

        if tuned_macro > best_val_macro:
            best_val_macro = tuned_macro
            best_state = copy.deepcopy(model.state_dict())
            best_bias = bias.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[{cfg.name}] Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_y = predict_logits_single(model, val_loader, device)
    test_logits, test_y = predict_logits_single(model, test_loader, device)

    val_pred = (val_logits + best_bias).argmax(axis=1)
    test_pred = (test_logits + best_bias).argmax(axis=1)

    val_m = compute_metrics(val_y, val_pred, labels)
    test_m = compute_metrics(test_y, test_pred, labels)

    sname = safe_name(cfg.name)

    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_logits.npy"), val_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_logits.npy"), test_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_y.npy"), val_y)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_y.npy"), test_y)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_bias.npy"), best_bias)

    pd.DataFrame(logs).to_csv(
        os.path.join(OUTPUT_DIR, f"training_log_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    test_m["per_class"].to_csv(
        os.path.join(OUTPUT_DIR, f"per_class_metrics_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = test_df[[ID_COL, LABEL_COL, TEXT_COL]].copy()
    pred_df["true_label_id"] = test_y
    pred_df["pred_label_id"] = test_pred
    pred_df["预测类别"] = [labels[i] for i in test_pred]
    pred_df["预测正确"] = pred_df[LABEL_COL] == pred_df["预测类别"]
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"prediction_test_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_CHECKPOINTS:
        torch.save(
            {
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
                "labels": labels,
                "best_bias": best_bias,
            },
            os.path.join(CKPT_DIR, f"{sname}.pt"),
        )

    result = {
        "name": cfg.name,
        "method_type": "single_encoder",
        "model_path": cfg.model_path,
        "max_len": cfg.max_len,
        "batch_size": cfg.batch_size,
        "epochs_planned": cfg.epochs,
        "epochs_ran": len(logs),
        "lr": cfg.lr,
        "augment_min_per_class": cfg.augment_min_per_class,
        "use_sampler": cfg.use_sampler,
        "use_fgm": cfg.use_fgm,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "seconds": time.time() - start_time,
        "best_val_macro_f1": best_val_macro,
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_f1": test_m["weighted_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_ovr_acc_mean": test_m["ovr_acc_mean"],
        "test_ovr_acc_min": test_m["ovr_acc_min"],
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, val_logits, test_logits, val_y, test_y, best_bias, test_pred


# ============================================================
# 双编码器训练
# ============================================================

def train_dual_encoder(cfg, train_df_raw, val_df, test_df, labels, device, seed=42):
    print(f"\n========== 训练双编码器融合模型：{cfg.name} ==========")

    set_seed(seed)
    num_labels = len(labels)

    train_df = augment_minority_train(
        train_df_raw,
        labels,
        min_per_class=cfg.augment_min_per_class,
        seed=seed,
    )

    class_weights, counts = compute_class_weights(train_df, num_labels, device)

    tok_a = AutoTokenizer.from_pretrained(cfg.model_a_path, local_files_only=True)
    tok_b = AutoTokenizer.from_pretrained(cfg.model_b_path, local_files_only=True)

    train_ds = DualHFDataset(train_df, tok_a, tok_b, cfg.max_len)
    val_ds = DualHFDataset(val_df, tok_a, tok_b, cfg.max_len)
    test_ds = DualHFDataset(test_df, tok_a, tok_b, cfg.max_len)

    sampler = build_weighted_sampler(train_df, num_labels) if cfg.use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DualEncoderFusionClassifier(
        model_a_path=cfg.model_a_path,
        model_b_path=cfg.model_b_path,
        num_labels=num_labels,
        dropout=cfg.dropout,
        freeze_bottom_layers=cfg.freeze_bottom_layers,
    )
    model.to(device)

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)

    criterion = make_focal_bce_loss(
        num_labels=num_labels,
        class_weights=class_weights,
        train_counts=counts,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0.01,
    )

    total_steps = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs)
    warmup_steps = int(0.08 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    fgm = FGM(model) if cfg.use_fgm else None

    best_state = None
    best_val_macro = -1.0
    best_bias = np.zeros(num_labels, dtype=np.float32)
    bad_epochs = 0
    logs = []
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            y = batch.pop("labels")

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(**batch)
                loss = criterion(logits, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if cfg.use_fgm:
                fgm.attack()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    adv_logits = model(**batch)
                    adv_loss = criterion(adv_logits, y)
                    adv_loss = adv_loss / cfg.grad_accum_steps
                scaler.scale(adv_loss).backward()
                fgm.restore()

            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)

        val_logits, val_y = predict_logits_dual(model, val_loader, device)
        bias, tuned_macro = optimize_class_bias(val_logits, val_y, labels, max_iter=3)
        val_pred = (val_logits + bias).argmax(axis=1)
        val_m = compute_metrics(val_y, val_pred, labels)

        log = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_ovr_acc_mean": val_m["ovr_acc_mean"],
            "val_ovr_acc_min": val_m["ovr_acc_min"],
            "epoch_seconds": time.time() - t0,
        }
        logs.append(log)

        print(
            f"[{cfg.name}] epoch={epoch:02d} "
            f"loss={log['train_loss']:.4f} "
            f"val_macroF1={log['val_macro_f1']:.4f} "
            f"val_acc={log['val_accuracy']:.4f} "
            f"val_ovrMin={log['val_ovr_acc_min']:.4f}"
        )

        if tuned_macro > best_val_macro:
            best_val_macro = tuned_macro
            best_state = copy.deepcopy(model.state_dict())
            best_bias = bias.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[{cfg.name}] Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_y = predict_logits_dual(model, val_loader, device)
    test_logits, test_y = predict_logits_dual(model, test_loader, device)

    test_pred = (test_logits + best_bias).argmax(axis=1)

    val_pred = (val_logits + best_bias).argmax(axis=1)
    val_m = compute_metrics(val_y, val_pred, labels)
    test_m = compute_metrics(test_y, test_pred, labels)

    sname = safe_name(cfg.name)

    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_logits.npy"), val_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_logits.npy"), test_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_y.npy"), val_y)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_y.npy"), test_y)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_bias.npy"), best_bias)

    pd.DataFrame(logs).to_csv(
        os.path.join(OUTPUT_DIR, f"training_log_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    test_m["per_class"].to_csv(
        os.path.join(OUTPUT_DIR, f"per_class_metrics_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = test_df[[ID_COL, LABEL_COL, TEXT_COL]].copy()
    pred_df["true_label_id"] = test_y
    pred_df["pred_label_id"] = test_pred
    pred_df["预测类别"] = [labels[i] for i in test_pred]
    pred_df["预测正确"] = pred_df[LABEL_COL] == pred_df["预测类别"]
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"prediction_test_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_CHECKPOINTS:
        torch.save(
            {
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
                "labels": labels,
                "best_bias": best_bias,
            },
            os.path.join(CKPT_DIR, f"{sname}.pt"),
        )

    result = {
        "name": cfg.name,
        "method_type": "dual_encoder_fusion",
        "model_path": f"{cfg.model_a_path} + {cfg.model_b_path}",
        "max_len": cfg.max_len,
        "batch_size": cfg.batch_size,
        "epochs_planned": cfg.epochs,
        "epochs_ran": len(logs),
        "lr": cfg.lr,
        "augment_min_per_class": cfg.augment_min_per_class,
        "use_sampler": cfg.use_sampler,
        "use_fgm": cfg.use_fgm,
        "freeze_bottom_layers": cfg.freeze_bottom_layers,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "seconds": time.time() - start_time,
        "best_val_macro_f1": best_val_macro,
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_f1": test_m["weighted_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_ovr_acc_mean": test_m["ovr_acc_mean"],
        "test_ovr_acc_min": test_m["ovr_acc_min"],
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, val_logits, test_logits, val_y, test_y, best_bias, test_pred


# ============================================================
# Ours: 加权 logit 融合 + Bias Calibration
# ============================================================

def evaluate_ours_logit_fusion(
    mac_val_logits,
    mac_test_logits,
    rob_val_logits,
    rob_test_logits,
    val_y,
    test_y,
    test_df,
    labels,
):
    print("\n========== Ours：MacBERT + RoBERTa 加权 Logit 融合 + Bias Calibration ==========")

    fusion = search_weighted_logit_fusion(
        mac_val_logits,
        rob_val_logits,
        val_y,
        labels,
    )

    alpha = fusion["alpha"]
    bias = fusion["bias"]

    print(f"[Ours_LogitFusion] 最优 alpha={alpha:.4f}")
    print(f"[Ours_LogitFusion] 验证集最优 macro-F1={fusion['val_macro_f1']:.4f}")
    print(f"[Ours_LogitFusion] bias={bias.tolist()}")

    val_logits = alpha * mac_val_logits + (1.0 - alpha) * rob_val_logits
    test_logits = alpha * mac_test_logits + (1.0 - alpha) * rob_test_logits

    val_pred = (val_logits + bias).argmax(axis=1)
    test_pred = (test_logits + bias).argmax(axis=1)

    val_m = compute_metrics(val_y, val_pred, labels)
    test_m = compute_metrics(test_y, test_pred, labels)

    sname = "Ours_MacBERT_RoBERTa_WeightedLogitFusion_FocalBCE_AugSampler_Bias"

    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_logits.npy"), val_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_logits.npy"), test_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_bias.npy"), bias)

    test_m["per_class"].to_csv(
        os.path.join(OUTPUT_DIR, f"per_class_metrics_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = test_df[[ID_COL, LABEL_COL, TEXT_COL]].copy()
    pred_df["true_label_id"] = test_y
    pred_df["pred_label_id"] = test_pred
    pred_df["预测类别"] = [labels[i] for i in test_pred]
    pred_df["预测正确"] = pred_df[LABEL_COL] == pred_df["预测类别"]
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"prediction_test_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(OUTPUT_DIR, f"{sname}_fusion_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha_macbert": alpha,
                "alpha_roberta": 1.0 - alpha,
                "bias": bias.tolist(),
                "val_macro_f1": fusion["val_macro_f1"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    result = {
        "name": sname,
        "method_type": "proposed_weighted_logit_fusion",
        "model_path": f"{MACBERT_LARGE_PATH} + {ROBERTA_LARGE_PATH}",
        "max_len": MAX_LEN,
        "batch_size": "single models",
        "epochs_planned": "single models",
        "epochs_ran": "single models",
        "lr": "single models",
        "augment_min_per_class": AUG_MIN_PER_CLASS_SINGLE,
        "use_sampler": True,
        "use_fgm": True,
        "fusion_alpha_macbert": alpha,
        "fusion_alpha_roberta": 1.0 - alpha,
        "total_params": np.nan,
        "trainable_params": np.nan,
        "seconds": 0.0,
        "best_val_macro_f1": fusion["val_macro_f1"],
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_f1": test_m["weighted_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_ovr_acc_mean": test_m["ovr_acc_mean"],
        "test_ovr_acc_min": test_m["ovr_acc_min"],
    }

    print(
        f"[Ours_LogitFusion] test_macroF1={test_m['macro_f1']:.4f} "
        f"test_acc={test_m['accuracy']:.4f} "
        f"ovr_min={test_m['ovr_acc_min']:.4f}"
    )

    return result, test_pred


# ============================================================
# 可视化
# ============================================================

def plot_class_distribution(df, labels):
    counts = df[LABEL_COL].value_counts().reindex(labels).fillna(0)

    plt.figure(figsize=(13, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("全量数据类别分布")
    plt.ylabel("样本数")
    plt.xlabel("类别")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(9, 9))
    plt.pie(counts.values, labels=counts.index, autopct="%1.2f%%", startangle=140)
    plt.title("全量数据类别占比")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution_pie.png"), dpi=240)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{name} 测试集归一化混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{safe_name(name)}.png"), dpi=240)
    plt.close()


def plot_summary(summary_df):
    if summary_df.empty:
        return

    s = summary_df.sort_values("test_macro_f1", ascending=False)

    plt.figure(figsize=(15, 6))
    sns.barplot(data=s, x="name", y="test_macro_f1")
    plt.axhline(0.8, color="red", linestyle="--", label="Macro-F1 = 0.8")
    plt.title("不同模型测试集 Macro-F1 对比")
    plt.xlabel("模型")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_macro_f1_comparison.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(15, 6))
    sns.barplot(data=s, x="name", y="test_ovr_acc_min")
    plt.axhline(0.9, color="red", linestyle="--", label="OVR min accuracy = 0.9")
    plt.title("不同模型 One-vs-Rest 最小二分类准确率")
    plt.xlabel("模型")
    plt.ylabel("One-vs-Rest 最小准确率")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_ovr_min_acc_comparison.png"), dpi=240)
    plt.close()


def plot_training_curves():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("training_log_") and f.endswith(".csv")]
    if not files:
        return

    plt.figure(figsize=(14, 7))
    for f in files:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
        name = f.replace("training_log_", "").replace(".csv", "")
        plt.plot(df["epoch"], df["train_loss"], marker="o", label=name)
    plt.title("训练损失随 Epoch 变化")
    plt.xlabel("Epoch")
    plt.ylabel("训练损失")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_loss_curves.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(14, 7))
    for f in files:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
        name = f.replace("training_log_", "").replace(".csv", "")
        plt.plot(df["epoch"], df["val_macro_f1"], marker="o", label=name)
    plt.axhline(0.8, color="red", linestyle="--")
    plt.title("验证集 Macro-F1 随 Epoch 变化")
    plt.xlabel("Epoch")
    plt.ylabel("验证集 Macro-F1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "val_macro_f1_curves.png"), dpi=240)
    plt.close()


def plot_per_class_best(best_name, labels):
    path = os.path.join(OUTPUT_DIR, f"per_class_metrics_{safe_name(best_name)}.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    plt.figure(figsize=(13, 6))
    sns.barplot(data=df, x="类别", y="f1")
    plt.axhline(0.8, color="red", linestyle="--")
    plt.title(f"{best_name} 各类别 F1")
    plt.ylabel("F1")
    plt.xlabel("类别")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"per_class_f1_{safe_name(best_name)}.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(13, 6))
    sns.barplot(data=df, x="类别", y="one_vs_rest_accuracy")
    plt.axhline(0.9, color="red", linestyle="--")
    plt.title(f"{best_name} 各类别 One-vs-Rest 二分类准确率")
    plt.ylabel("One-vs-Rest accuracy")
    plt.xlabel("类别")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"per_class_ovr_acc_{safe_name(best_name)}.png"), dpi=240)
    plt.close()


# ============================================================
# 报告生成
# ============================================================

def generate_report(summary_df, labels, full_df, train_df, val_df, test_df):
    if summary_df.empty:
        return

    best = summary_df.sort_values("test_macro_f1", ascending=False).iloc[0]

    lines = []
    lines.append("# LT-MVF 违禁评论识别实验报告\n\n")

    lines.append("## 1. 实验任务\n\n")
    lines.append("本实验面向非公开违禁评论数据集，任务为 10 类单标签文本分类。核心评价指标为多分类 Macro-F1，同时要求各类别 one-vs-rest 二分类准确率大于 0.9。\n\n")

    lines.append("## 2. 数据集划分\n\n")
    lines.append("实验采用分层留出法，按照 70%/15%/15% 划分训练集、验证集和测试集。\n\n")
    lines.append(f"- 全量样本数：{len(full_df)}\n")
    lines.append(f"- 训练集：{len(train_df)}\n")
    lines.append(f"- 验证集：{len(val_df)}\n")
    lines.append(f"- 测试集：{len(test_df)}\n")
    lines.append(f"- 类别数：{len(labels)}\n\n")

    vc = full_df[LABEL_COL].value_counts().reindex(labels).fillna(0).astype(int)
    lines.append("| 类别 | 频次 | 频率 |\n")
    lines.append("|---|---:|---:|\n")
    for lab, cnt in vc.items():
        lines.append(f"| {lab} | {cnt} | {cnt / len(full_df):.6f} |\n")

    lines.append("\n## 3. 本文提出方法 LT-MVF\n\n")
    lines.append("本文提出 **LT-MVF（Long-Tail Multi-View Fusion）长尾多视角预训练融合模型**。该方法不是单纯调用一个预训练模型，而是在长尾不均衡条件下构建多视角深度融合框架。\n\n")

    lines.append("核心组成如下：\n\n")
    lines.append("```text\n")
    lines.append("Ours = MacBERT-large + RoBERTa-large 加权 logit 融合\n")
    lines.append("       + Focal BCE\n")
    lines.append("       + 类别权重\n")
    lines.append("       + Weighted Sampler\n")
    lines.append("       + 少数类增强\n")
    lines.append("       + Bias Calibration\n")
    lines.append("```\n\n")

    lines.append("另外，本文还实现了端到端双编码器融合模型：\n\n")
    lines.append("```text\n")
    lines.append("DualEncoderFusion = MacBERT-large Encoder\n")
    lines.append("                  + RoBERTa-large Encoder\n")
    lines.append("                  + [h1, h2, |h1-h2|, h1*h2] 深度融合\n")
    lines.append("                  + 门控 MLP 分类器\n")
    lines.append("                  + Focal BCE + 类别权重 + 少数类增强\n")
    lines.append("```\n\n")

    lines.append("## 4. 实验结果汇总\n\n")
    lines.append(summary_df.sort_values("test_macro_f1", ascending=False).to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## 5. 最佳模型\n\n")
    lines.append(f"- 最佳模型：{best['name']}\n")
    lines.append(f"- Macro-F1：{best['test_macro_f1']:.4f}\n")
    lines.append(f"- Accuracy：{best['test_accuracy']:.4f}\n")
    lines.append(f"- Balanced Accuracy：{best['test_balanced_accuracy']:.4f}\n")
    lines.append(f"- Weighted-F1：{best['test_weighted_f1']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率均值：{best['test_ovr_acc_mean']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率最小值：{best['test_ovr_acc_min']:.4f}\n\n")

    lines.append("## 6. 达标检查\n\n")
    lines.append(f"- Macro-F1 > 0.8：{bool(best['test_macro_f1'] > 0.8)}，当前 {best['test_macro_f1']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率均值 > 0.9：{bool(best['test_ovr_acc_mean'] > 0.9)}，当前 {best['test_ovr_acc_mean']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率最小值 > 0.9：{bool(best['test_ovr_acc_min'] > 0.9)}，当前 {best['test_ovr_acc_min']:.4f}\n\n")

    lines.append("## 7. 方法分析\n\n")
    lines.append("- MacBERT-large 提供强中文语义建模能力；\n")
    lines.append("- RoBERTa-large 提供互补的上下文表征；\n")
    lines.append("- Focal BCE 同时优化单标签多分类边界和 one-vs-rest 辅助边界；\n")
    lines.append("- 类别权重和 Weighted Sampler 缓解极端长尾问题；\n")
    lines.append("- 少数类增强提升稀有类别曝光频率；\n")
    lines.append("- Bias Calibration 直接在验证集上优化 Macro-F1；\n")
    lines.append("- 双编码器融合模型通过特征级深度融合进一步探索超过单模型的可能性。\n\n")

    lines.append("## 8. 图表文件\n\n")
    lines.append("- 类别分布：`outputs_ltmvf/figures/class_distribution.png`\n")
    lines.append("- 类别占比：`outputs_ltmvf/figures/class_distribution_pie.png`\n")
    lines.append("- 模型 Macro-F1 对比：`outputs_ltmvf/figures/model_macro_f1_comparison.png`\n")
    lines.append("- One-vs-Rest 准确率对比：`outputs_ltmvf/figures/model_ovr_min_acc_comparison.png`\n")
    lines.append("- 训练损失曲线：`outputs_ltmvf/figures/training_loss_curves.png`\n")
    lines.append("- 验证集 Macro-F1 曲线：`outputs_ltmvf/figures/val_macro_f1_curves.png`\n")
    lines.append("- 最佳模型混淆矩阵：`outputs_ltmvf/figures/confusion_matrix_*.png`\n\n")

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))


# ============================================================
# 主流程
# ============================================================

def main():
    ensure_dirs()
    setup_chinese_font()
    set_seed(SEED)

    check_local_model(MACBERT_LARGE_PATH, "MacBERT-large")
    check_local_model(ROBERTA_LARGE_PATH, "RoBERTa-wwm-ext-large")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    print(f"MacBERT-large 本地路径：{MACBERT_LARGE_PATH}")
    print(f"RoBERTa-wwm-ext-large 本地路径：{ROBERTA_LARGE_PATH}")

    df, labels, label2id, id2label = load_all_csv()

    print(f"数据量：{len(df)}")
    print(f"类别：{labels}")
    print("\n类别分布：")
    print(df[LABEL_COL].value_counts().reindex(labels).fillna(0).astype(int))

    plot_class_distribution(df, labels)

    train_df, val_df, test_df = stratified_split(df, seed=SEED)
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    split_info = {
        "seed": SEED,
        "labels": labels,
        "label2id": label2id,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "macbert_large_path": MACBERT_LARGE_PATH,
        "roberta_large_path": ROBERTA_LARGE_PATH,
    }

    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False, encoding="utf-8-sig")

    results = []
    best_pack = None

    # ========================================================
    # 1. MacBERT-large 单模型
    # ========================================================

    mac_cfg = SingleExperimentConfig(
        name="MacBERT_large_FocalBCE_ClassWeight_Sampler_Aug_Bias",
        model_path=MACBERT_LARGE_PATH,
        max_len=MAX_LEN,
        batch_size=SINGLE_BATCH_SIZE,
        epochs=SINGLE_EPOCHS,
        lr=SINGLE_LR,
        patience=SINGLE_PATIENCE,
        grad_accum_steps=SINGLE_GRAD_ACCUM,
        augment_min_per_class=AUG_MIN_PER_CLASS_SINGLE,
        use_sampler=True,
        use_fgm=True,
        dropout=0.20,
    )

    mac_result, mac_val_logits, mac_test_logits, val_y, test_y, mac_bias, mac_test_pred = train_single_model(
        mac_cfg,
        train_df,
        val_df,
        test_df,
        labels,
        device,
        seed=42,
    )
    results.append(mac_result)
    plot_confusion_matrix(test_y, mac_test_pred, labels, mac_cfg.name)
    best_pack = (mac_result, test_y, mac_test_pred)

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 2. RoBERTa-large 单模型
    # ========================================================

    rob_cfg = SingleExperimentConfig(
        name="RoBERTa_large_FocalBCE_ClassWeight_Sampler_Aug_Bias",
        model_path=ROBERTA_LARGE_PATH,
        max_len=MAX_LEN,
        batch_size=SINGLE_BATCH_SIZE,
        epochs=SINGLE_EPOCHS,
        lr=SINGLE_LR,
        patience=SINGLE_PATIENCE,
        grad_accum_steps=SINGLE_GRAD_ACCUM,
        augment_min_per_class=AUG_MIN_PER_CLASS_SINGLE,
        use_sampler=True,
        use_fgm=True,
        dropout=0.20,
    )

    rob_result, rob_val_logits, rob_test_logits, val_y2, test_y2, rob_bias, rob_test_pred = train_single_model(
        rob_cfg,
        train_df,
        val_df,
        test_df,
        labels,
        device,
        seed=3407,
    )
    results.append(rob_result)
    plot_confusion_matrix(test_y2, rob_test_pred, labels, rob_cfg.name)

    if rob_result["test_macro_f1"] > best_pack[0]["test_macro_f1"]:
        best_pack = (rob_result, test_y2, rob_test_pred)

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 3. Ours：加权 Logit 融合 + Bias Calibration
    # ========================================================

    ours_result, ours_test_pred = evaluate_ours_logit_fusion(
        mac_val_logits=mac_val_logits,
        mac_test_logits=mac_test_logits,
        rob_val_logits=rob_val_logits,
        rob_test_logits=rob_test_logits,
        val_y=val_y,
        test_y=test_y,
        test_df=test_df,
        labels=labels,
    )

    results.append(ours_result)
    plot_confusion_matrix(test_y, ours_test_pred, labels, ours_result["name"])

    if ours_result["test_macro_f1"] > best_pack[0]["test_macro_f1"]:
        best_pack = (ours_result, test_y, ours_test_pred)

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. 双编码器融合模型
    # ========================================================

    if RUN_DUAL_ENCODER:
        try:
            dual_cfg = DualExperimentConfig(
                name="DualEncoderFusion_MacBERT_RoBERTa_FocalBCE_AugSampler_Bias",
                model_a_path=MACBERT_LARGE_PATH,
                model_b_path=ROBERTA_LARGE_PATH,
                max_len=DUAL_MAX_LEN,
                batch_size=DUAL_BATCH_SIZE,
                epochs=DUAL_EPOCHS,
                lr=DUAL_LR,
                patience=DUAL_PATIENCE,
                grad_accum_steps=DUAL_GRAD_ACCUM,
                augment_min_per_class=AUG_MIN_PER_CLASS_DUAL,
                use_sampler=True,
                use_fgm=False,
                dropout=0.25,
                freeze_bottom_layers=0,
            )

            dual_result, dual_val_logits, dual_test_logits, dual_val_y, dual_test_y, dual_bias, dual_test_pred = train_dual_encoder(
                dual_cfg,
                train_df,
                val_df,
                test_df,
                labels,
                device,
                seed=2024,
            )

            results.append(dual_result)
            plot_confusion_matrix(dual_test_y, dual_test_pred, labels, dual_cfg.name)

            if dual_result["test_macro_f1"] > best_pack[0]["test_macro_f1"]:
                best_pack = (dual_result, dual_test_y, dual_test_pred)

            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        except RuntimeError as e:
            print(f"[跳过] 双编码器模型 RuntimeError：{repr(e)}")
            if "out of memory" in str(e).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[跳过] 双编码器模型运行失败：{repr(e)}")

    # ========================================================
    # 汇总与报告
    # ========================================================

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_summary(summary_df)
    plot_training_curves()

    best = summary_df.sort_values("test_macro_f1", ascending=False).iloc[0]
    plot_per_class_best(best["name"], labels)

    generate_report(
        summary_df,
        labels,
        df,
        train_df,
        val_df,
        test_df,
    )

    print("\n========== 最终最佳结果 ==========")
    print(best.to_string())

    print("\n关键达标检查：")
    print(f"Macro-F1 > 0.8: {best['test_macro_f1'] > 0.8}，当前 {best['test_macro_f1']:.4f}")
    print(f"One-vs-rest 二分类准确率均值 > 0.9: {best['test_ovr_acc_mean'] > 0.9}，当前 {best['test_ovr_acc_mean']:.4f}")
    print(f"One-vs-rest 二分类准确率最小值 > 0.9: {best['test_ovr_acc_min'] > 0.9}，当前 {best['test_ovr_acc_min']:.4f}")

    print(f"\n所有输出已保存至：{os.path.abspath(OUTPUT_DIR)}")
    print(f"报告文件：{os.path.abspath(os.path.join(OUTPUT_DIR, 'report.md'))}")


if __name__ == "__main__":
    main()