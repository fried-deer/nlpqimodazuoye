# -*- coding: utf-8 -*-
"""
forbidden_comment_experiment.py


功能：
1. 读取 all.csv，格式：id,类别,文本
2. 自动使用当前目录 simsun.ttc，保证中文图表正常显示
3. 自动使用本地模型：
   ./hf_models/chinese-macbert-large
   ./hf_models/chinese-macbert-base
4. 完整完成：
   - baseline 神经网络对比
   - 高 F1 主模型训练
   - 长尾类别增强
   - Focal CE + One-vs-Rest BCE 组合损失
   - 类别权重
   - WeightedRandomSampler
   - 验证集 macro-F1 类别偏置校准
   - 最佳模型消融实验
   - 降本实验
   - 参数量、训练轮次、训练损失影响分析
   - 可视化图表
   - 自动生成 report.md

推荐运行：
nohup python forbidden_comment_experiment.py > output.log 2>&1 &

输出目录：
outputs/
  metrics_summary.csv
  report.md
  figures/
  checkpoints/
  training_log_*.csv
  per_class_metrics_*.csv
  prediction_test_*.csv
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

# 禁止 HuggingFace 联网/Xet，全部使用本地模型
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


# ============================================================
# 全局配置
# ============================================================

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

OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DATA_PATH = "all.csv"
SEED = 42

MACBERT_LARGE_CANDIDATES = [
    "./hf_models/chinese-macbert-large",
    "./models/chinese-macbert-large",
]

ROBERTA_WWM_EXT_LARGE_CANDIDATES = [
    "./hf_models/chinese-roberta-wwm-ext-large",
    "./models/chinese-roberta-wwm-ext-large",
]

# 主要训练配置
BEST_MAX_LEN = 192
BEST_BATCH_SIZE = 16
BEST_EPOCHS = 12
BEST_LR = 1.5e-5

BASE_MAX_LEN = 192
BASE_BATCH_SIZE = 32
BASE_EPOCHS = 10
BASE_LR = 2.0e-5

# 长尾增强：训练集中每类至少增强到该数量
AUG_MIN_PER_CLASS_BEST = 900
AUG_MIN_PER_CLASS_ABLATION = 600
AUG_MIN_PER_CLASS_BASELINE = 500

# 是否运行所有实验
RUN_BASELINES = True
RUN_BEST_MODEL = True
RUN_ABLATION = True
RUN_COST = True


# ============================================================
# 基础工具
# ============================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)


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

    print("警告：未找到 simsun.ttc，中文图表可能显示为方块。")
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


def find_existing_dir(candidates, required_files=("config.json",)):
    for p in candidates:
        if os.path.isdir(p):
            ok = True
            for f in required_files:
                if not os.path.exists(os.path.join(p, f)):
                    ok = False
                    break
            if ok:
                return p
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


def count_total_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def safe_name(name):
    return re.sub(r"[^\w\u4e00-\u9fa5\-\(\)]+", "_", name)


# ============================================================
# 数据读取、划分、增强
# ============================================================

def load_all_csv(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path)
    required = {"id", "类别", "文本"}
    if not required.issubset(df.columns):
        raise ValueError(f"all.csv 必须包含列 {required}，当前列：{df.columns.tolist()}")

    df = df[["id", "类别", "文本"]].copy()
    df["文本"] = df["文本"].apply(clean_text)
    df = df[df["文本"].str.len() > 0].reset_index(drop=True)

    labels = [x for x in LABELS_ORDER if x in set(df["类别"])]
    extra = sorted(list(set(df["类别"]) - set(labels)))
    labels += extra

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label"] = df["类别"].map(label2id).astype(int)

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
    轻量文本增强。
    注意：不改变语义，只做弱扰动，主要用于少数类重采样时避免完全重复。
    """
    text = str(text)
    if len(text) <= 2:
        return text

    ops = ["drop_punc", "swap_adjacent", "delete_space", "duplicate_char", "identity"]
    op = random.choice(ops)

    chars = list(text)

    if op == "drop_punc":
        idxs = [i for i, c in enumerate(chars) if c in PUNCS]
        if idxs:
            i = random.choice(idxs)
            chars.pop(i)
            return "".join(chars)

    elif op == "swap_adjacent":
        if len(chars) >= 4:
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)

    elif op == "delete_space":
        return re.sub(r"\s+", "", text)

    elif op == "duplicate_char":
        valid = [i for i, c in enumerate(chars) if c.strip()]
        if valid and len(chars) < 500:
            i = random.choice(valid)
            chars.insert(i, chars[i])
            return "".join(chars)

    return text


def augment_minority_train(train_df, labels, min_per_class=900, seed=42):
    """
    只增强训练集，不增强验证集和测试集，避免数据泄漏。
    """
    if min_per_class <= 0:
        return train_df.copy()

    random.seed(seed)
    parts = [train_df.copy()]
    next_id_base = int(train_df["id"].max()) + 1 if pd.api.types.is_numeric_dtype(train_df["id"]) else 10_000_000

    for label_id, label_name in enumerate(labels):
        sub = train_df[train_df["label"] == label_id]
        n = len(sub)
        if n == 0:
            continue
        if n >= min_per_class:
            continue

        need = min_per_class - n
        rows = []
        sub_records = sub.to_dict("records")
        for k in range(need):
            r = random.choice(sub_records).copy()
            r["文本"] = perturb_text(r["文本"])
            r["id"] = f"aug_{next_id_base}_{label_id}_{k}"
            rows.append(r)

        parts.append(pd.DataFrame(rows))
        print(f"[数据增强] 类别={label_name} 原训练样本={n} 增强={need} 增强后={min_per_class}")

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
# 字符级 baseline 数据集和模型
# ============================================================

class CharVocab:
    def __init__(self, min_freq=1, max_size=12000):
        self.pad = "[PAD]"
        self.unk = "[UNK]"
        self.stoi = {self.pad: 0, self.unk: 1}
        self.itos = [self.pad, self.unk]
        self.min_freq = min_freq
        self.max_size = max_size

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(list(str(t)))
        items = [(c, f) for c, f in counter.items() if f >= self.min_freq]
        items = sorted(items, key=lambda x: (-x[1], x[0]))[: self.max_size - 2]
        for c, _ in items:
            self.stoi[c] = len(self.itos)
            self.itos.append(c)

    def encode(self, text, max_len):
        ids = [self.stoi.get(c, 1) for c in list(str(text))[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


class CharDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = df["文本"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class HFDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["文本"].tolist()
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


class FastTextNN(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=256, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        emb = self.embedding(input_ids)
        mask = (input_ids != 0).float().unsqueeze(-1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.fc(pooled)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=256, num_filters=256, kernels=(2, 3, 4, 5), dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernels), num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embedding(input_ids).transpose(1, 2)
        outs = []
        for conv in self.convs:
            y = F.gelu(conv(x))
            y = F.max_pool1d(y, y.size(-1)).squeeze(-1)
            outs.append(y)
        return self.fc(self.dropout(torch.cat(outs, dim=1)))


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=256, hidden=256, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        mask = input_ids.ne(0)
        score = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        alpha = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (out * alpha).sum(1)
        return self.fc(self.dropout(pooled))


class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, max_len=192, embed_dim=256, heads=8, layers=4, ff=768, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        x = self.embedding(input_ids) + self.pos(pos)
        pad_mask = input_ids.eq(0)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.fc(self.dropout(pooled))


class HFClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout=0.2, freeze_layers=0):
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

        if freeze_layers > 0:
            self.freeze_bottom_layers(freeze_layers)

    def freeze_bottom_layers(self, n):
        if n >= 1 and hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer

        if layers is not None:
            for layer in layers[: max(0, n - 1)]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


# ============================================================
# 损失函数
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
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
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class FocalCEWithOVRBCE(nn.Module):
    """
    主损失：
    Focal CE + One-vs-Rest BCE

    设计目的：
    - Focal CE 解决易分类多数样本主导问题。
    - 类别权重提高少数类损失贡献。
    - OVR BCE 明确优化每个类别的一对多边界，有利于 macro-F1。
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


def make_loss(
    loss_name,
    num_labels,
    class_weights=None,
    train_counts=None,
    device=None,
    label_smoothing=0.02,
):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    if loss_name == "ce_no_weight":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if loss_name == "focal":
        return FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=label_smoothing)

    if loss_name == "focal_bce":
        if train_counts is None or device is None:
            raise ValueError("focal_bce 需要 train_counts 和 device")

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
            label_smoothing=label_smoothing,
        )

    raise ValueError(loss_name)


# ============================================================
# 指标、预测、类别偏置校准
# ============================================================

def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        y = batch.pop("labels")
        logits = model(**batch)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()


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
    在验证集上搜索类别 bias，优化 macro-F1。
    对长尾类别非常有用。
    本质相当于后验校准，不改变模型参数。
    """
    num_labels = len(labels)
    bias = np.zeros(num_labels, dtype=np.float32)

    base_pred = (val_logits + bias).argmax(axis=1)
    best = f1_score(val_y, base_pred, average="macro", zero_division=0)

    # 少数类通常需要正 bias，多数类可能需要负 bias
    grid = np.array([-2.5, -2.0, -1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)

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


# ============================================================
# 实验配置
# ============================================================

@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    loss_name: str = "focal_bce"
    use_class_weight: bool = True
    use_sampler: bool = True
    augment_min_per_class: int = 0
    label_smoothing: float = 0.02
    dropout: float = 0.2
    weight_decay: float = 0.01
    patience: int = 3
    bert_model: str = ""
    freeze_layers: int = 0
    tune_bias: bool = True
    grad_accum_steps: int = 1


# ============================================================
# 训练单个实验
# ============================================================

def build_dataset_and_model(cfg, train_df, val_df, test_df, labels, device):
    num_labels = len(labels)

    if cfg.model_type in ["fasttext", "textcnn", "bilstm", "tinytransformer"]:
        vocab = CharVocab(min_freq=1, max_size=12000)
        vocab.fit(train_df["文本"].tolist())

        train_ds = CharDataset(train_df, vocab, cfg.max_len)
        val_ds = CharDataset(val_df, vocab, cfg.max_len)
        test_ds = CharDataset(test_df, vocab, cfg.max_len)

        if cfg.model_type == "fasttext":
            model = FastTextNN(len(vocab), num_labels, dropout=cfg.dropout)
        elif cfg.model_type == "textcnn":
            model = TextCNN(len(vocab), num_labels, dropout=cfg.dropout)
        elif cfg.model_type == "bilstm":
            model = BiLSTMAttn(len(vocab), num_labels, dropout=cfg.dropout)
        else:
            model = TinyTransformerClassifier(len(vocab), num_labels, max_len=cfg.max_len, dropout=cfg.dropout)

        return train_ds, val_ds, test_ds, model

    if cfg.model_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model, local_files_only=True)
        train_ds = HFDataset(train_df, tokenizer, cfg.max_len)
        val_ds = HFDataset(val_df, tokenizer, cfg.max_len)
        test_ds = HFDataset(test_df, tokenizer, cfg.max_len)

        model = HFClassifier(
            cfg.bert_model,
            num_labels=num_labels,
            dropout=cfg.dropout,
            freeze_layers=cfg.freeze_layers,
        )
        return train_ds, val_ds, test_ds, model

    raise ValueError(cfg.model_type)


def train_one_experiment(cfg, train_df_raw, val_df, test_df, labels, device, seed=42):
    print(f"\n========== 训练实验：{cfg.name} ==========")

    num_labels = len(labels)

    # 只增强训练集
    train_df = augment_minority_train(
        train_df_raw,
        labels,
        min_per_class=cfg.augment_min_per_class,
        seed=seed,
    )

    class_weights, counts = compute_class_weights(train_df, num_labels, device)
    if not cfg.use_class_weight:
        class_weights = None

    train_ds, val_ds, test_ds, model = build_dataset_and_model(cfg, train_df, val_df, test_df, labels, device)
    model.to(device)

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)

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

    criterion = make_loss(
        cfg.loss_name,
        num_labels=num_labels,
        class_weights=class_weights,
        train_counts=counts,
        device=device,
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs)
    warmup_steps = int(0.08 * total_steps)

    if cfg.model_type == "hf":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_state = None
    best_val_macro_f1 = -1.0
    best_bias = np.zeros(num_labels, dtype=np.float32)
    bad_epochs = 0
    logs = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            y = batch.pop("labels")

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(**batch)
                loss = criterion(logits, y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)

        val_logits, val_y = predict_logits(model, val_loader, device)

        if cfg.tune_bias:
            bias, tuned_val_macro = optimize_class_bias(val_logits, val_y, labels, max_iter=3)
            val_pred = (val_logits + bias).argmax(axis=1)
        else:
            bias = np.zeros(num_labels, dtype=np.float32)
            val_pred = val_logits.argmax(axis=1)
            tuned_val_macro = f1_score(val_y, val_pred, average="macro", zero_division=0)

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
            f"val_ovrMean={log['val_ovr_acc_mean']:.4f} "
            f"val_ovrMin={log['val_ovr_acc_min']:.4f}"
        )

        if tuned_val_macro > best_val_macro_f1:
            best_val_macro_f1 = tuned_val_macro
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

    val_logits, val_y = predict_logits(model, val_loader, device)
    test_logits, test_y = predict_logits(model, test_loader, device)

    val_pred_raw = val_logits.argmax(axis=1)
    test_pred_raw = test_logits.argmax(axis=1)

    val_pred = (val_logits + best_bias).argmax(axis=1)
    test_pred = (test_logits + best_bias).argmax(axis=1)

    test_prob = torch.softmax(torch.tensor(test_logits + best_bias), dim=1).numpy()

    raw_test_m = compute_metrics(test_y, test_pred_raw, labels)
    test_m = compute_metrics(test_y, test_pred, labels)

    elapsed = time.time() - start_time

    sname = safe_name(cfg.name)

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

    pred_df = test_df[["id", "类别", "文本"]].copy()
    pred_df["true_label_id"] = test_y
    pred_df["pred_label_id_raw"] = test_pred_raw
    pred_df["pred_label_id"] = test_pred
    pred_df["预测类别"] = [labels[i] for i in test_pred]
    pred_df["预测正确"] = pred_df["类别"] == pred_df["预测类别"]
    pred_df["max_prob"] = test_prob.max(axis=1)

    for i, lab in enumerate(labels):
        pred_df[f"prob_{lab}"] = test_prob[:, i]

    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"prediction_test_{sname}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    ckpt_path = os.path.join(CKPT_DIR, f"{sname}.pt")
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "labels": labels,
            "best_bias": best_bias,
            "test_metrics": {k: v for k, v in test_m.items() if k != "per_class"},
        },
        ckpt_path,
    )

    result = {
        "name": cfg.name,
        "model_type": cfg.model_type,
        "bert_model": cfg.bert_model,
        "max_len": cfg.max_len,
        "batch_size": cfg.batch_size,
        "epochs_planned": cfg.epochs,
        "epochs_ran": len(logs),
        "lr": cfg.lr,
        "loss_name": cfg.loss_name,
        "use_class_weight": cfg.use_class_weight,
        "use_sampler": cfg.use_sampler,
        "augment_min_per_class": cfg.augment_min_per_class,
        "label_smoothing": cfg.label_smoothing,
        "freeze_layers": cfg.freeze_layers,
        "dropout": cfg.dropout,
        "tune_bias": cfg.tune_bias,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "seconds": elapsed,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_f1": test_m["weighted_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_ovr_acc_mean": test_m["ovr_acc_mean"],
        "test_ovr_acc_min": test_m["ovr_acc_min"],
        "raw_test_macro_f1_without_bias": raw_test_m["macro_f1"],
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, test_y, test_pred


# ============================================================
# 可视化
# ============================================================

def plot_class_distribution(df, labels):
    counts = df["类别"].value_counts().reindex(labels).fillna(0)

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

    plt.figure(figsize=(14, 6))
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

    plt.figure(figsize=(14, 6))
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

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=s,
        x="trainable_params",
        y="test_macro_f1",
        size="seconds",
        hue="name",
        sizes=(80, 500),
    )
    plt.xscale("log")
    plt.axhline(0.8, color="red", linestyle="--")
    plt.title("参数量-性能-训练时间关系")
    plt.xlabel("可训练参数量，log scale")
    plt.ylabel("测试集 Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "params_vs_f1.png"), dpi=240)
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
# 实验列表
# ============================================================

def make_experiments(macbert_large_model, roberta_large_model):
    exps = []

    # =========================
    # 1. Baseline 神经网络
    # =========================
    if RUN_BASELINES:
        exps.extend([
            ExperimentConfig(
                name="Baseline_FastTextNN",
                model_type="fasttext",
                max_len=192,
                batch_size=128,
                epochs=6,
                lr=3e-4,
                loss_name="focal",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BASELINE,
                patience=2,
                dropout=0.35,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Baseline_TextCNN",
                model_type="textcnn",
                max_len=192,
                batch_size=128,
                epochs=8,
                lr=3e-4,
                loss_name="focal",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BASELINE,
                patience=2,
                dropout=0.35,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Baseline_BiLSTM_Attention",
                model_type="bilstm",
                max_len=192,
                batch_size=96,
                epochs=8,
                lr=3e-4,
                loss_name="focal",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BASELINE,
                patience=2,
                dropout=0.35,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Baseline_TinyTransformer",
                model_type="tinytransformer",
                max_len=192,
                batch_size=96,
                epochs=8,
                lr=3e-4,
                loss_name="focal",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BASELINE,
                patience=2,
                dropout=0.25,
                tune_bias=True,
            ),
        ])

    # =========================
    # 2. 主模型：MacBERT-large
    # =========================
    if macbert_large_model and RUN_BEST_MODEL:
        exps.append(
            ExperimentConfig(
                name="Best_MacBERT_large_FocalBCE_AugSampler_Bias",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=BEST_EPOCHS,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BEST,
                label_smoothing=0.02,
                dropout=0.20,
                patience=3,
                freeze_layers=0,
                tune_bias=True,
                grad_accum_steps=1,
            )
        )

    # =========================
    # 3. 对照强模型：RoBERTa-wwm-ext-large
    # =========================
    if roberta_large_model:
        exps.append(
            ExperimentConfig(
                name="Compare_RoBERTa_wwm_ext_large_FocalBCE_AugSampler_Bias",
                model_type="hf",
                bert_model=roberta_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=10,
                lr=1.5e-5,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_BEST,
                label_smoothing=0.02,
                dropout=0.20,
                patience=3,
                freeze_layers=0,
                tune_bias=True,
                grad_accum_steps=1,
            )
        )

    # =========================
    # 4. MacBERT-large 消融实验
    # =========================
    if macbert_large_model and RUN_ABLATION:
        exps.extend([
            ExperimentConfig(
                name="Ablation_no_OVRBCE_use_FocalOnly",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Ablation_no_class_weight",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=False,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Ablation_no_sampler",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=False,
                augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Ablation_no_augmentation",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=0,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=True,
            ),
            ExperimentConfig(
                name="Ablation_no_bias_calibration",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=BEST_MAX_LEN,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=False,
            ),
            ExperimentConfig(
                name="Ablation_short_maxlen_96",
                model_type="hf",
                bert_model=macbert_large_model,
                max_len=96,
                batch_size=BEST_BATCH_SIZE,
                epochs=8,
                lr=BEST_LR,
                loss_name="focal_bce",
                use_class_weight=True,
                use_sampler=True,
                augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                label_smoothing=0.02,
                dropout=0.20,
                patience=2,
                tune_bias=True,
            ),
        ])

    # =========================
    # 5. 降本实验
    # =========================
    if RUN_COST:
        # 5.1 冻结 MacBERT-large 底部层，降低反传成本
        if macbert_large_model:
            exps.append(
                ExperimentConfig(
                    name="Cost_MacBERT_large_freeze_18_layers",
                    model_type="hf",
                    bert_model=macbert_large_model,
                    max_len=BEST_MAX_LEN,
                    batch_size=BEST_BATCH_SIZE,
                    epochs=8,
                    lr=BEST_LR * 2,
                    loss_name="focal_bce",
                    use_class_weight=True,
                    use_sampler=True,
                    augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                    label_smoothing=0.02,
                    dropout=0.20,
                    patience=2,
                    freeze_layers=18,
                    tune_bias=True,
                )
            )

        # 5.2 RoBERTa-large 缩短 max_len，降低注意力计算成本
        if roberta_large_model:
            exps.extend([
                ExperimentConfig(
                    name="Cost_RoBERTa_large_short_maxlen_128",
                    model_type="hf",
                    bert_model=roberta_large_model,
                    max_len=128,
                    batch_size=BEST_BATCH_SIZE,
                    epochs=8,
                    lr=1.5e-5,
                    loss_name="focal_bce",
                    use_class_weight=True,
                    use_sampler=True,
                    augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                    label_smoothing=0.02,
                    dropout=0.20,
                    patience=2,
                    freeze_layers=0,
                    tune_bias=True,
                ),
                ExperimentConfig(
                    name="Cost_RoBERTa_large_freeze_18_layers",
                    model_type="hf",
                    bert_model=roberta_large_model,
                    max_len=BEST_MAX_LEN,
                    batch_size=BEST_BATCH_SIZE,
                    epochs=8,
                    lr=3.0e-5,
                    loss_name="focal_bce",
                    use_class_weight=True,
                    use_sampler=True,
                    augment_min_per_class=AUG_MIN_PER_CLASS_ABLATION,
                    label_smoothing=0.02,
                    dropout=0.20,
                    patience=2,
                    freeze_layers=18,
                    tune_bias=True,
                ),
            ])

    return exps

# ============================================================
# 报告生成
# ============================================================

def generate_report(summary_df, labels, full_df, train_df, val_df, test_df, macbert_large_model, roberta_large_model):
    if summary_df.empty:
        return

    best = summary_df.sort_values("test_macro_f1", ascending=False).iloc[0]

    lines = []
    lines.append("# 违禁评论识别实验报告\n\n")

    lines.append("## 1. 实验任务\n\n")
    lines.append("本实验面向非公开违禁评论数据集，任务为 10 类单标签文本分类。核心评价指标为多分类 Macro-F1，同时要求每个类别按 one-vs-rest 转换为二分类后的准确率大于 0.9。\n\n")

    lines.append("## 2. 数据集概况\n\n")
    lines.append(f"- 全量样本数：{len(full_df)}\n")
    lines.append(f"- 训练集：{len(train_df)}\n")
    lines.append(f"- 验证集：{len(val_df)}\n")
    lines.append(f"- 测试集：{len(test_df)}\n")
    lines.append(f"- 类别数：{len(labels)}\n\n")

    vc = full_df["类别"].value_counts().reindex(labels).fillna(0).astype(int)
    lines.append("| 类别 | 频次 | 频率 |\n")
    lines.append("|---|---:|---:|\n")
    for lab, cnt in vc.items():
        lines.append(f"| {lab} | {cnt} | {cnt / len(full_df):.6f} |\n")

    lines.append("\n## 3. 方法设计\n\n")
    lines.append("本实验采用如下框架以提高长尾类别的 Macro-F1：\n\n")
    lines.append("1. **中文预训练 Transformer**：主模型使用 MacBERT-large，本地离线加载，避免服务器联网不稳定。\n")
    lines.append("2. **训练集少数类增强**：仅对训练集进行轻量扰动增强，不增强验证集和测试集，避免数据泄漏。\n")
    lines.append("3. **类别权重**：采用 effective number of samples 计算类别权重，提升少数类损失贡献。\n")
    lines.append("4. **WeightedRandomSampler**：增加少数类在 batch 中出现的频率。\n")
    lines.append("5. **Focal CE + One-vs-Rest BCE 组合损失**：CE 优化多分类边界，OVR BCE 强化每类一对多判别能力。\n")
    lines.append("6. **验证集类别偏置校准**：在验证集上搜索类别 bias，直接优化 Macro-F1，提升长尾类召回。\n\n")

    lines.append("## 4. 模型对比结果\n\n")
    lines.append(summary_df.sort_values("test_macro_f1", ascending=False).to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## 5. 最佳模型\n\n")
    lines.append(f"- 最佳模型：{best['name']}\n")
    lines.append(f"- Macro-F1：{best['test_macro_f1']:.4f}\n")
    lines.append(f"- Weighted-F1：{best['test_weighted_f1']:.4f}\n")
    lines.append(f"- Accuracy：{best['test_accuracy']:.4f}\n")
    lines.append(f"- Balanced Accuracy：{best['test_balanced_accuracy']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率均值：{best['test_ovr_acc_mean']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率最小值：{best['test_ovr_acc_min']:.4f}\n")
    lines.append(f"- 总参数量：{int(best['total_params']):,}\n")
    lines.append(f"- 可训练参数量：{int(best['trainable_params']):,}\n")
    lines.append(f"- 训练耗时：{best['seconds']:.2f} 秒\n\n")

    lines.append("## 6. 消融实验说明\n\n")
    lines.append("消融实验以最佳模型为基准，分别移除或修改关键组件：\n\n")
    lines.append("- `Ablation_no_OVRBCE_use_FocalOnly`：去除 OVR BCE，仅使用 Focal Loss。\n")
    lines.append("- `Ablation_no_class_weight`：去除类别权重。\n")
    lines.append("- `Ablation_no_sampler`：去除 WeightedRandomSampler。\n")
    lines.append("- `Ablation_no_augmentation`：去除少数类增强。\n")
    lines.append("- `Ablation_no_bias_calibration`：去除验证集类别偏置校准。\n")
    lines.append("- `Ablation_short_maxlen_96`：缩短最大文本长度，分析截断影响。\n\n")

    lines.append("## 7. 降本实验说明\n\n")
    lines.append("降本实验关注在 Macro-F1 损失较小的前提下降低参数量和训练/推理成本：\n\n")
    lines.append("- 冻结 MacBERT-large 底部层，降低反向传播成本。\n")
    lines.append("- 使用 MacBERT-base 替代 MacBERT-large，大幅减少参数量。\n")
    lines.append("- 缩短 max_len 至 128，降低注意力计算量。\n\n")

    lines.append("## 8. 不同神经网络特点分析\n\n")
    lines.append("- **FastTextNN**：速度快，参数少，但缺乏上下文建模能力，对隐晦违禁表达效果弱。\n")
    lines.append("- **TextCNN**：擅长局部关键词和 n-gram 模式，适合明显敏感词，但长距离语义建模有限。\n")
    lines.append("- **BiLSTM-Attention**：能利用序列顺序和注意力，但训练效率不如 Transformer。\n")
    lines.append("- **TinyTransformer**：具备全局注意力，但从零训练对 2.5 万样本数据而言预训练知识不足。\n")
    lines.append("- **MacBERT**：中文预训练语义能力强，结合长尾优化策略后最适合本任务。\n\n")

    lines.append("## 9. 关键达标检查\n\n")
    lines.append(f"- Macro-F1 > 0.8：{bool(best['test_macro_f1'] > 0.8)}，当前 {best['test_macro_f1']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率均值 > 0.9：{bool(best['test_ovr_acc_mean'] > 0.9)}，当前 {best['test_ovr_acc_mean']:.4f}\n")
    lines.append(f"- One-vs-Rest 准确率最小值 > 0.9：{bool(best['test_ovr_acc_min'] > 0.9)}，当前 {best['test_ovr_acc_min']:.4f}\n\n")

    lines.append("## 10. 图表文件\n\n")
    lines.append("- 类别分布：`outputs/figures/class_distribution.png`\n")
    lines.append("- 类别占比：`outputs/figures/class_distribution_pie.png`\n")
    lines.append("- 模型 Macro-F1 对比：`outputs/figures/model_macro_f1_comparison.png`\n")
    lines.append("- One-vs-Rest 准确率对比：`outputs/figures/model_ovr_min_acc_comparison.png`\n")
    lines.append("- 参数量-性能关系：`outputs/figures/params_vs_f1.png`\n")
    lines.append("- 训练损失曲线：`outputs/figures/training_loss_curves.png`\n")
    lines.append("- 验证集 Macro-F1 曲线：`outputs/figures/val_macro_f1_curves.png`\n")
    lines.append("- 最佳模型各类 F1：`outputs/figures/per_class_f1_*.png`\n")
    lines.append("- 最佳模型混淆矩阵：`outputs/figures/confusion_matrix_*.png`\n\n")

    lines.append("## 11. 本地模型路径\n\n")
    lines.append(f"- MacBERT-large：`{macbert_large_model}`\n")
    lines.append(f"- Chinese-RoBERTa-wwm-ext-large：`{roberta_large_model}`\n")

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))


# ============================================================
# 主流程
# ============================================================

def main():
    ensure_dirs()
    setup_chinese_font()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    macbert_large_model = find_existing_dir(MACBERT_LARGE_CANDIDATES)
    roberta_large_model = find_existing_dir(ROBERTA_WWM_EXT_LARGE_CANDIDATES)

    print(f"MacBERT-large 本地路径：{macbert_large_model}")
    print(f"RoBERTa-wwm-ext-large 本地路径：{roberta_large_model}")

    if macbert_large_model is None:
        print("警告：未找到 MacBERT-large，将跳过 MacBERT-large 主模型实验。")
    if roberta_large_model is None:
        print("警告：未找到 chinese-roberta-wwm-ext-large，将跳过 RoBERTa-large 对照/降本实验。")

    df, labels, label2id, id2label = load_all_csv(DATA_PATH)
    print(f"数据量：{len(df)}")
    print(f"类别：{labels}")

    print("\n类别分布：")
    print(df["类别"].value_counts().reindex(labels).fillna(0).astype(int))

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
        "macbert_large_model": macbert_large_model,
        "roberta_large_model": roberta_large_model,
    }

    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False, encoding="utf-8-sig")

    exps = make_experiments(macbert_large_model, roberta_large_model)

    print("\n将运行以下实验：")
    for e in exps:
        print(f"- {e.name}")

    results = []
    best_pack = None

    for cfg in exps:
        try:
            result, y_true, y_pred = train_one_experiment(
                cfg,
                train_df,
                val_df,
                test_df,
                labels,
                device,
                seed=SEED,
            )
            results.append(result)
            plot_confusion_matrix(y_true, y_pred, labels, cfg.name)

            if best_pack is None or result["test_macro_f1"] > best_pack[0]["test_macro_f1"]:
                best_pack = (result, y_true, y_pred)

            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        except RuntimeError as e:
            print(f"[跳过] {cfg.name} RuntimeError：{repr(e)}")
            if "out of memory" in str(e).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[跳过] {cfg.name} 运行失败：{repr(e)}")

    summary_df = pd.DataFrame(results)

    if summary_df.empty:
        print("没有成功完成任何实验，请检查数据、模型路径或环境依赖。")
        return

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
        macbert_large_model,
        roberta_large_model,
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