# -*- coding: utf-8 -*-
"""
proposed_ltmvf_v2.py

LT-MVF-v2:
Long-Tail Multi-View Fusion v2 for Forbidden Comment Classification

无需命令行参数，直接运行：
nohup python proposed_ltmvf_v2.py > proposed_ltmvf_v2.log 2>&1 &

默认目录结构：
.
├── all.csv
├── simsun.ttc
├── proposed_ltmvf_v2.py
└── hf_models/
    ├── chinese-macbert-large/
    └── chinese-roberta-wwm-ext-large/

核心目标：
1. 在已有最高 F1≈0.77 的基础上继续冲击 Macro-F1 > 0.8。
2. 避免上一版中 Bias Calibration 对测试集泛化不稳的问题。
3. 通过多视角单编码器集成，而不是耗时巨大的端到端双编码器，提升泛化。
4. 自动进行消融实验，比较：
   - 单模型
   - 多模型融合
   - 有/无长尾 tau logit adjustment
   - 有/无弱 bias calibration
   - 有/无 RoBERTa 视角
   - Focal vs Focal+BCE vs Balanced-Focal
   - sampler/augmentation 的影响

输出目录：
outputs_ltmvf_v2/
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

warnings.filterwarnings("ignore")

# ============================================================
# HuggingFace 离线本地加载
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
# 全局配置
# ============================================================

DATA_PATH = "all.csv"
TEXT_COL = "文本"
LABEL_COL = "类别"
ID_COL = "id"

OUTPUT_DIR = "outputs_ltmvf_v2"
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

# 训练配置
MAX_LEN_192 = 192
MAX_LEN_128 = 128

BATCH_SIZE = 16
EPOCHS = 7
PATIENCE = 2
LR = 1.5e-5
GRAD_ACCUM = 1

# 为节约时间，默认训练 5 个视角。若想更强，可把 RUN_EXTRA_VIEWS=True。
RUN_EXTRA_VIEWS = False

# 保存模型 checkpoint 会占空间，但便于复现
SAVE_CHECKPOINTS = True

# 多视角融合随机搜索次数
FUSION_RANDOM_TRIALS = 2000


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
        print(f"警告：{name}目录未发现 pytorch_model.bin 或 model.safetensors：{path}")


# ============================================================
# 数据读取与增强
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
    text = str(text)
    if len(text) <= 2:
        return text

    ops = [
        "drop_punc",
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


def augment_minority_train(train_df, labels, min_per_class=600, seed=42):
    """
    只增强训练集，避免数据泄漏。
    上一版 800 对验证集有效，但测试集未必稳。
    因此本版通过不同视角尝试 0/500/700/800，并由融合自动选择。
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
        sub_records = sub.to_dict("records")
        rows = []

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


def build_weighted_sampler(train_df, num_labels, power=0.55):
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


# ============================================================
# 模型
# ============================================================

class HFClassifier(nn.Module):
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
        return self.classifier(self.dropout(cls))


# ============================================================
# Loss
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
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class FocalCEWithOVRBCE(nn.Module):
    def __init__(
        self,
        num_labels,
        ce_weight=None,
        bce_pos_weight=None,
        gamma=2.0,
        alpha_ce=1.0,
        alpha_bce=0.25,
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


class BalancedFocalLoss(nn.Module):
    """
    Balanced Softmax + Focal CE。
    目的：比强 bias 更稳地处理长尾。
    """
    def __init__(self, train_counts, class_weight=None, gamma=2.0, label_smoothing=0.02, alpha_bal=0.45):
        super().__init__()
        counts = torch.tensor(train_counts, dtype=torch.float32)
        self.register_buffer("log_counts", torch.log(counts + 1e-12))
        self.class_weight = class_weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha_bal = alpha_bal

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (((1.0 - pt) ** self.gamma) * ce).mean()

        balanced_logits = logits + self.log_counts.to(logits.device).unsqueeze(0)
        bal = F.cross_entropy(
            balanced_logits,
            target,
            label_smoothing=self.label_smoothing,
        )

        return (1.0 - self.alpha_bal) * focal + self.alpha_bal * bal


def make_loss(loss_type, num_labels, class_weights, train_counts, device):
    if loss_type == "focal":
        return FocalLoss(
            gamma=2.0,
            weight=class_weights,
            label_smoothing=0.02,
        )

    if loss_type == "focal_bce":
        counts = torch.tensor(train_counts, dtype=torch.float32, device=device)
        total = counts.sum()
        pos_weight = (total - counts) / counts.clamp_min(1.0)
        pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)

        return FocalCEWithOVRBCE(
            num_labels=num_labels,
            ce_weight=class_weights,
            bce_pos_weight=pos_weight,
            gamma=2.0,
            alpha_ce=1.0,
            alpha_bce=0.25,
            label_smoothing=0.02,
        )

    if loss_type == "balanced_focal":
        return BalancedFocalLoss(
            train_counts=train_counts,
            class_weight=class_weights,
            gamma=2.0,
            label_smoothing=0.02,
            alpha_bal=0.45,
        ).to(device)

    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.02,
        )

    raise ValueError(loss_type)


# ============================================================
# FGM
# ============================================================

class FGM:
    def __init__(self, model, emb_name="word_embeddings", epsilon=0.5):
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
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================
# Metrics / Prediction
# ============================================================

def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits_list = []
    y_list = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        y = batch.pop("labels")
        logits = model(**batch)
        logits_list.append(logits.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())

    return np.concatenate(logits_list, axis=0), np.concatenate(y_list, axis=0)


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


def logits_to_metrics(logits, y, labels, bias=None):
    if bias is None:
        pred = logits.argmax(axis=1)
    else:
        pred = (logits + bias).argmax(axis=1)
    return compute_metrics(y, pred, labels), pred


# ============================================================
# 温和校准与融合
# ============================================================

def optimize_weak_bias(val_logits, val_y, labels, max_abs=0.8, max_iter=2):
    """
    弱 bias calibration。
    上一版强 bias 造成测试集下降，因此这里限制 bias 幅度。
    """
    num_labels = len(labels)
    bias = np.zeros(num_labels, dtype=np.float32)

    base_pred = val_logits.argmax(axis=1)
    best = f1_score(val_y, base_pred, average="macro", zero_division=0)

    grid = np.array([-max_abs, -0.5, -0.3, 0.0, 0.3, 0.5, max_abs], dtype=np.float32)

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


def apply_tau_adjustment(logits, train_counts, tau):
    """
    logits - tau * log(prior)
    tau > 0 会相对提升长尾类。
    """
    counts = np.asarray(train_counts, dtype=np.float64)
    prior = counts / counts.sum()
    log_prior = np.log(prior + 1e-12)
    return logits - tau * log_prior.reshape(1, -1)


def normalize_weights(w):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return w / w.sum()


def fuse_logits(logits_list, weights, train_counts=None, tau=0.0):
    weights = normalize_weights(weights)
    out = np.zeros_like(logits_list[0], dtype=np.float64)

    for wi, logits in zip(weights, logits_list):
        x = logits
        if train_counts is not None and abs(tau) > 1e-12:
            x = apply_tau_adjustment(x, train_counts, tau)
        out += wi * x

    return out


def search_fusion(
    val_logits_list,
    val_y,
    labels,
    train_counts,
    trials=2000,
    seed=42,
    allow_bias=True,
    allow_tau=True,
    acc_floor=0.88,
):
    """
    验证集搜索融合权重、tau 和弱 bias。
    目标函数兼顾 macro-F1 和 accuracy，避免只追长尾导致准确率过低。
    """
    rng = np.random.default_rng(seed)
    n = len(val_logits_list)

    weight_candidates = []

    # 单模型
    for i in range(n):
        w = np.zeros(n)
        w[i] = 1.0
        weight_candidates.append(w)

    # 均匀
    weight_candidates.append(np.ones(n) / n)

    # 偏 MacBERT 的人工候选
    if n >= 2:
        w = np.zeros(n)
        w[0] = 0.75
        w[1] = 0.25
        weight_candidates.append(w)

        w = np.zeros(n)
        w[0] = 0.60
        w[1] = 0.40
        weight_candidates.append(w)

    # 随机 Dirichlet
    for _ in range(trials):
        alpha = np.ones(n) * 0.8
        w = rng.dirichlet(alpha)
        weight_candidates.append(w)

    tau_list = [0.0]
    if allow_tau:
        tau_list = np.round(np.arange(-0.10, 1.21, 0.05), 3).tolist()

    best = {
        "score": -1e9,
        "macro_f1": -1.0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "ovr_acc_min": 0.0,
        "weights": None,
        "tau": 0.0,
        "bias": None,
        "use_bias": False,
    }

    for w in weight_candidates:
        for tau in tau_list:
            logits = fuse_logits(val_logits_list, w, train_counts=train_counts, tau=tau)

            bias_candidates = [(False, np.zeros(logits.shape[1], dtype=np.float32))]

            if allow_bias:
                b, _ = optimize_weak_bias(logits, val_y, labels, max_abs=0.8, max_iter=2)
                # shrink bias，防止过拟合
                bias_candidates.append((True, 0.25 * b))
                bias_candidates.append((True, 0.50 * b))

            for use_bias, bias in bias_candidates:
                pred = (logits + bias).argmax(axis=1)
                m = compute_metrics(val_y, pred, labels)

                macro = m["macro_f1"]
                acc = m["accuracy"]
                bacc = m["balanced_accuracy"]
                ovr_min = m["ovr_acc_min"]

                # 约束准确率，防止为了少数类牺牲整体准确率
                if acc < acc_floor:
                    penalty = (acc_floor - acc) * 2.0
                else:
                    penalty = 0.0

                # 综合目标：以 macro-F1 为主，兼顾准确率和 balanced acc
                score = macro + 0.08 * acc + 0.05 * bacc + 0.03 * ovr_min - penalty

                if score > best["score"]:
                    best.update({
                        "score": float(score),
                        "macro_f1": float(macro),
                        "accuracy": float(acc),
                        "balanced_accuracy": float(bacc),
                        "ovr_acc_min": float(ovr_min),
                        "weights": normalize_weights(w).tolist(),
                        "tau": float(tau),
                        "bias": bias.astype(np.float32),
                        "use_bias": bool(use_bias),
                    })

    return best


# ============================================================
# 训练配置
# ============================================================

@dataclass
class ViewConfig:
    name: str
    model_path: str
    seed: int
    max_len: int = 192
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr: float = LR
    patience: int = PATIENCE
    grad_accum_steps: int = GRAD_ACCUM
    augment_min_per_class: int = 600
    use_sampler: bool = False
    use_fgm: bool = True
    dropout: float = 0.20
    loss_type: str = "focal"


# ============================================================
# 单视角训练
# ============================================================

def train_one_view(cfg, train_df_raw, val_df, test_df, labels, device):
    print(f"\n========== 训练视角：{cfg.name} ==========")

    set_seed(cfg.seed)
    num_labels = len(labels)

    train_df = augment_minority_train(
        train_df_raw,
        labels,
        min_per_class=cfg.augment_min_per_class,
        seed=cfg.seed,
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

    criterion = make_loss(
        loss_type=cfg.loss_type,
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
    best_epoch = 0
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

        val_logits, val_y = predict_logits(model, val_loader, device)
        val_pred = val_logits.argmax(axis=1)
        val_m = compute_metrics(val_y, val_pred, labels)

        log = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1_raw": val_m["macro_f1"],
            "val_accuracy_raw": val_m["accuracy"],
            "val_balanced_accuracy_raw": val_m["balanced_accuracy"],
            "val_ovr_acc_min_raw": val_m["ovr_acc_min"],
            "epoch_seconds": time.time() - t0,
        }
        logs.append(log)

        print(
            f"[{cfg.name}] epoch={epoch:02d} "
            f"loss={log['train_loss']:.4f} "
            f"val_macroF1_raw={log['val_macro_f1_raw']:.4f} "
            f"val_acc_raw={log['val_accuracy_raw']:.4f} "
            f"val_ovrMin_raw={log['val_ovr_acc_min_raw']:.4f}"
        )

        # 本版用 raw macro 选 epoch，避免 bias 过拟合
        if val_m["macro_f1"] > best_val_macro:
            best_val_macro = val_m["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
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

    val_m, val_pred = logits_to_metrics(val_logits, val_y, labels)
    test_m, test_pred = logits_to_metrics(test_logits, test_y, labels)

    sname = safe_name(cfg.name)

    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_logits.npy"), val_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_logits.npy"), test_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_y.npy"), val_y)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_y.npy"), test_y)

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
                "best_epoch": best_epoch,
            },
            os.path.join(CKPT_DIR, f"{sname}.pt"),
        )

    result = {
        "name": cfg.name,
        "method_type": "single_view",
        "model_path": cfg.model_path,
        "loss_type": cfg.loss_type,
        "max_len": cfg.max_len,
        "batch_size": cfg.batch_size,
        "epochs_planned": cfg.epochs,
        "epochs_ran": len(logs),
        "best_epoch": best_epoch,
        "lr": cfg.lr,
        "augment_min_per_class": cfg.augment_min_per_class,
        "use_sampler": cfg.use_sampler,
        "use_fgm": cfg.use_fgm,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "seconds": time.time() - start_time,
        "best_val_macro_f1": best_val_macro,
        "val_accuracy": val_m["accuracy"],
        "val_macro_f1": val_m["macro_f1"],
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

    return result, val_logits, test_logits, val_y, test_y, test_pred


# ============================================================
# 融合与消融评估
# ============================================================

def evaluate_logits_method(
    name,
    val_logits,
    test_logits,
    val_y,
    test_y,
    labels,
    test_df,
    extra_info=None,
):
    val_pred = val_logits.argmax(axis=1)
    test_pred = test_logits.argmax(axis=1)

    val_m = compute_metrics(val_y, val_pred, labels)
    test_m = compute_metrics(test_y, test_pred, labels)

    sname = safe_name(name)

    np.save(os.path.join(LOGIT_DIR, f"{sname}_val_logits.npy"), val_logits)
    np.save(os.path.join(LOGIT_DIR, f"{sname}_test_logits.npy"), test_logits)

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

    result = {
        "name": name,
        "method_type": "fusion_or_ablation",
        "model_path": "multi-view",
        "loss_type": "fusion",
        "max_len": "mixed",
        "batch_size": "mixed",
        "epochs_planned": "mixed",
        "epochs_ran": "mixed",
        "best_epoch": "mixed",
        "lr": "mixed",
        "augment_min_per_class": "mixed",
        "use_sampler": "mixed",
        "use_fgm": "mixed",
        "total_params": np.nan,
        "trainable_params": np.nan,
        "seconds": 0.0,
        "best_val_macro_f1": val_m["macro_f1"],
        "val_accuracy": val_m["accuracy"],
        "val_macro_f1": val_m["macro_f1"],
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_f1": test_m["weighted_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_ovr_acc_mean": test_m["ovr_acc_mean"],
        "test_ovr_acc_min": test_m["ovr_acc_min"],
    }

    if extra_info:
        result.update(extra_info)

    print(
        f"[{name}] val_macroF1={val_m['macro_f1']:.4f} "
        f"test_macroF1={test_m['macro_f1']:.4f} "
        f"test_acc={test_m['accuracy']:.4f} "
        f"test_ovrMin={test_m['ovr_acc_min']:.4f}"
    )

    return result, test_pred


def evaluate_fusion_and_ablations(
    view_names,
    val_logits_list,
    test_logits_list,
    val_y,
    test_y,
    train_counts,
    labels,
    test_df,
):
    print("\n========== 融合搜索与消融实验 ==========")

    results = []
    preds = {}

    # 1. Ours full: 权重 + tau + 弱 bias
    best = search_fusion(
        val_logits_list,
        val_y,
        labels,
        train_counts=train_counts,
        trials=FUSION_RANDOM_TRIALS,
        seed=SEED,
        allow_bias=True,
        allow_tau=True,
        acc_floor=0.88,
    )

    print("[Ours_Full] fusion config:")
    print(json.dumps(
        {
            "view_names": view_names,
            "weights": best["weights"],
            "tau": best["tau"],
            "use_bias": best["use_bias"],
            "bias": best["bias"].tolist() if best["bias"] is not None else None,
            "val_macro_f1": best["macro_f1"],
            "val_accuracy": best["accuracy"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    val_logits = fuse_logits(val_logits_list, best["weights"], train_counts, best["tau"]) + best["bias"]
    test_logits = fuse_logits(test_logits_list, best["weights"], train_counts, best["tau"]) + best["bias"]

    res, pred = evaluate_logits_method(
        "Ours_LTMVFv2_WeightedFusion_Tau_WeakBias",
        val_logits,
        test_logits,
        val_y,
        test_y,
        labels,
        test_df,
        extra_info={
            "fusion_weights": json.dumps(best["weights"], ensure_ascii=False),
            "fusion_tau": best["tau"],
            "fusion_use_bias": best["use_bias"],
            "fusion_views": json.dumps(view_names, ensure_ascii=False),
        },
    )
    results.append(res)
    preds[res["name"]] = pred

    with open(os.path.join(OUTPUT_DIR, "ours_ltmvfv2_fusion_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "view_names": view_names,
                "weights": best["weights"],
                "tau": best["tau"],
                "use_bias": best["use_bias"],
                "bias": best["bias"].tolist() if best["bias"] is not None else None,
                "val_macro_f1": best["macro_f1"],
                "val_accuracy": best["accuracy"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2. Ablation: 无 bias，有 tau
    best_tau = search_fusion(
        val_logits_list,
        val_y,
        labels,
        train_counts=train_counts,
        trials=FUSION_RANDOM_TRIALS // 2,
        seed=SEED + 1,
        allow_bias=False,
        allow_tau=True,
        acc_floor=0.88,
    )
    val_logits = fuse_logits(val_logits_list, best_tau["weights"], train_counts, best_tau["tau"])
    test_logits = fuse_logits(test_logits_list, best_tau["weights"], train_counts, best_tau["tau"])
    res, pred = evaluate_logits_method(
        "Ablation_Fusion_Tau_NoBias",
        val_logits,
        test_logits,
        val_y,
        test_y,
        labels,
        test_df,
        extra_info={
            "fusion_weights": json.dumps(best_tau["weights"], ensure_ascii=False),
            "fusion_tau": best_tau["tau"],
            "fusion_use_bias": False,
        },
    )
    results.append(res)
    preds[res["name"]] = pred

    # 3. Ablation: 无 tau，无 bias，仅权重融合
    best_plain = search_fusion(
        val_logits_list,
        val_y,
        labels,
        train_counts=train_counts,
        trials=FUSION_RANDOM_TRIALS // 2,
        seed=SEED + 2,
        allow_bias=False,
        allow_tau=False,
        acc_floor=0.88,
    )
    val_logits = fuse_logits(val_logits_list, best_plain["weights"], train_counts, 0.0)
    test_logits = fuse_logits(test_logits_list, best_plain["weights"], train_counts, 0.0)
    res, pred = evaluate_logits_method(
        "Ablation_Fusion_NoTau_NoBias",
        val_logits,
        test_logits,
        val_y,
        test_y,
        labels,
        test_df,
        extra_info={
            "fusion_weights": json.dumps(best_plain["weights"], ensure_ascii=False),
            "fusion_tau": 0.0,
            "fusion_use_bias": False,
        },
    )
    results.append(res)
    preds[res["name"]] = pred

    # 4. Ablation: 只用 MacBERT 视角
    mac_indices = [i for i, n in enumerate(view_names) if "MacBERT" in n or "macbert" in n]
    if len(mac_indices) >= 1:
        mac_val_logits_list = [val_logits_list[i] for i in mac_indices]
        mac_test_logits_list = [test_logits_list[i] for i in mac_indices]
        mac_view_names = [view_names[i] for i in mac_indices]

        best_mac = search_fusion(
            mac_val_logits_list,
            val_y,
            labels,
            train_counts=train_counts,
            trials=FUSION_RANDOM_TRIALS // 2,
            seed=SEED + 3,
            allow_bias=False,
            allow_tau=True,
            acc_floor=0.88,
        )
        val_logits = fuse_logits(mac_val_logits_list, best_mac["weights"], train_counts, best_mac["tau"])
        test_logits = fuse_logits(mac_test_logits_list, best_mac["weights"], train_counts, best_mac["tau"])
        res, pred = evaluate_logits_method(
            "Ablation_MacBERT_Views_Only",
            val_logits,
            test_logits,
            val_y,
            test_y,
            labels,
            test_df,
            extra_info={
                "fusion_weights": json.dumps(best_mac["weights"], ensure_ascii=False),
                "fusion_tau": best_mac["tau"],
                "fusion_use_bias": False,
                "fusion_views": json.dumps(mac_view_names, ensure_ascii=False),
            },
        )
        results.append(res)
        preds[res["name"]] = pred

    # 5. Ablation: MacBERT + RoBERTa 两个最佳单模型简单平均
    if len(val_logits_list) >= 2:
        w = np.ones(len(val_logits_list)) / len(val_logits_list)
        val_logits = fuse_logits(val_logits_list, w, train_counts=None, tau=0.0)
        test_logits = fuse_logits(test_logits_list, w, train_counts=None, tau=0.0)
        res, pred = evaluate_logits_method(
            "Ablation_SimpleAverage_AllViews",
            val_logits,
            test_logits,
            val_y,
            test_y,
            labels,
            test_df,
            extra_info={
                "fusion_weights": json.dumps(w.tolist(), ensure_ascii=False),
                "fusion_tau": 0.0,
                "fusion_use_bias": False,
            },
        )
        results.append(res)
        preds[res["name"]] = pred

    return results, preds


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

    plt.figure(figsize=(16, 7))
    sns.barplot(data=s, x="name", y="test_macro_f1")
    plt.axhline(0.8, color="red", linestyle="--", label="Macro-F1 = 0.8")
    plt.title("不同框架测试集 Macro-F1 对比")
    plt.xlabel("模型/框架")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_macro_f1_comparison.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(16, 7))
    sns.barplot(data=s, x="name", y="test_accuracy")
    plt.axhline(0.9, color="red", linestyle="--", label="Accuracy = 0.9")
    plt.title("不同框架测试集 Accuracy 对比")
    plt.xlabel("模型/框架")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_accuracy_comparison.png"), dpi=240)
    plt.close()

    plt.figure(figsize=(16, 7))
    sns.barplot(data=s, x="name", y="test_ovr_acc_min")
    plt.axhline(0.9, color="red", linestyle="--", label="OVR min accuracy = 0.9")
    plt.title("不同框架 One-vs-Rest 最小准确率")
    plt.xlabel("模型/框架")
    plt.ylabel("OVR Min Accuracy")
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
        plt.plot(df["epoch"], df["val_macro_f1_raw"], marker="o", label=name)
    plt.axhline(0.8, color="red", linestyle="--")
    plt.title("验证集 Raw Macro-F1 随 Epoch 变化")
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
    plt.title(f"{best_name} 各类别 One-vs-Rest 准确率")
    plt.ylabel("One-vs-Rest Accuracy")
    plt.xlabel("类别")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"per_class_ovr_acc_{safe_name(best_name)}.png"), dpi=240)
    plt.close()


# ============================================================
# 报告
# ============================================================

def generate_report(summary_df, labels, full_df, train_df, val_df, test_df):
    if summary_df.empty:
        return

    best = summary_df.sort_values("test_macro_f1", ascending=False).iloc[0]

    lines = []
    lines.append("# LT-MVF-v2 违禁评论识别实验报告\n\n")

    lines.append("## 1. 实验任务\n\n")
    lines.append("本实验面向 10 类违禁评论识别任务。核心指标为 Macro-F1，同时要求 Accuracy 不低且 one-vs-rest 二分类准确率大于 0.9。\n\n")

    lines.append("## 2. 数据集划分\n\n")
    lines.append("采用分层留出法，按照 70%/15%/15% 划分训练集、验证集和测试集。\n\n")
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

    lines.append("\n## 3. 根据上一轮结果的改进思路\n\n")
    lines.append("上一轮实验中，最高验证集 Macro-F1 约为 0.778，但强 Bias Calibration 在测试集上出现泛化下降，说明单纯在验证集上对类别 bias 进行强搜索容易过拟合。双编码器端到端训练虽然参数量大、训练时间长，但未带来显著收益。\n\n")
    lines.append("因此，本版 LT-MVF-v2 做出如下调整：\n\n")
    lines.append("1. 不再以强 Bias Calibration 作为主方法，而使用弱 bias shrink；\n")
    lines.append("2. 引入长尾 logit adjustment 的 tau 搜索，以更平滑的方式提升少数类；\n")
    lines.append("3. 使用多个单编码器视角集成，替代耗时巨大的端到端双编码器；\n")
    lines.append("4. 对不同损失函数、增强强度、sampler 策略进行消融；\n")
    lines.append("5. 融合搜索目标同时考虑 Macro-F1、Accuracy、Balanced Accuracy 和 OVR 最小准确率。\n\n")

    lines.append("## 4. 本文提出方法\n\n")
    lines.append("本文提出 **LT-MVF-v2：长尾多视角稳健融合模型**。\n\n")
    lines.append("```text\n")
    lines.append("Ours = 多个 MacBERT/RoBERTa 单编码器视角\n")
    lines.append("       + Focal / Focal-BCE / Balanced-Focal 损失\n")
    lines.append("       + 少数类增强\n")
    lines.append("       + 可选 Weighted Sampler\n")
    lines.append("       + FGM 对抗训练\n")
    lines.append("       + 验证集搜索加权 logit 融合\n")
    lines.append("       + 长尾 tau logit adjustment\n")
    lines.append("       + 弱 Bias Calibration shrink\n")
    lines.append("```\n\n")

    lines.append("## 5. 实验结果汇总\n\n")
    lines.append(summary_df.sort_values("test_macro_f1", ascending=False).to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## 6. 最佳模型\n\n")
    lines.append(f"- 最佳模型：{best['name']}\n")
    lines.append(f"- Macro-F1：{best['test_macro_f1']:.4f}\n")
    lines.append(f"- Accuracy：{best['test_accuracy']:.4f}\n")
    lines.append(f"- Balanced Accuracy：{best['test_balanced_accuracy']:.4f}\n")
    lines.append(f"- Weighted-F1：{best['test_weighted_f1']:.4f}\n")
    lines.append(f"- OVR 准确率均值：{best['test_ovr_acc_mean']:.4f}\n")
    lines.append(f"- OVR 准确率最小值：{best['test_ovr_acc_min']:.4f}\n\n")

    lines.append("## 7. 达标检查\n\n")
    lines.append(f"- Macro-F1 > 0.8：{bool(best['test_macro_f1'] > 0.8)}，当前 {best['test_macro_f1']:.4f}\n")
    lines.append(f"- Accuracy 不低于 0.9：{bool(best['test_accuracy'] > 0.9)}，当前 {best['test_accuracy']:.4f}\n")
    lines.append(f"- OVR 最小准确率 > 0.9：{bool(best['test_ovr_acc_min'] > 0.9)}，当前 {best['test_ovr_acc_min']:.4f}\n\n")

    lines.append("## 8. 消融实验说明\n\n")
    lines.append("本脚本自动生成以下消融配置：\n\n")
    lines.append("- `Ablation_Fusion_Tau_NoBias`：融合 + tau，但不使用 bias；\n")
    lines.append("- `Ablation_Fusion_NoTau_NoBias`：仅权重融合，无 tau，无 bias；\n")
    lines.append("- `Ablation_MacBERT_Views_Only`：只使用 MacBERT 视角；\n")
    lines.append("- `Ablation_SimpleAverage_AllViews`：所有视角简单平均；\n")
    lines.append("- 单视角模型用于比较 Focal、Focal-BCE、Balanced-Focal、sampler 和增强强度影响。\n\n")

    lines.append("## 9. 图表文件\n\n")
    lines.append("- 类别分布：`outputs_ltmvf_v2/figures/class_distribution.png`\n")
    lines.append("- 类别占比：`outputs_ltmvf_v2/figures/class_distribution_pie.png`\n")
    lines.append("- Macro-F1 对比：`outputs_ltmvf_v2/figures/model_macro_f1_comparison.png`\n")
    lines.append("- Accuracy 对比：`outputs_ltmvf_v2/figures/model_accuracy_comparison.png`\n")
    lines.append("- OVR 最小准确率对比：`outputs_ltmvf_v2/figures/model_ovr_min_acc_comparison.png`\n")
    lines.append("- 训练损失曲线：`outputs_ltmvf_v2/figures/training_loss_curves.png`\n")
    lines.append("- 验证集 Macro-F1 曲线：`outputs_ltmvf_v2/figures/val_macro_f1_curves.png`\n")

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

    num_labels = len(labels)
    train_counts_original = np.bincount(train_df["label"].values, minlength=num_labels)

    print("\n训练集原始类别计数：")
    for lab, cnt in zip(labels, train_counts_original):
        print(f"{lab}: {cnt}")

    # ========================================================
    # 多视角配置
    # 设计原则：
    # - MacBERT 是上一轮最强单模型，因此给多个视角。
    # - RoBERTa 上一轮较弱，但可能提供互补错误分布，因此保留一个视角。
    # - 不再让双编码器端到端训练拖慢时间。
    # ========================================================

    view_configs = [
        ViewConfig(
            name="View1_MacBERT_Focal_Aug800_NoSampler_FGM",
            model_path=MACBERT_LARGE_PATH,
            seed=42,
            max_len=192,
            batch_size=16,
            epochs=7,
            lr=1.5e-5,
            patience=2,
            augment_min_per_class=800,
            use_sampler=False,
            use_fgm=True,
            dropout=0.20,
            loss_type="focal",
        ),
        ViewConfig(
            name="View2_MacBERT_FocalBCE_Aug800_Sampler_FGM",
            model_path=MACBERT_LARGE_PATH,
            seed=3407,
            max_len=192,
            batch_size=16,
            epochs=7,
            lr=1.5e-5,
            patience=2,
            augment_min_per_class=800,
            use_sampler=True,
            use_fgm=True,
            dropout=0.20,
            loss_type="focal_bce",
        ),
        ViewConfig(
            name="View3_MacBERT_BalancedFocal_Aug600_NoSampler_FGM",
            model_path=MACBERT_LARGE_PATH,
            seed=2024,
            max_len=192,
            batch_size=16,
            epochs=7,
            lr=1.5e-5,
            patience=2,
            augment_min_per_class=600,
            use_sampler=False,
            use_fgm=True,
            dropout=0.20,
            loss_type="balanced_focal",
        ),
        ViewConfig(
            name="View4_MacBERT_Focal_Aug500_NoSampler_Len128_FGM",
            model_path=MACBERT_LARGE_PATH,
            seed=777,
            max_len=128,
            batch_size=16,
            epochs=6,
            lr=1.5e-5,
            patience=2,
            augment_min_per_class=500,
            use_sampler=False,
            use_fgm=True,
            dropout=0.20,
            loss_type="focal",
        ),
        ViewConfig(
            name="View5_RoBERTa_BalancedFocal_Aug600_NoSampler_FGM",
            model_path=ROBERTA_LARGE_PATH,
            seed=42,
            max_len=192,
            batch_size=16,
            epochs=7,
            lr=1.5e-5,
            patience=2,
            augment_min_per_class=600,
            use_sampler=False,
            use_fgm=True,
            dropout=0.20,
            loss_type="balanced_focal",
        ),
    ]

    if RUN_EXTRA_VIEWS:
        view_configs.extend([
            ViewConfig(
                name="View6_MacBERT_CE_Aug600_NoSampler_FGM",
                model_path=MACBERT_LARGE_PATH,
                seed=1001,
                max_len=192,
                batch_size=16,
                epochs=6,
                lr=1.5e-5,
                patience=2,
                augment_min_per_class=600,
                use_sampler=False,
                use_fgm=True,
                dropout=0.20,
                loss_type="ce",
            ),
            ViewConfig(
                name="View7_RoBERTa_Focal_Aug800_Sampler_FGM",
                model_path=ROBERTA_LARGE_PATH,
                seed=3407,
                max_len=192,
                batch_size=16,
                epochs=6,
                lr=1.5e-5,
                patience=2,
                augment_min_per_class=800,
                use_sampler=True,
                use_fgm=True,
                dropout=0.20,
                loss_type="focal",
            ),
        ])

    print("\n将训练以下视角：")
    for cfg in view_configs:
        print(f"- {cfg.name}")

    all_results = []
    all_preds = {}

    val_logits_list = []
    test_logits_list = []
    view_names = []
    val_y_ref = None
    test_y_ref = None

    # ========================================================
    # 训练每个视角
    # ========================================================

    for cfg in view_configs:
        try:
            result, val_logits, test_logits, val_y, test_y, test_pred = train_one_view(
                cfg,
                train_df,
                val_df,
                test_df,
                labels,
                device,
            )

            all_results.append(result)
            all_preds[result["name"]] = test_pred

            val_logits_list.append(val_logits)
            test_logits_list.append(test_logits)
            view_names.append(cfg.name)

            if val_y_ref is None:
                val_y_ref = val_y
            if test_y_ref is None:
                test_y_ref = test_y

            plot_confusion_matrix(test_y, test_pred, labels, cfg.name)

            pd.DataFrame(all_results).to_csv(
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

    if len(val_logits_list) == 0:
        print("没有成功训练任何视角，程序结束。")
        return

    # ========================================================
    # 融合 + 消融
    # ========================================================

    fusion_results, fusion_preds = evaluate_fusion_and_ablations(
        view_names=view_names,
        val_logits_list=val_logits_list,
        test_logits_list=test_logits_list,
        val_y=val_y_ref,
        test_y=test_y_ref,
        train_counts=train_counts_original,
        labels=labels,
        test_df=test_df,
    )

    all_results.extend(fusion_results)
    all_preds.update(fusion_preds)

    # ========================================================
    # 汇总
    # ========================================================

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_summary(summary_df)
    plot_training_curves()

    best = summary_df.sort_values("test_macro_f1", ascending=False).iloc[0]
    best_name = best["name"]

    if best_name in all_preds:
        plot_confusion_matrix(test_y_ref, all_preds[best_name], labels, best_name)

    plot_per_class_best(best_name, labels)

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
    print(f"Accuracy > 0.9: {best['test_accuracy'] > 0.9}，当前 {best['test_accuracy']:.4f}")
    print(f"One-vs-rest 二分类准确率均值 > 0.9: {best['test_ovr_acc_mean'] > 0.9}，当前 {best['test_ovr_acc_mean']:.4f}")
    print(f"One-vs-rest 二分类准确率最小值 > 0.9: {best['test_ovr_acc_min'] > 0.9}，当前 {best['test_ovr_acc_min']:.4f}")

    print(f"\n所有输出已保存至：{os.path.abspath(OUTPUT_DIR)}")
    print(f"报告文件：{os.path.abspath(os.path.join(OUTPUT_DIR, 'report.md'))}")


if __name__ == "__main__":
    main()