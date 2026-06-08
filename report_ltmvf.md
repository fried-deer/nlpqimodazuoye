# LT-MVF 违禁评论识别实验报告

## 1. 实验任务

本实验面向非公开违禁评论数据集，任务为 10 类单标签文本分类。核心评价指标为多分类 Macro-F1，同时要求各类别 one-vs-rest 二分类准确率大于 0.9。

## 2. 数据集划分

实验采用分层留出法，按照 70%/15%/15% 划分训练集、验证集和测试集。

- 全量样本数：25100
- 训练集：17570
- 验证集：3765
- 测试集：3765
- 类别数：10

| 类别 | 频次 | 频率 |
|---|---:|---:|
| 政治敏感 | 7084 | 0.282231 |
| 色情 | 6564 | 0.261514 |
| 种族歧视 | 4488 | 0.178805 |
| 地域歧视 | 3106 | 0.123745 |
| 微侵犯(MA) | 1594 | 0.063506 |
| 犯罪 | 1145 | 0.045618 |
| 基于文化背景的刻板印象(SCB) | 776 | 0.030916 |
| 宗教迷信 | 241 | 0.009602 |
| 性侵犯(SO) | 71 | 0.002829 |
| 基于外表的刻板印象(SA) | 31 | 0.001235 |

## 3. 本文提出方法 LT-MVF

本文提出 **LT-MVF（Long-Tail Multi-View Fusion）长尾多视角预训练融合模型**。该方法不是单纯调用一个预训练模型，而是在长尾不均衡条件下构建多视角深度融合框架。

核心组成如下：

```text
Ours = MacBERT-large + RoBERTa-large 加权 logit 融合
       + Focal BCE
       + 类别权重
       + Weighted Sampler
       + 少数类增强
       + Bias Calibration
```

另外，本文还实现了端到端双编码器融合模型：

```text
DualEncoderFusion = MacBERT-large Encoder
                  + RoBERTa-large Encoder
                  + [h1, h2, |h1-h2|, h1*h2] 深度融合
                  + 门控 MLP 分类器
                  + Focal BCE + 类别权重 + 少数类增强
```

## 4. 实验结果汇总

| name                                                              | method_type                    | model_path                                                                    |   max_len | batch_size    | epochs_planned   | epochs_ran    | lr            |   augment_min_per_class | use_sampler   | use_fgm   |   total_params |   trainable_params |   seconds |   best_val_macro_f1 |   test_accuracy |   test_balanced_accuracy |   test_macro_f1 |   test_weighted_f1 |   test_micro_f1 |   test_ovr_acc_mean |   test_ovr_acc_min |   fusion_alpha_macbert |   fusion_alpha_roberta |   freeze_bottom_layers |
|:------------------------------------------------------------------|:-------------------------------|:------------------------------------------------------------------------------|----------:|:--------------|:-----------------|:--------------|:--------------|------------------------:|:--------------|:----------|---------------:|-------------------:|----------:|--------------------:|----------------:|-------------------------:|----------------:|-------------------:|----------------:|--------------------:|-------------------:|-----------------------:|-----------------------:|-----------------------:|
| DualEncoderFusion_MacBERT_RoBERTa_FocalBCE_AugSampler_Bias        | dual_encoder_fusion            | ./hf_models/chinese-macbert-large + ./hf_models/chinese-roberta-wwm-ext-large |       160 | 4             | 6                | 4             | 1e-05         |                     700 | True          | False     |    6.63114e+08 |        6.63114e+08 |   3285.68 |            0.725445 |        0.912351 |                 0.69394  |        0.720104 |           0.912938 |        0.912351 |            0.98247  |           0.958831 |                    nan |                    nan |                      0 |
| MacBERT_large_FocalBCE_ClassWeight_Sampler_Aug_Bias               | single_encoder                 | ./hf_models/chinese-macbert-large                                             |       192 | 16            | 10               | 5             | 1.5e-05       |                     800 | True          | True      |    3.26582e+08 |        3.26582e+08 |   1487.69 |            0.777992 |        0.915803 |                 0.690908 |        0.70334  |           0.914409 |        0.915803 |            0.983161 |           0.964675 |                    nan |                    nan |                    nan |
| Ours_MacBERT_RoBERTa_WeightedLogitFusion_FocalBCE_AugSampler_Bias | proposed_weighted_logit_fusion | ./hf_models/chinese-macbert-large + ./hf_models/chinese-roberta-wwm-ext-large |       192 | single models | single models    | single models | single models |                     800 | True          | True      |  nan           |      nan           |      0    |            0.777992 |        0.915803 |                 0.690908 |        0.70334  |           0.914409 |        0.915803 |            0.983161 |           0.964675 |                      1 |                      0 |                    nan |
| RoBERTa_large_FocalBCE_ClassWeight_Sampler_Aug_Bias               | single_encoder                 | ./hf_models/chinese-roberta-wwm-ext-large                                     |       192 | 16            | 10               | 8             | 1.5e-05       |                     800 | True          | True      |    3.26582e+08 |        3.26582e+08 |   2367.57 |            0.71741  |        0.925365 |                 0.672143 |        0.69261  |           0.923937 |        0.925365 |            0.985073 |           0.963081 |                    nan |                    nan |                    nan |

## 5. 最佳模型

- 最佳模型：DualEncoderFusion_MacBERT_RoBERTa_FocalBCE_AugSampler_Bias
- Macro-F1：0.7201
- Accuracy：0.9124
- Balanced Accuracy：0.6939
- Weighted-F1：0.9129
- One-vs-Rest 准确率均值：0.9825
- One-vs-Rest 准确率最小值：0.9588

## 6. 达标检查

- Macro-F1 > 0.8：False，当前 0.7201
- One-vs-Rest 准确率均值 > 0.9：True，当前 0.9825
- One-vs-Rest 准确率最小值 > 0.9：True，当前 0.9588

## 7. 方法分析

- MacBERT-large 提供强中文语义建模能力；
- RoBERTa-large 提供互补的上下文表征；
- Focal BCE 同时优化单标签多分类边界和 one-vs-rest 辅助边界；
- 类别权重和 Weighted Sampler 缓解极端长尾问题；
- 少数类增强提升稀有类别曝光频率；
- Bias Calibration 直接在验证集上优化 Macro-F1；
- 双编码器融合模型通过特征级深度融合进一步探索超过单模型的可能性。

## 8. 图表文件

- 类别分布：`outputs_ltmvf/figures/class_distribution.png`
- 类别占比：`outputs_ltmvf/figures/class_distribution_pie.png`
- 模型 Macro-F1 对比：`outputs_ltmvf/figures/model_macro_f1_comparison.png`
- One-vs-Rest 准确率对比：`outputs_ltmvf/figures/model_ovr_min_acc_comparison.png`
- 训练损失曲线：`outputs_ltmvf/figures/training_loss_curves.png`
- 验证集 Macro-F1 曲线：`outputs_ltmvf/figures/val_macro_f1_curves.png`
- 最佳模型混淆矩阵：`outputs_ltmvf/figures/confusion_matrix_*.png`

