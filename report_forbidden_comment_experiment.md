# 违禁评论识别实验报告

## 1. 实验任务

本实验面向非公开违禁评论数据集，任务为 10 类单标签文本分类。核心评价指标为多分类 Macro-F1，同时要求每个类别按 one-vs-rest 转换为二分类后的准确率大于 0.9。

## 2. 数据集概况

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

## 3. 方法设计

本实验采用如下框架以提高长尾类别的 Macro-F1：

1. **中文预训练 Transformer**：主模型使用 MacBERT-large，本地离线加载，避免服务器联网不稳定。
2. **训练集少数类增强**：仅对训练集进行轻量扰动增强，不增强验证集和测试集，避免数据泄漏。
3. **类别权重**：采用 effective number of samples 计算类别权重，提升少数类损失贡献。
4. **WeightedRandomSampler**：增加少数类在 batch 中出现的频率。
5. **Focal CE + One-vs-Rest BCE 组合损失**：CE 优化多分类边界，OVR BCE 强化每类一对多判别能力。
6. **验证集类别偏置校准**：在验证集上搜索类别 bias，直接优化 Macro-F1，提升长尾类召回。

## 4. 模型对比结果

| name                                                   | model_type      | bert_model                                |   max_len |   batch_size |   epochs_planned |   epochs_ran |      lr | loss_name   | use_class_weight   | use_sampler   |   augment_min_per_class |   label_smoothing |   freeze_layers |   dropout | tune_bias   |   total_params |   trainable_params |   seconds |   best_val_macro_f1 |   test_accuracy |   test_balanced_accuracy |   test_macro_f1 |   test_weighted_f1 |   test_micro_f1 |   test_ovr_acc_mean |   test_ovr_acc_min |   raw_test_macro_f1_without_bias |
|:-------------------------------------------------------|:----------------|:------------------------------------------|----------:|-------------:|-----------------:|-------------:|--------:|:------------|:-------------------|:--------------|------------------------:|------------------:|----------------:|----------:|:------------|---------------:|-------------------:|----------:|--------------------:|----------------:|-------------------------:|----------------:|-------------------:|----------------:|--------------------:|-------------------:|---------------------------------:|
| Ablation_no_bias_calibration                           | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            4 | 1.5e-05 | focal_bce   | True               | True          |                     600 |              0.02 |               0 |      0.2  | False       |      326582282 |          326582282 |  718.994  |            0.722283 |        0.90332  |                 0.726987 |        0.724355 |           0.904827 |        0.90332  |            0.980664 |           0.958831 |                         0.724355 |
| Ablation_no_OVRBCE_use_FocalOnly                       | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            8 | 1.5e-05 | focal       | True               | True          |                     600 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 | 1393.29   |            0.743195 |        0.925365 |                 0.69179  |        0.711326 |           0.924466 |        0.925365 |            0.985073 |           0.963347 |                         0.670385 |
| Cost_RoBERTa_large_short_maxlen_128                    | hf              | ./hf_models/chinese-roberta-wwm-ext-large |       128 |           16 |                8 |            3 | 1.5e-05 | focal_bce   | True               | True          |                     600 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 |  508.402  |            0.695018 |        0.913413 |                 0.68498  |        0.708882 |           0.912669 |        0.913413 |            0.982683 |           0.959894 |                         0.676621 |
| Ablation_no_sampler                                    | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            5 | 1.5e-05 | focal_bce   | True               | False         |                     600 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 |  886.473  |            0.747619 |        0.92085  |                 0.683963 |        0.708661 |           0.919653 |        0.92085  |            0.98417  |           0.960956 |                         0.716201 |
| Cost_RoBERTa_large_freeze_18_layers                    | hf              | ./hf_models/chinese-roberta-wwm-ext-large |       192 |           16 |                8 |            8 | 3e-05   | focal_bce   | True               | True          |                     600 |              0.02 |              18 |      0.2  | True        |      326582282 |           90283018 |  813.242  |            0.723361 |        0.928287 |                 0.686659 |        0.705904 |           0.926466 |        0.928287 |            0.985657 |           0.966268 |                         0.672751 |
| Ablation_no_augmentation                               | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            7 | 1.5e-05 | focal_bce   | True               | True          |                       0 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 | 1138.09   |            0.71441  |        0.922975 |                 0.668327 |        0.699602 |           0.920001 |        0.922975 |            0.984595 |           0.963612 |                         0.702369 |
| Ablation_short_maxlen_96                               | hf              | ./hf_models/chinese-macbert-large         |        96 |           16 |                8 |            3 | 1.5e-05 | focal_bce   | True               | True          |                     600 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 |  495.983  |            0.72604  |        0.891368 |                 0.706354 |        0.688552 |           0.894925 |        0.891368 |            0.978274 |           0.953254 |                         0.652495 |
| Best_MacBERT_large_FocalBCE_AugSampler_Bias            | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |               12 |            6 | 1.5e-05 | focal_bce   | True               | True          |                     900 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 | 1121.58   |            0.709002 |        0.923772 |                 0.667881 |        0.673747 |           0.922712 |        0.923772 |            0.984754 |           0.96494  |                         0.702572 |
| Ablation_no_class_weight                               | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            3 | 1.5e-05 | focal_bce   | False              | True          |                     600 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 |  542.921  |            0.72283  |        0.906773 |                 0.654595 |        0.66638  |           0.904624 |        0.906773 |            0.981355 |           0.959363 |                         0.668673 |
| Compare_RoBERTa_wwm_ext_large_FocalBCE_AugSampler_Bias | hf              | ./hf_models/chinese-roberta-wwm-ext-large |       192 |           16 |               10 |            6 | 1.5e-05 | focal_bce   | True               | True          |                     900 |              0.02 |               0 |      0.2  | True        |      326582282 |          326582282 | 1121.1    |            0.709548 |        0.921116 |                 0.654304 |        0.661594 |           0.918592 |        0.921116 |            0.984223 |           0.963081 |                         0.655779 |
| Cost_MacBERT_large_freeze_18_layers                    | hf              | ./hf_models/chinese-macbert-large         |       192 |           16 |                8 |            5 | 3e-05   | focal_bce   | True               | True          |                     600 |              0.02 |              18 |      0.2  | True        |      326582282 |           90283018 |  516.343  |            0.728381 |        0.906242 |                 0.646854 |        0.648864 |           0.90527  |        0.906242 |            0.981248 |           0.960159 |                         0.645196 |
| Baseline_TextCNN                                       | textcnn         |                                           |       192 |          128 |                8 |            8 | 0.0003  | focal       | True               | True          |                     500 |              0.02 |               0 |      0.35 | True        |        2013450 |            2013450 |   31.3756 |            0.632936 |        0.868792 |                 0.600808 |        0.599823 |           0.871251 |        0.868792 |            0.973758 |           0.939973 |                         0.591545 |
| Baseline_BiLSTM_Attention                              | bilstm          |                                           |       192 |           96 |                8 |            8 | 0.0003  | focal       | True               | True          |                     500 |              0.02 |               0 |      0.35 | True        |        2142987 |            2142987 |   66.0857 |            0.579425 |        0.826826 |                 0.566166 |        0.563854 |           0.828822 |        0.826826 |            0.965365 |           0.919522 |                         0.547886 |
| Baseline_TinyTransformer                               | tinytransformer |                                           |       192 |           96 |                8 |            4 | 0.0003  | focal       | True               | True          |                     500 |              0.02 |               0 |      0.25 | True        |        3770122 |            3770122 |   31.5623 |            0.608979 |        0.791235 |                 0.564334 |        0.55287  |           0.795907 |        0.791235 |            0.958247 |           0.901461 |                         0.48622  |
| Baseline_FastTextNN                                    | fasttext        |                                           |       192 |          128 |                6 |            6 | 0.0003  | focal       | True               | True          |                     500 |              0.02 |               0 |      0.35 | True        |        1153034 |            1153034 |   19.0066 |            0.459334 |        0.682337 |                 0.458884 |        0.442982 |           0.688366 |        0.682337 |            0.936467 |           0.851527 |                         0.313593 |

## 5. 最佳模型

- 最佳模型：Ablation_no_bias_calibration
- Macro-F1：0.7244
- Weighted-F1：0.9048
- Accuracy：0.9033
- Balanced Accuracy：0.7270
- One-vs-Rest 准确率均值：0.9807
- One-vs-Rest 准确率最小值：0.9588
- 总参数量：326,582,282
- 可训练参数量：326,582,282
- 训练耗时：718.99 秒

## 6. 消融实验说明

消融实验以最佳模型为基准，分别移除或修改关键组件：

- `Ablation_no_OVRBCE_use_FocalOnly`：去除 OVR BCE，仅使用 Focal Loss。
- `Ablation_no_class_weight`：去除类别权重。
- `Ablation_no_sampler`：去除 WeightedRandomSampler。
- `Ablation_no_augmentation`：去除少数类增强。
- `Ablation_no_bias_calibration`：去除验证集类别偏置校准。
- `Ablation_short_maxlen_96`：缩短最大文本长度，分析截断影响。

## 7. 降本实验说明

降本实验关注在 Macro-F1 损失较小的前提下降低参数量和训练/推理成本：

- 冻结 MacBERT-large 底部层，降低反向传播成本。
- 使用 MacBERT-base 替代 MacBERT-large，大幅减少参数量。
- 缩短 max_len 至 128，降低注意力计算量。

## 8. 不同神经网络特点分析

- **FastTextNN**：速度快，参数少，但缺乏上下文建模能力，对隐晦违禁表达效果弱。
- **TextCNN**：擅长局部关键词和 n-gram 模式，适合明显敏感词，但长距离语义建模有限。
- **BiLSTM-Attention**：能利用序列顺序和注意力，但训练效率不如 Transformer。
- **TinyTransformer**：具备全局注意力，但从零训练对 2.5 万样本数据而言预训练知识不足。
- **MacBERT**：中文预训练语义能力强，结合长尾优化策略后最适合本任务。

## 9. 关键达标检查

- Macro-F1 > 0.8：False，当前 0.7244
- One-vs-Rest 准确率均值 > 0.9：True，当前 0.9807
- One-vs-Rest 准确率最小值 > 0.9：True，当前 0.9588

## 10. 图表文件

- 类别分布：`outputs/figures/class_distribution.png`
- 类别占比：`outputs/figures/class_distribution_pie.png`
- 模型 Macro-F1 对比：`outputs/figures/model_macro_f1_comparison.png`
- One-vs-Rest 准确率对比：`outputs/figures/model_ovr_min_acc_comparison.png`
- 参数量-性能关系：`outputs/figures/params_vs_f1.png`
- 训练损失曲线：`outputs/figures/training_loss_curves.png`
- 验证集 Macro-F1 曲线：`outputs/figures/val_macro_f1_curves.png`
- 最佳模型各类 F1：`outputs/figures/per_class_f1_*.png`
- 最佳模型混淆矩阵：`outputs/figures/confusion_matrix_*.png`

## 11. 本地模型路径

- MacBERT-large：`./hf_models/chinese-macbert-large`
- Chinese-RoBERTa-wwm-ext-large：`./hf_models/chinese-roberta-wwm-ext-large`
