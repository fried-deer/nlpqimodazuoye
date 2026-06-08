# LT-MVF-v2 违禁评论识别实验报告

## 1. 实验任务

本实验面向 10 类违禁评论识别任务。核心指标为 Macro-F1，同时要求 Accuracy 不低且 one-vs-rest 二分类准确率大于 0.9。

## 2. 数据集划分

采用分层留出法，按照 70%/15%/15% 划分训练集、验证集和测试集。

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

## 3. 根据上一轮结果的改进思路

上一轮实验中，最高验证集 Macro-F1 约为 0.778，但强 Bias Calibration 在测试集上出现泛化下降，说明单纯在验证集上对类别 bias 进行强搜索容易过拟合。双编码器端到端训练虽然参数量大、训练时间长，但未带来显著收益。

因此，本版 LT-MVF-v2 做出如下调整：

1. 不再以强 Bias Calibration 作为主方法，而使用弱 bias shrink；
2. 引入长尾 logit adjustment 的 tau 搜索，以更平滑的方式提升少数类；
3. 使用多个单编码器视角集成，替代耗时巨大的端到端双编码器；
4. 对不同损失函数、增强强度、sampler 策略进行消融；
5. 融合搜索目标同时考虑 Macro-F1、Accuracy、Balanced Accuracy 和 OVR 最小准确率。

## 4. 本文提出方法

本文提出 **LT-MVF-v2：长尾多视角稳健融合模型**。

```text
Ours = 多个 MacBERT/RoBERTa 单编码器视角
       + Focal / Focal-BCE / Balanced-Focal 损失
       + 少数类增强
       + 可选 Weighted Sampler
       + FGM 对抗训练
       + 验证集搜索加权 logit 融合
       + 长尾 tau logit adjustment
       + 弱 Bias Calibration shrink
```

## 5. 实验结果汇总

| name                                             | method_type        | model_path                                | loss_type      | max_len   | batch_size   | epochs_planned   | epochs_ran   | best_epoch   | lr      | augment_min_per_class   | use_sampler   | use_fgm   |   total_params |   trainable_params |   seconds |   best_val_macro_f1 |   val_accuracy |   val_macro_f1 |   test_accuracy |   test_balanced_accuracy |   test_macro_f1 |   test_weighted_f1 |   test_micro_f1 |   test_ovr_acc_mean |   test_ovr_acc_min | fusion_weights                                                                                             |   fusion_tau |   fusion_use_bias | fusion_views                                                                                                                                                                                                                                         |
|:-------------------------------------------------|:-------------------|:------------------------------------------|:---------------|:----------|:-------------|:-----------------|:-------------|:-------------|:--------|:------------------------|:--------------|:----------|---------------:|-------------------:|----------:|--------------------:|---------------:|---------------:|----------------:|-------------------------:|----------------:|-------------------:|----------------:|--------------------:|-------------------:|:-----------------------------------------------------------------------------------------------------------|-------------:|------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ablation_Fusion_Tau_NoBias                       | fusion_or_ablation | multi-view                                | fusion         | mixed     | mixed        | mixed            | mixed        | mixed        | mixed   | mixed                   | mixed         | mixed     |  nan           |      nan           |     0     |            0.748721 |       0.934396 |       0.748721 |        0.932802 |                 0.697333 |        0.726446 |           0.931375 |        0.932802 |            0.98656  |           0.96494  | [0.0865348595941755, 0.031234044900093568, 0.2868058066127042, 0.5365593494056439, 0.05886593948738278]    |         0.25 |                 0 | nan                                                                                                                                                                                                                                                  |
| Ablation_MacBERT_Views_Only                      | fusion_or_ablation | multi-view                                | fusion         | mixed     | mixed        | mixed            | mixed        | mixed        | mixed   | mixed                   | mixed         | mixed     |  nan           |      nan           |     0     |            0.749029 |       0.934396 |       0.749029 |        0.93174  |                 0.694841 |        0.723797 |           0.930191 |        0.93174  |            0.986348 |           0.963878 | [0.04209673678837047, 0.0713803075560845, 0.34953381564377567, 0.5369891400117693]                         |         0.25 |                 0 | ["View1_MacBERT_Focal_Aug800_NoSampler_FGM", "View2_MacBERT_FocalBCE_Aug800_Sampler_FGM", "View3_MacBERT_BalancedFocal_Aug600_NoSampler_FGM", "View4_MacBERT_Focal_Aug500_NoSampler_Len128_FGM"]                                                     |
| View1_MacBERT_Focal_Aug800_NoSampler_FGM         | single_view        | ./hf_models/chinese-macbert-large         | focal          | 192       | 16           | 7                | 6            | 4            | 1.5e-05 | 800                     | False         | True      |    3.26582e+08 |        3.26582e+08 |  1743.05  |            0.698413 |       0.925365 |       0.698413 |        0.92085  |                 0.69163  |        0.710451 |           0.920803 |        0.92085  |            0.98417  |           0.961753 | nan                                                                                                        |       nan    |               nan | nan                                                                                                                                                                                                                                                  |
| View3_MacBERT_BalancedFocal_Aug600_NoSampler_FGM | single_view        | ./hf_models/chinese-macbert-large         | balanced_focal | 192       | 16           | 7                | 7            | 6            | 1.5e-05 | 600                     | False         | True      |    3.26582e+08 |        3.26582e+08 |  1953.71  |            0.709717 |       0.928818 |       0.709717 |        0.930943 |                 0.677062 |        0.707081 |           0.92905  |        0.930943 |            0.986189 |           0.963081 | nan                                                                                                        |       nan    |               nan | nan                                                                                                                                                                                                                                                  |
| Ablation_SimpleAverage_AllViews                  | fusion_or_ablation | multi-view                                | fusion         | mixed     | mixed        | mixed            | mixed        | mixed        | mixed   | mixed                   | mixed         | mixed     |  nan           |      nan           |     0     |            0.699406 |       0.932005 |       0.699406 |        0.933068 |                 0.669604 |        0.685959 |           0.93103  |        0.933068 |            0.986614 |           0.96494  | [0.2, 0.2, 0.2, 0.2, 0.2]                                                                                  |         0    |                 0 | nan                                                                                                                                                                                                                                                  |
| Ours_LTMVFv2_WeightedFusion_Tau_WeakBias         | fusion_or_ablation | multi-view                                | fusion         | mixed     | mixed        | mixed            | mixed        | mixed        | mixed   | mixed                   | mixed         | mixed     |  nan           |      nan           |     0     |            0.757415 |       0.937849 |       0.757415 |        0.932537 |                 0.672223 |        0.685383 |           0.931133 |        0.932537 |            0.986507 |           0.964409 | [0.020208406766561167, 0.006777684684763986, 0.19184884946718792, 0.6355212045304286, 0.14564385455105844] |         0.15 |                 1 | ["View1_MacBERT_Focal_Aug800_NoSampler_FGM", "View2_MacBERT_FocalBCE_Aug800_Sampler_FGM", "View3_MacBERT_BalancedFocal_Aug600_NoSampler_FGM", "View4_MacBERT_Focal_Aug500_NoSampler_Len128_FGM", "View5_RoBERTa_BalancedFocal_Aug600_NoSampler_FGM"] |
| Ablation_Fusion_NoTau_NoBias                     | fusion_or_ablation | multi-view                                | fusion         | mixed     | mixed        | mixed            | mixed        | mixed        | mixed   | mixed                   | mixed         | mixed     |  nan           |      nan           |     0     |            0.747303 |       0.934661 |       0.747303 |        0.93174  |                 0.656685 |        0.671263 |           0.928563 |        0.93174  |            0.986348 |           0.964409 | [0.03345779203188016, 0.0024389950368066487, 0.11804283846268673, 0.8204188608246762, 0.02564151364395031] |         0    |                 0 | nan                                                                                                                                                                                                                                                  |
| View5_RoBERTa_BalancedFocal_Aug600_NoSampler_FGM | single_view        | ./hf_models/chinese-roberta-wwm-ext-large | balanced_focal | 192       | 16           | 7                | 5            | 3            | 1.5e-05 | 600                     | False         | True      |    3.26582e+08 |        3.26582e+08 |  1430.32  |            0.679498 |       0.927224 |       0.679498 |        0.930146 |                 0.663616 |        0.669863 |           0.929311 |        0.930146 |            0.986029 |           0.961487 | nan                                                                                                        |       nan    |               nan | nan                                                                                                                                                                                                                                                  |
| View4_MacBERT_Focal_Aug500_NoSampler_Len128_FGM  | single_view        | ./hf_models/chinese-macbert-large         | focal          | 128       | 16           | 6                | 6            | 4            | 1.5e-05 | 500                     | False         | True      |    3.26582e+08 |        3.26582e+08 |  1548.37  |            0.743245 |       0.92988  |       0.743245 |        0.925631 |                 0.649101 |        0.662646 |           0.921759 |        0.925631 |            0.985126 |           0.963612 | nan                                                                                                        |       nan    |               nan | nan                                                                                                                                                                                                                                                  |
| View2_MacBERT_FocalBCE_Aug800_Sampler_FGM        | single_view        | ./hf_models/chinese-macbert-large         | focal_bce      | 192       | 16           | 7                | 3            | 1            | 1.5e-05 | 800                     | True          | True      |    3.26582e+08 |        3.26582e+08 |   892.397 |            0.678119 |       0.908632 |       0.678119 |        0.901726 |                 0.652857 |        0.64183  |           0.902129 |        0.901726 |            0.980345 |           0.960691 | nan                                                                                                        |       nan    |               nan | nan                                                                                                                                                                                                                                                  |

## 6. 最佳模型

- 最佳模型：Ablation_Fusion_Tau_NoBias
- Macro-F1：0.7264
- Accuracy：0.9328
- Balanced Accuracy：0.6973
- Weighted-F1：0.9314
- OVR 准确率均值：0.9866
- OVR 准确率最小值：0.9649

## 7. 达标检查

- Macro-F1 > 0.8：False，当前 0.7264
- Accuracy 不低于 0.9：True，当前 0.9328
- OVR 最小准确率 > 0.9：True，当前 0.9649

## 8. 消融实验说明

本脚本自动生成以下消融配置：

- `Ablation_Fusion_Tau_NoBias`：融合 + tau，但不使用 bias；
- `Ablation_Fusion_NoTau_NoBias`：仅权重融合，无 tau，无 bias；
- `Ablation_MacBERT_Views_Only`：只使用 MacBERT 视角；
- `Ablation_SimpleAverage_AllViews`：所有视角简单平均；
- 单视角模型用于比较 Focal、Focal-BCE、Balanced-Focal、sampler 和增强强度影响。

## 9. 图表文件

- 类别分布：`outputs_ltmvf_v2/figures/class_distribution.png`
- 类别占比：`outputs_ltmvf_v2/figures/class_distribution_pie.png`
- Macro-F1 对比：`outputs_ltmvf_v2/figures/model_macro_f1_comparison.png`
- Accuracy 对比：`outputs_ltmvf_v2/figures/model_accuracy_comparison.png`
- OVR 最小准确率对比：`outputs_ltmvf_v2/figures/model_ovr_min_acc_comparison.png`
- 训练损失曲线：`outputs_ltmvf_v2/figures/training_loss_curves.png`
- 验证集 Macro-F1 曲线：`outputs_ltmvf_v2/figures/val_macro_f1_curves.png`
