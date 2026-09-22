# 主干一：基础验证与CNN基线（原步骤1–4）

## 目标与环境

先验证Fisher数学、VOC数据与评估，再交付可复用CNN。环境沿用Windows、Conda `dinov3_pspnet`、PyTorch 2.11.0+cu128、RTX4060 Laptop约8GB；VOC2012 train=5717、val=5823，20类图片级多标签分类。

## 核心步骤与结果

| 步骤 | 实现目标 | 验证/训练结论 |
|---|---|---|
| 1：NumPy参考 | 传统FV与简化Fisher编码 | 手算及前向检查通过；D=256、K=32，输出16384维 |
| 2：Fisher Layer | 可训练参数、变长mask、反向传播 | 6项测试通过，NumPy误差约2.22e-16；只做模块检查及一次合成更新 |
| 3：VOC数据/指标 | raw_label=-1/0/1，忽略0标签；AP/mAP | 5项测试通过，数据抽检、Windows加载和masked BCE正常 |
| 4：整图AlexNet | ImageNet权重→整图微调→256维特征→20类头 | 选定224基线，完整val mAP **73.39%**；导出与设备一致性通过 |

步骤1–3未进行真实图片分类训练。步骤4经历小样本拟合、续训和224/320对照；不再在主线重复逐轮日志。

## 固定交付

| 模型包 | 输入 | 完整val mAP | bottle / sofa / pottedplant AP |
|---|---|---:|---|
| `artifacts/alexnet_baseline_v1`（默认） | 224×224 | 73.39% | 43.98% / 48.48% / 49.83% |
| `artifacts/alexnet_candidate_320_v1` | 320×320 | 75.00% | 45.93% / 52.72% / 52.36% |

320候选保留为可选项；瓶子提升1.96个百分点未达到事先设定的2个百分点门槛，因此默认包继续使用224。该门槛不是统计显著性标准。

`models.baseline_bundle.load_baseline`同时加载模型和匹配预处理；整图特征为`[B,256]`，logits为`[B,20]`。val已用于选模，成绩不是独立test结果。

## 入口与证据

- 基础验证：`step01_fisher`、`step02_verify.py`、`step03_verify.py`。
- 基线训练：`step04_train.py`、`step04_continue.py`、`step04_targeted.py`；诊断/交付：`step04_diagnose.py`、`step04_export.py`、`step04_infer.py`。
- 原始证据：`runs/step02_verification.json`、`runs/step03_verification.json`、`runs/step04_continue_20260919_154811/`及模型包内`baseline.json`、`verification.json`。

```powershell
conda activate dinov3_pspnet
python step01_fisher
python step02_verify.py --require-cuda
python step03_verify.py --workers 2 --samples 32
```

## 边界与后续

整图输入为正方形缩放，主要弱类是瓶子、沙发、盆栽；不根据本阶段结果判断完整FisherNet效果。后续已转入[局部特征与GMM初始化](02_features_and_initialization.md)。
