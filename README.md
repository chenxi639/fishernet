# Deep FisherNet 复现

这是对论文 **Deep FisherNet for Object Classification** 的阶段性工程复现。项目已经贯通 VOC2012 多标签数据、AlexNet 基线、共享 CNN + SPP 局部描述子、32 成分 Fisher 编码、端到端联合训练和多尺度评估。

## 当前结果

| 阶段 | 协议 | 完整 VOC2012 val mAP |
|---|---|---:|
| 整图 AlexNet 基线 | 224 输入 | 73.39% |
| FisherNet 第 6 轮 | 480/576/688 三尺度，原分类头 | 81.4742% |
| FisherNet 第 7 轮 | 480/576/688 三尺度，原分类头 | **81.4932%** |

第 7 轮相对第 6 轮仅提高 0.019 个百分点；按图像配对重采样得到的 95% 区间为 `[-0.0403, +0.1041]` 个百分点，因此尚不能认定为稳定提升。当前结果使用同一验证集选模和比较，也不能直接视为论文表格的严格复现。

## 模型主线

```text
image
  -> shared AlexNet convolutional features
  -> dense regions + 6x6 SPP
  -> 256-D local descriptors
  -> 32-component Fisher layer (2 x 32 x 256 = 16384)
  -> signed power + L2 normalization
  -> 20-class multi-label head
```

GMM 只负责初始化 Fisher 参数；联合训练时分类头、Fisher 参数、全连接层和 CNN 共同更新。

## 仓库结构

```text
data/        VOC 标签、预处理、密集区域和变长 batch
metrics/     VOC AP / mAP
models/      AlexNet、SPP、Fisher layer 和分类器
report/      三份主干实验报告及其关键图表
docs/        汇总后的实验分析文档
step*.py     验证、训练、评估和诊断入口
```

数据集、预训练权重、训练 checkpoint、缓存和逐次运行日志不进入仓库。脚本默认在项目根目录下读取 `datasets/`、`weights/`、`artifacts/` 和 `runs/`。

## 环境

实测环境：Windows、Python 3.11.16、PyTorch 2.11.0 + CUDA 12.8、RTX 4060 Laptop 8 GB。先按本机 CUDA 版本安装 PyTorch，再安装其余依赖：

```powershell
conda create -n fishernet python=3.11 -y
conda activate fishernet
# 从 https://pytorch.org/get-started/locally/ 选择与本机 CUDA 匹配的安装命令
pip install -r requirements.txt
```

## 数据与外部产物

VOC2012 目录应放在：

```text
datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/
  JPEGImages/
  ImageSets/Main/
```

复现实验还需要：

- `weights/alexnet-owt-7be5be79.pth`：AlexNet 预训练参数；
- `artifacts/alexnet_baseline_v1/`：第四步导出的基线包；
- `runs/step08_gmm_*/gmm.npz`：第八步生成的 GMM 初始化；
- 后续评估脚本所指定的 checkpoint 或预测缓存。

其中后两项均可由仓库脚本重新生成。部分后期诊断脚本保留了本次实验运行目录名，用于复核既有结果；迁移到新实验时应通过参数或代码中的路径入口指向新产物。

## 建议执行顺序

```powershell
# 纯数学与可微 Fisher 验证
python step02_verify.py

# 数据标签和 AP/mAP 验证（需要 VOC2012）
python step03_verify.py

# 训练整图基线
python step04_train.py --mode baseline --epochs 2

# 依次验证 SPP、密集局部描述子和采样逻辑
python step05_verify.py
python step06_verify.py
python step07_verify.py

# 采样训练特征并拟合 GMM
python step07_sample.py
python step08_gmm.py --source <step07_run_directory>

# 联合训练前验证与短程训练
python step09_verify.py
python step09_pilot.py --train-images 256 --val-images 256 --epochs 2
```

全量训练、续训、多尺度评估和当前结果所用的具体命令见[联合训练主报告](report/03_joint_training.md)。

## 文档

1. [基础验证与 CNN 基线](report/01_foundations_and_backbone.md)
2. [局部特征与 GMM 初始化](report/02_features_and_initialization.md)
3. [联合训练、诊断与下一步](report/03_joint_training.md)
4. [完整实验分析报告](docs/Deep_FisherNet_复现实验分析报告.docx)

过程性讨论、旧方案选择记录、完整错例联系表和逐次运行日志未纳入公开仓库。关键结论均保留了指标、协议、限制和对应的聚合图表。

## 当前限制与下一步

- 论文配置、预训练来源和训练日程尚未完全对齐，不能把当前 mAP 与论文结果直接作单因素比较。
- 第 6、7 轮差异低于当前统计不确定性，下一轮应从共同 checkpoint 做单变量对照。
- `bottle`、`pottedplant`、`sofa` 的 train–val AP 差距约 29–31 点，优先验证分类头和 CNN 分组 weight decay、数据增强及类别采样。
- 需要独立保留最终测试协议，避免继续复用验证集造成选择偏差。
