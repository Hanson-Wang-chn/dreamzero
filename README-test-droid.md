# `test_droid.py` 使用说明

`test_droid.py` 用来在 DROID 数据集上比较两种 video flow-matching 设置：

1. `standard`：保持原始 flow-matching 路径，action 先加噪，再与 video 一起参与建模。
2. `gt_action`：在相同 video 噪声、相同 timestep 下，把 action 直接替换成干净的 GT action，并把 action timestep 设为 0。

脚本只统计 **video prediction 部分** 的 loss，不统计 action loss，因此更适合回答下面这个问题：

`把 action 当作 GT 提供后，video flow matching 的表现会不会明显变化？`

## 结果含义

脚本会同时输出两组指标：

- `video_l2_raw_*`
  - 纯 video flow 预测与 target 之间的 L2 / MSE，未乘 scheduler 权重。
- `video_l2_weighted_*`
  - 更接近仓库原始训练逻辑的指标，对应 `wan_flow_matching_action_tf.py` 里的 `weighted_dynamics_loss`。

最重要的对比项通常是：

- `video_l2_weighted_standard`
- `video_l2_weighted_gt_action`
- `video_l2_weighted_gt_minus_standard`

其中：

- `gt_minus_standard < 0`：说明使用 GT action 后，video loss 更低。
- `gt_minus_standard > 0`：说明使用 GT action 后，video loss 更高。
- `gt_minus_standard` 绝对值很小：说明两种模式效果接近。

## 固定测试样本

测试样本直接写在 `test_droid.py` 文件开头：

```python
DEFAULT_SAMPLE_SPECS = [
    SampleSpec(...),
    ...
]
```

每条样本包含：

- `episode_index`
- `base_index`
- `note`

如果你想换测试数据，直接修改这里即可。

## 默认路径

脚本默认使用下面两个路径：

- checkpoint：`./checkpoints`
- 数据集：`/mnt/data/dataset/lerobot/GEAR-Dreams/DreamZero-DROID-Data`

如果你的 checkpoint 不在 `./checkpoints`，可以用 `--model_path` 覆盖。

## 单卡运行

如果你只想用第 5 张物理卡，也就是服务器后四张卡里的第一张：

```bash
CUDA_VISIBLE_DEVICES=4 python test_droid.py --device cuda:0
```

注意这里的 `cuda:0` 指的是 **可见卡里的第 0 张**，不是整机的物理 0 号卡。

如果你想只用第 7 张物理卡：

```bash
CUDA_VISIBLE_DEVICES=7 python test_droid.py --device cuda:0
```

## 使用后四张卡里的某几张

如果你只想用物理卡 `4` 和 `6`：

```bash
CUDA_VISIBLE_DEVICES=4,6 torchrun --standalone --nproc_per_node=2 test_droid.py
```

脚本会把 `DEFAULT_SAMPLE_SPECS` 里的样本按 rank 轮转分给不同进程，每张卡各算自己那部分，最后由 rank 0 汇总。

## 使用后四张卡全部运行

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 test_droid.py
```

这会：

- 在 4 个进程中各自加载一份 DreamZero-DROID 模型
- 每个进程处理一部分固定样本
- 最终在 rank 0 打印完整汇总表，并保存结果文件

这不是模型并行，而是 **多进程分样本并行评测**。它的优点是实现简单、稳定，也方便你按 GPU 数量灵活扩展或缩减。

## 常用命令

### 1. 使用默认路径，结果保存到默认目录

```bash
CUDA_VISIBLE_DEVICES=4 python test_droid.py --device cuda:0
```

### 2. 指定 checkpoint 和输出目录

```bash
CUDA_VISIBLE_DEVICES=4 python test_droid.py \
  --device cuda:0 \
  --model_path /path/to/DreamZero-DROID \
  --output_dir results/test_droid_run1
```

### 3. 使用后四张卡全部跑

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 test_droid.py \
  --model_path ./checkpoints \
  --output_dir results/test_droid_4gpu
```

### 4. 强制把所有样本都用同一个 prompt

默认情况下，脚本会读取数据集里的 language instruction。  
如果你想统一成一条 prompt，可以手动指定：

```bash
CUDA_VISIBLE_DEVICES=4 python test_droid.py \
  --device cuda:0 \
  --prompt "Move the pan forward and use the brush to clean the pan"
```

## 输出文件

默认输出目录是 `results/test_droid`，里面会生成：

- `per_sample.jsonl`
  - 每个样本一行，包含 episode、base_index、prompt、raw/weighted loss 和差值。
- `summary.json`
  - 全部样本的平均值、标准差、最小值、最大值，以及本次运行的参数。

## 终端输出示例

脚本在终端会打印一张表，主要字段如下：

- `raw/no-gt`
- `raw/gt`
- `raw diff`
- `weighted/no-gt`
- `weighted/gt`
- `weighted diff`

推荐优先看 `weighted diff` 的均值：

- 接近 `0`：说明把 action 改成 GT 后，video flow-matching 表现基本不变。
- 明显小于 `0`：说明 GT action 对 video flow-matching 有帮助。
- 明显大于 `0`：说明 GT action 反而让 video flow-matching 变差。

## 参数说明

- `--model_path`
  - DreamZero-DROID checkpoint 目录。
- `--dataset_path`
  - DROID LeRobot 数据集目录。
- `--device`
  - 单进程模式下使用的设备，默认 `cuda:0`。
- `--output_dir`
  - 结果输出目录。
- `--prompt`
  - 可选，覆盖数据集自带 instruction。
- `--seed`
  - 基础随机种子。脚本会在每个样本上用 `seed + sample_id` 保持可复现。
- `--master_port`
  - 单进程 fallback 的分布式初始化端口，一般不用改。

## 实现说明

脚本的比较方式是：

1. 从 DROID LeRobot 数据集中读出固定样本的多视角视频、state、action、language。
2. 用 checkpoint 自带的训练配置做同口径预处理。
3. 视频部分按单个 DreamZero-DROID 训练块取样，也就是每条样本使用 9 张原始帧：
   - 1 张条件帧
   - 8 张后续帧
   这样经过 VAE 后会得到 3 个 latent frames，对应 1 个 24-step action block。
4. 构造与原 flow-matching 训练一致的 video 噪声、action 噪声和 timestep。
5. 分别计算：
   - `standard`：noisy action 参与建模时的 video-only loss
   - `gt_action`：clean GT action 参与建模时的 video-only loss
6. 汇总两者差值。

因此，这个脚本比较的是 **同一批样本、同一批 video 噪声、同一组 timestep 下，两种 action 条件对 video flow-matching 误差的影响**。

# 运行结果

========================================================================================================================
Per-sample video flow-matching loss comparison
========================================================================================================================
sample   episode    base     raw/no-gt        raw/gt      raw diff    weighted/no-gt     weighted/gt   weighted diff
------------------------------------------------------------------------------------------------------------------------
     0         0      24      0.016551      0.016412     -0.000140          0.024550        0.024342       -0.000208
     1         1      24      0.048394      0.046841     -0.001553          0.022201        0.021488       -0.000712
/mnt/data/miniconda3/envs/dreamzero/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
     2         2      24      0.020289      0.020258     -0.000031          0.029346        0.029301       -0.000044
     3        15      24      0.028978      0.029215      0.000236          0.019459        0.019618        0.000159
/mnt/data/miniconda3/envs/dreamzero/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
     4        42      24      0.035117      0.034999     -0.000118          0.072821        0.072575       -0.000246
------------------------------------------------------------------------------------------------------------------------
Means: raw/no-gt=0.029866, raw/gt=0.029545, raw diff=-0.000321, weighted/no-gt=0.033675, weighted/gt=0.033465, weighted diff=-0.000210
========================================================================================================================

结论：把原本 noisy 的 action 换成 GT 后，video flow-matching 基本还能保持差不多的结果，甚至平均略有帮助。因此，GT action 不会明显破坏 DreamZero 的 video flow matching，模型对这类条件替换是比较稳的。
