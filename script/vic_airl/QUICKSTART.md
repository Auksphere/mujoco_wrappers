# Quick Start Guide - Variable Impedance Control Learning via AIRL

快速开始使用AIRL学习可变阻抗控制策略。

## 1分钟快速测试

```bash
cd script/vic_airl

# 测试环境设置
python test_setup.py

# 如果所有测试通过，运行完整工作流
python run_workflow.py
```

## 逐步运行（推荐用于学习）

### 步骤1: 检查配置

```bash
# 查看当前配置
python config.py

# 根据需要修改 config.py 文件
```

### 步骤2: 生成专家演示

```bash
python generate_expert_data.py
```

这将生成10个专家演示轨迹（约10分钟）。生成的数据保存在：
- `data/expert_demonstrations.pkl`

### 步骤3: 训练AIRL策略

```bash
python train_airl.py
```

默认训练1000个epoch（约2-4小时，取决于硬件）。训练输出：
- `models/airl/best_model.pt` - 最佳模型
- `models/airl/checkpoint_*.pt` - 定期检查点
- `models/airl/final_model.pt` - 最终模型

### 步骤4: 评估策略

```bash
# 评估10个episode
python evaluate_policy.py --model ../../models/airl/best_model.pt --n-episodes 10

# 单个episode带可视化
python evaluate_policy.py --model ../../models/airl/best_model.pt --render

# 与专家对比
python evaluate_policy.py --model ../../models/airl/best_model.pt --expert ../../data/expert_demonstrations.pkl
```

## 常见问题

### Q1: 如何调整训练参数？

编辑 `config.py` 文件：

```python
AIRL_CONFIG = {
    'n_epochs': 2000,        # 训练更多epoch
    'hidden_dim': 512,       # 使用更大的网络
    'lr_policy': 1e-4,       # 降低学习率
    # ...
}
```

### Q2: 如何生成更多/更少的专家演示？

```python
# 在 generate_expert_data.py 中修改
dataset = generator.generate_dataset(
    n_demonstrations=20,  # 改为20个演示
    save_path="data/expert_demonstrations.pkl"
)
```

或使用命令行（需要在代码中添加argparse支持）：

```bash
python generate_expert_data.py --n-demos 20
```

### Q3: 如何使用已有的检查点继续训练？

在 `train_airl.py` 中添加：

```python
# 加载检查点
trainer.load('models/airl/checkpoint_500.pt')

# 继续训练
trainer.train(n_epochs=500, ...)  # 再训练500个epoch
```

### Q4: 训练很慢怎么办？

1. 使用GPU加速：
```python
# 在 config.py 中设置
HARDWARE_CONFIG = {
    'device': 'cuda',  # 使用GPU
}
```

2. 减少episode数量：
```python
AIRL_CONFIG = {
    'n_episodes_per_epoch': 5,  # 从10减到5
}
```

3. 减少batch size：
```python
AIRL_CONFIG = {
    'batch_size': 128,  # 从256减到128
}
```

### Q5: 如何可视化训练过程？

可以使用TensorBoard（需要先安装）：

```bash
pip install tensorboard

# 在 config.py 中启用
LOGGING_CONFIG = {
    'tensorboard': True,
}

# 运行训练后查看
tensorboard --logdir=logs
```

## 文件结构说明

```
script/vic_airl/
├── config.py                   # 配置文件
├── generate_expert_data.py     # 专家数据生成
├── train_airl.py               # AIRL训练
├── evaluate_policy.py          # 策略评估
├── test_setup.py               # 环境测试
├── run_workflow.py             # 完整工作流
├── README.md                   # 详细文档
├── QUICKSTART.md               # 本文件
└── requirements.txt            # 依赖列表
```

## 预期输出

### 专家数据生成

```
Generating 10 expert demonstrations...

--- Demonstration 1/10 ---
Generating expert demonstration for 10.0s (250 steps)...
100%|████████████| 250/250 [00:15<00:00, 16.23it/s]
Max force: 42.34N, Final error: 3.21mm

...

Dataset saved to data/expert_demonstrations.pkl
```

### 训练输出

```
Loaded 2500 expert transitions
Training AIRL:   0%|          | 0/1000 [00:00<?, ?it/s]

Epoch 0
  Discriminator Loss: 0.6932
  Policy Loss: -1.2345
  Value Loss: 0.4567
  Eval Reward: -45.67
  Saved best model (reward: -45.67)

...
```

### 评估输出

```
Episode 1/10: Success=True, Reward=-12.34, MaxForce=38.21N, FinalError=2.45mm
...

--- Evaluation Summary ---
Success Rate: 80.0%
Avg Reward: -15.23
Avg Max Force: 41.23N
Avg Final Error: 3.12mm
```

## 下一步

1. **调参优化**：根据评估结果调整配置参数
2. **可视化分析**：查看生成的轨迹图 `plots/airl_trajectory.png`
3. **真实机器人部署**：将学到的策略迁移到真实机器人
4. **扩展任务**：尝试其他力控任务

## 获取帮助

- 查看详细文档：`README.md`
- 查看配置选项：`python config.py`
- 运行测试：`python test_setup.py`
- 查看论文：Zhang et al., "Learning Variable Impedance Control via Inverse Reinforcement Learning", IEEE RA-L 2021

## 问题反馈

如果遇到问题，请检查：
1. 所有依赖是否正确安装（`pip install -r requirements.txt`）
2. MuJoCo模型文件路径是否正确
3. Python版本是否 >= 3.8
4. 是否有足够的磁盘空间保存数据和模型

祝实验顺利！🚀
