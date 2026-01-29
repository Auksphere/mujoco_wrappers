# Variable Impedance Control Learning via AIRL

基于论文 "Learning Variable Impedance Control via Inverse Reinforcement Learning for Force-Related Tasks" (Zhang et al., IEEE RA-L 2021) 的实现。

## 概述

本项目使用Adversarial Inverse Reinforcement Learning (AIRL)从专家演示中学习可变阻抗控制策略，用于完成Peg-in-Hole插入任务。

### 核心组件

1. **Peg-in-Hole环境** (`envs/peg_in_hole_env.py`)
   - 基于MuJoCo的Gymnasium环境
   - 状态空间：关节位置/速度、末端位姿、力/力矩传感器读数
   - 动作空间：阻抗参数 (Kp, Kd) 用于线性和角度控制

2. **专家数据生成器** (`script/vic_airl/generate_expert_data.py`)
   - 使用手工调参的导纳控制器生成成功的插入轨迹
   - 专家参数针对peg-in-hole任务优化

3. **AIRL训练器** (`script/vic_airl/train_airl.py`)
   - 策略网络：输出阻抗参数
   - 判别器网络：学习奖励函数
   - 价值网络：估计状态价值

4. **评估脚本** (`script/vic_airl/evaluate_policy.py`)
   - 评估训练好的策略
   - 可视化轨迹、力和阻抗参数
   - 与专家演示对比


## 使用方法

### 步骤1: 生成专家演示数据

```bash
cd mujoco_wrappers
python generate_expert_data.py
```

这将生成10个专家演示轨迹并保存到 `data/expert_demonstrations.pkl`。

可以修改参数：
```python
# 在 generate_expert_data.py 的 main 函数中
dataset = generator.generate_dataset(
    n_demonstrations=20,  # 生成更多演示
    save_path="data/expert_demonstrations.pkl"
)
```

### 步骤2: 训练AIRL策略

```bash
python train_airl.py
```

训练配置：
- 默认训练1000个epoch
- 每个epoch收集10个episode
- 模型保存在 `models/airl/`
- 自动保存最佳模型和定期检查点

可调参数：
```python
trainer = AIRLTrainer(
    env=env,
    expert_data=expert_data,
    hidden_dim=256,           # 网络隐藏层维度
    lr_policy=3e-4,           # 策略学习率
    lr_discriminator=3e-4,    # 判别器学习率
    batch_size=256,           # 批大小
    gamma=0.99                # 折扣因子
)
```

### 步骤3: 评估训练好的策略

```bash
# 评估10个episode
python evaluate_policy.py --model models/airl/best_model.pt --n_episodes 10

# 带渲染的单个episode
python evaluate_policy.py --model models/airl/best_model.pt --render

# 与专家对比
python evaluate_policy.py --model models/airl/best_model.pt --expert data/expert_demonstrations.pkl
```

## 方法说明

### AIRL算法

AIRL通过以下方式学习奖励函数和策略：

1. **判别器**学习区分专家轨迹和策略轨迹：
   ```
   D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + π(a|s))
   ```
   其中 `f(s,a,s') = r(s,a) + γV(s') - V(s)` 是优势函数

2. **策略网络**学习最大化从判别器获得的奖励：
   ```
   π*(a|s) = argmax E[r(s,a)]
   ```

3. **交替训练**：
   - 更新判别器来区分专家和策略
   - 更新策略来"愚弄"判别器
   - 更新价值网络估计回报

### 阻抗参数

策略网络输出12维动作：
- `Kp_linear` (3D): x, y, z方向的线性刚度 [100-2000 N/m]
- `Kd_linear` (3D): x, y, z方向的线性阻尼 [10-200 Ns/m]
- `Kp_angular` (3D): roll, pitch, yaw的角度刚度 [50-500 Nm/rad]
- `Kd_angular` (3D): roll, pitch, yaw的角度阻尼 [5-100 Nms/rad]

网络输出归一化到[0, 1]，然后缩放到实际范围。

### 专家策略

专家使用固定的导纳参数：
```python
Mc = diag([20, 20, 20, 5, 5, 5])        # 质量
Dc = diag([15, 15, 30, 8, 8, 8])        # 阻尼（z方向更高）
Kc = diag([800, 800, 400, 200, 200, 200])  # 刚度（z方向更低）
```

这些参数针对peg-in-hole任务优化，在插入方向（z轴）提供较低的刚度和较高的阻尼。

## 项目结构

```
mujoco_wrappers/
├── envs/
│   └── peg_in_hole_env.py          # Gymnasium环境
├── models/
│   └── jaka_zu12/
│       └── jaka_pih.xml            # MuJoCo场景文件
├── script/
│   ├── vic/
│   │   ├── admittance_publisher.py # 导纳控制实现
│   │   ├── misc_func.py            # 辅助函数
│   │   └── filter.py               # 滤波器
│   └── vic_airl/
│       ├── generate_expert_data.py # 专家数据生成
│       ├── train_airl.py           # AIRL训练
│       ├── evaluate_policy.py      # 策略评估
│       └── README.md               # 本文件
├── controllers/
│   └── ik_arm.py                   # IK求解器
└── data/
    └── expert_demonstrations.pkl   # 专家演示数据
```

## 预期结果

训练成功后，策略应该能够：
1. 学习到与任务相关的阻抗参数变化
2. 在插入阶段降低z方向刚度以适应接触
3. 在x-y方向保持较高刚度以维持对准
4. 达到与专家相近的成功率和力控性能

典型性能指标：
- 成功率：> 80%
- 最大接触力：< 50N
- 最终误差：< 5mm
- 平均episode奖励：与专家相近

## 论文参考

```bibtex
@article{zhang2021learning,
  title={Learning Variable Impedance Control via Inverse Reinforcement Learning for Force-Related Tasks},
  author={Zhang, Jianfeng and Koppenhöfer, Christian and Mukhopadhyay, Rudra and Buss, Martin},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={2489--2496},
  year={2021},
  publisher={IEEE}
}
```

## 故障排除

### 问题1: IK求解失败
- 检查初始配置是否合理
- 调整IK求解器参数（`ilimit`, `tol`）
- 确保目标位姿在工作空间内

### 问题2: 训练不稳定
- 降低学习率
- 增加batch size
- 调整判别器和策略更新频率
- 检查专家数据质量

### 问题3: 策略不学习
- 确保专家演示成功且多样
- 增加训练epoch数
- 调整网络架构（增加隐藏层）
- 检查奖励信号是否合理

### 问题4: MuJoCo仿真不稳定
- 降低physics_dt（更精细的仿真）
- 调整接触参数
- 检查模型文件的物理参数

## 扩展方向

1. **在线学习**：在与环境交互时持续更新策略
2. **多任务学习**：学习适用于多种力控任务的通用阻抗策略
3. **迁移学习**：将学到的策略迁移到真实机器人
4. **探索其他IRL方法**：GAIL, SQIL等
5. **集成现有控制器**：使用学到的阻抗参数增强现有控制器

- 基于 `script/vic/admittance_publisher.py` 的导纳控制实现
- 参考 Zhang et al. (2021) 的AIRL方法
- 使用 MuJoCo 和 Gymnasium 框架