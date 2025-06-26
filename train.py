import mujoco
import mujoco.viewer

# 加载模型
model = mujoco.MjModel.from_xml_path('./models/ur10/scene.xml')
data = mujoco.MjData(model)

# 可视化
mujoco.viewer.launch_passive(model, data)

# 输出所有关节的运动范围
print(model.jnt_range)