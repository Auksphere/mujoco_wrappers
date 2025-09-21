# 主要内容：
gufic_publisher.py是mujoco仿真和ros2话题发布工具，misc_func.py是各种工具函数，控制律的增益以及预定要跟踪的理想轨迹（后期可以加入轨迹规划）
除此之外，在Chin,JAKA以及ur10机器人的model文件夹下各自新建了几个场景文件，
rbt_wiping_surface.xml包含机器人与一个25cm x 25cm的平台，
rbt_wiping_sphere.xml包含机器人与一个半径0.3m的球，球心距地面0.1m

# 使用方法：
直接执行gufic_publisher.py,选择jaka/chin,和四种任务regulation/line/circle/sphere中的一种。
目前sphere任务只针对JAKA调了参，Chin暂时没有调参。