cmake_minimum_required(VERSION 3.10)
project(mujoco_publisher)

# find dependencies
find_package(ament_cmake REQUIRED)

# 安装 Python 脚本
install(PROGRAMS
  script/mujoco_publisher.py
  script/mujoco_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()