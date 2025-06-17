cmake_minimum_required(VERSION 3.10)
project(mujoco_wrappers)

# find dependencies
find_package(ament_cmake REQUIRED)

# Install targets
install(PROGRAMS
  controllers/torque_mpc_publisher.py
  controllers/torque_publisher.py
  controllers/pd_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()