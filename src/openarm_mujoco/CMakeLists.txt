cmake_minimum_required(VERSION 3.8)
project(openarm_mujoco)

find_package(ament_cmake REQUIRED)

install(DIRECTORY v1
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
