cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0009)
  cmake_policy(SET CMP0009 NEW)
endif()
project(orcahand_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Instala únicamente las carpetas que realmente existen en el repositorio
foreach(dir assets v1 v2 launch rviz urdf meshes)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    install(DIRECTORY ${dir}
      DESTINATION share/${PROJECT_NAME}
    )
  endif()
endforeach()

ament_package()