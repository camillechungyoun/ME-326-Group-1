cmake_minimum_required(VERSION 3.0.2)
project(me326_locobot_example)

find_package(catkin_simple REQUIRED)
#find_package(catkin REQUIRED COMPONENTS
#   roscpp
#   rospy
#   std_msgs
#   geometry_msgs
#   message_generation
#)
catkin_python_setup()
catkin_simple()


#add_message_files(
#	FILES
#	custommsg.msg
#)

#generate_messages(
#	DEPENDENCIES
#	geometry_msgs
#	std_msgs
#)



#catkin_package(
#	CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime
#)




## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${PCL_LIBRARIES})

#C++ implementation of service
cs_add_executable(matching_ptcld_serv src/matching_ptcld_serv.cpp)

cs_install()

#cs_install_scripts(scripts/my_script.py) #optional command - https://github.com/catkin/catkin_simple 

cs_export()
