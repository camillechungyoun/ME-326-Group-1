# ME 326 Group 1 Collaborative Robotics Project at Stanford University
## Guide to Terminal Commands for Perception Code
Here is a guide with the terminal commands to use the files in the perception branch:

1. Run the Gazebo environment with the locobot.

$ roscd me326_locobot_example/launch/

$ ./launch_locobot_gazebo_moveit.sh

2. Run the main perception code.

$ rosrun me326_locobot_example matching_ptcld_serv

3. To see all of the cube locations being published to the topic "cube_3d_locations", run the following line.

$ rostopic echo cube_3d_locations

4. To see one particular coordinate of the blocks of a specific color (for example, the x coordinates of the blue blocks), run the following line.
   This can be replicated for the other colors of blocks (red, green, yellow) and for the other coordinates (y, z).

$ rostopic echo cube_3d_locations/blue_x


