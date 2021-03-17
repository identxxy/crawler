import roslaunch

cli_args = ['/home/mosaic/catkin_ws/src/robot/launch/id.launch','vel:=2.19']
roslaunch_args = cli_args[1:]
roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

parent.start()
