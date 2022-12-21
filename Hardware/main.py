import rospy
import tf
import random
import sys
import copy
import math
import actionlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import c
from robot import VirtualRobot
from nav_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from tf.transformations import quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction , MoveBaseGoal
from nav_msgs.srv import *

FD_ALGO = "WFD"
ROBOT_NAME = "locobot3"

'''
===========================USAGE===========================
Initialize FD_ALGO to the string "WFD" if you would like 
to evaluate the WFD algorithm.

Initialize FD_ALGO to the string "Greedy" if you would like 
to evaluate the greedy frontier detection algorithm.

Initialize ROBOT_NAME to the name of the robot you wish to use.
Our implementation was designed for a LoCoBot.
===========================================================
'''

# ROSMap
# This class manages all the coordinate system
# conversions and map processing
# 
# In our code, we use real coordinates as in +x denotes right and +y denotes front (in meters)
# While working with grids, we use row, col notation 
# This is important as in ROS, +x is front and +y is left (so the coordinate system is rotated)
class ROSMap:
    def __init__(self, map_msg):
        # get the map origin, (0,0) in grid, coordinates in real coordinates 
        self.origin = (-map_msg.info.origin.position.y, map_msg.info.origin.position.x)
        self.origin_theta = map_msg.info.origin.orientation.z

        # set the map width and height
        self.w, self.h = map_msg.info.width, map_msg.info.height # the width and height of map in terms of number of cells
        self.resolution = map_msg.info.resolution # the edge size of a cell in meters 

        map_arr = np.array(map_msg.data, np.float) # 1D array describing map
        map_arr[map_arr > c.WALL_THRESH] = c.WALL
        self.grid = map_arr.reshape((self.h, self.w)) # reshape to 2D array
    
    # Convert grid coordinates to real coordinates
    def grid_to_real(self, pos):
        row, col = pos # grid coordinates
        row = -row

        # Find location (x, y) in meters given the map's resolution
        mx, my = row * self.resolution, col * self.resolution 
        ox, oy = self.origin

        # Shift by map origin to get location with respect to global frame
        return (ox + mx, oy + my)

    # Convert real coordinates to grid coordinates
    def real_to_grid(self, pos):
        rx, ry = pos # real coordinates
        ox, oy = self.origin

        # Calculate position relative to the map origin
        mx, my = rx - ox, ry - oy

        # Round because grid coordinates must be integer tuples
        row, col = int(-round(mx / self.resolution)), int(round(my / self.resolution))
        return (row, col)

    # Merge map with cost map
    # (Take the unknown points on map and add them to cost map)
    def merge(m,c_map):
        assert(m.w == c_map.w and m.h == c_map.h)
        c_map = copy.deepcopy(c_map)
        c_map.grid[m.grid == c.UNEXPLORED] = c.UNEXPLORED

        return c_map

# RobotDriver
# This class manages all the ROS interactions, map reading,
# position reading, and command sending
class RobotDriver:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        
        # Initialize and define everything required for moving the robot to a goal
        self.client = actionlib.SimpleActionClient("/"+ str(self.robot_name) + "/move_base", MoveBaseAction)
        self.timeout = 20.0 # in seconds
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = str(self.robot_name) + "/map"

        # Prepare to use server
        print("Waiting for action lib server: /" + str(self.robot_name) + "/move_base")
        self.client.wait_for_server()

        # Initialize maps
        self.map = self.costmap = self.merged_map = None # will be ROSMap

        # Create subscribers
        self.costmap_sub = rospy.Subscriber('/' + str(robot_name) + '/move_base/global_costmap/costmap', OccupancyGrid, self.parse_costmap)
        self.map_sub = rospy.Subscriber('/' + str(robot_name) + '/rtabmap/grid_map', OccupancyGrid, self.parse_map)
        self.pos_sub = rospy.Subscriber('/' + str(robot_name) + '/mobile_base/odom', Odometry, self.parse_odom)

        # Prepare to use service
        rospy.wait_for_service(str(self.robot_name) + '/move_base/make_plan')
        self.make_plan_serv = rospy.ServiceProxy(str(self.robot_name) + '/move_base/make_plan', GetPlan)

        # Define tolerance for navigation stack
        self.tolerance = 0.1

        self.pos = None # will be tuple (x,y) in real coordinates
    
    # Callback for the map subscriber
    def parse_map(self, msg):
        self.map = ROSMap(msg)
    
    # Callback for the costmap sub
    def parse_costmap(self, msg):
        self.costmap = ROSMap(msg)

    # Callback for the odometry subscriber
    def parse_odom(self, msg):
        self.pos = (-msg.pose.pose.position.y, msg.pose.pose.position.x)

    # Send goal, in real coordinates, to robot
    def send_goal(self, pos, angle):
        x, y = pos

        # Define message vals
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.goal.target_pose.pose.position.x = float(y)
        self.goal.target_pose.pose.position.y = float(-x)
        self.goal.target_pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, float(angle))
        self.goal.target_pose.pose.orientation.x = q[0]
        self.goal.target_pose.pose.orientation.y = q[1]
        self.goal.target_pose.pose.orientation.z = q[2]
        self.goal.target_pose.pose.orientation.w = q[3]
        
        # Send goal and wait for result
        self.client.send_goal(self.goal)
        rospy.sleep(1.0)
        wait = self.client.wait_for_result(rospy.Duration(self.timeout))
        
        # Indicate success or failure in sending goal
        if not wait:
            print("Did not finish before timeout.")
            return False
        else:
            print("Finished goal.")
            return True

    # Determine whether a goal is reachable (i.e., whether there is a complete and "good" path to the robot)
    def goal_is_reachable(self, goal):
        # Define request
        request = GetPlanRequest()

        # Define start vals
        request.start.header.stamp = rospy.Time.now()
        request.start.header.frame_id = str(self.robot_name) + '/map'
        rx, ry = self.get_pos()
        request.start.pose.position.x = ry
        request.start.pose.position.y = -rx

        # Define goal vals
        request.goal.header.stamp = rospy.Time.now()
        request.goal.header.frame_id = str(self.robot_name) + '/map'
        gx, gy = goal
        request.goal.pose.position.x = gy
        request.goal.pose.position.y = -gx

        # Define tolerance
        request.tolerance = self.tolerance # meters in x/y
        rospy.logdebug("%s sending request: %s", self, request)

        # Track the number of times we attempt to get a path/wait to get a path
        attempts = 0
        while True:
            if rospy.is_shutdown():
                sys.exit(1)
            try:
                response = self.make_plan_serv(request)
                rospy.logdebug("%s received response: %s", self.robot_name, response)
                response.plan.header.frame_id = str(self.robot_name) + '/map'
                # If the navigation stack gave a path
                if len(response.plan.poses) > 0:
                    goal_found = False # True if the goal (or a point very close to it) is in path
                    rob_pos_found = False # True if the robot position (or a point very close to it) is in path
                    free_goal = False # True if goal is a free space in the cost map
                    
                    # Get cost map
                    cost_map = self.get_costmap()

                    # Check if goal is free in cost map
                    grow, gcol = cost_map.real_to_grid((gx, gy))
                    rrow, rcol = cost_map.real_to_grid((rx, ry))
                    if cost_map.grid[grow][gcol] != c.WALL:
                        free_goal = True

                    # See if the goal or robot position (or points close to them) are in the path
                    for pose in response.plan.poses:
                        pose = pose.pose
                        posex, posey = -pose.position.y, pose.position.x

                        goal_dist = math.sqrt((posey - gy)**2 + (posex - gx)**2)
                        if goal_dist <= 0.05:
                            goal_found = True
                        
                        rob_pos_dist = math.sqrt((posey - gy)**2 + (posex - gx)**2)
                        if rob_pos_dist <= 0.05:
                            rob_pos_found = True

                    # A point is reachable essentially if goal and robot position are reachable and if the goal is free in the cost map
                    return free_goal and rob_pos_found and goal_found
                else: 
                    return False
            except rospy.ServiceException as e:
                print("No path plan received. Robot may be stuck, busy, or dealing with noise!")
                rospy.sleep(1)
                # Only attempt to get the path three times
                if attempts >= 2:
                    return False
                attempts += 1

    # Return the ith point on a path created by the navigation stack
    # If the path does not exist, return None
    def get_point_on_path(self, goal, i):
        # Define request
        request = GetPlanRequest()

        # Define start vals
        request.start.header.stamp = rospy.Time.now()
        request.start.header.frame_id = str(self.robot_name) + '/map'
        rx, ry = self.get_pos()
        request.start.pose.position.x = ry
        request.start.pose.position.y = -rx

        # Define goal vals
        request.goal.header.stamp = rospy.Time.now()
        request.goal.header.frame_id = str(self.robot_name) + '/map'
        gx, gy = goal
        request.goal.pose.position.x = gy
        request.goal.pose.position.y = -gx

        # Define tolerance
        request.tolerance = self.tolerance # meters in x/y
        rospy.logdebug("%s sending request: %s", self, request)

        try:
            response = self.make_plan_serv(request)
            rospy.logdebug("%s received response: %s", self.robot_name, response)
            response.plan.header.frame_id = str(self.robot_name) + '/map'

            # Return the ith (or last) position in the path after converting it to real coordinate system
            if len(response.plan.poses) > 0:
                pose = response.plan.poses[min(i, len(response.plan.poses)-1)].pose
                x, y = -pose.position.y, pose.position.x
                return (x,y)
            else:
                return None
        except rospy.ServiceException as e:
            print("No path plan received. Robot may be stuck, busy, or dealing with noise!")
            return None

    # Get the map
    # Use this instead of self.map because it will wait until the map is updated
    def get_map(self):
        num_sleep = 0
        while self.map is None:
            print("[{}] Map is not yet initialized! Going to sleep until it is...".format(num_sleep))
            rospy.sleep(1)
            num_sleep += 1
            if rospy.is_shutdown():
                sys.exit(1)
                break
        return self.map
    
    # Get the cost map
    # Use this instead of self.costmap because it will wait until the cost map is updated
    def get_costmap(self):
        num_sleep = 0
        while self.costmap is None:
            print("[{}] Cost map is not yet initialized! Going to sleep until it is...".format(num_sleep))
            rospy.sleep(1)
            num_sleep += 1
            if rospy.is_shutdown():
                sys.exit(1)
                break
        return self.costmap

    # Get the merged map
    # Use this instead of self.merged_map because it will wait until the merged map is updated
    def get_mergedmap(self):
        self.get_map()
        self.get_costmap()
        num_sleep = 0
        while self.map.w != self.costmap.w or self.map.h != self.costmap.h:
            print("[{}] Map and cost map are not yet the same size: ({} x {}) and ({} x {})! Going to sleep until they are...".format(num_sleep, \
             self.map.w, self.map.h, self.costmap.w, self.costmap.h))
            rospy.sleep(1)
            num_sleep += 1
            if rospy.is_shutdown():
                sys.exit(1)
                break
        self.merged_map = ROSMap.merge(self.map, self.costmap)
        return self.merged_map

    # Get the robot position
    # Use this instead of self.pos because it will wait until the robot position is updated
    def get_pos(self):
        num_sleep = 0
        while self.pos is None:
            print("[{}] Position is not initialized! Going to sleep...".format(num_sleep))
            rospy.sleep(1)
            num_sleep += 1
            if rospy.is_shutdown():
                sys.exit(1)
                break
        return self.pos

def main():
    rospy.init_node("driver")

    # Create robot driver
    ros_robot = RobotDriver(ROBOT_NAME)

    # Define the number of times the robot safe wanders
    num_wanders = 0

    # Begin infinite loop because breaks will need to occur at specific places
    while True:
        print("")
        
        if num_wanders > 2:
            print("Finished safe wandering, if applicable.")
            break

        robot_pos = ros_robot.get_pos()
        print("Robot position: ", robot_pos)
        m = ros_robot.get_mergedmap()

        row, col = m.real_to_grid(robot_pos) # convert pos to grid coordinates
        vr = VirtualRobot(col, row, m.grid)

        if FD_ALGO.lower() == "wfd":
            frontier_cells = vr.get_frontier_WFD() # get frontier cells from WFD algo
        elif FD_ALGO.lower() == "greedy":
            frontier_cells = vr.get_frontier_greedy() # get frontier cells from greedy algo
        else:
            raise ValueError('FD_ALGO can only be initialized to "WFD" or "Greedy"')

        # Create list of points that are reachable on path to each frontier cell
        reachable_points = []
        # Only check for reachable points if there are frontier cells
        if len(frontier_cells) > 0:
            for fc in frontier_cells:
                fcol, frow = fc
                # Initialize index of point on path to frontier cell that will be checked
                # Use large value because the robot should get close to frontier cell
                idx_of_path_pt = 300
                point = None
                # Find the farthest reachable point in the path
                while point is None:
                    if rospy.is_shutdown():
                        sys.exit(0)
                    # Break if no point in the path is reachable
                    if idx_of_path_pt == 0:
                        break
                    point = ros_robot.get_point_on_path(m.grid_to_real((frow, fcol)), idx_of_path_pt)
                    idx_of_path_pt -= 1
                
                # Add reachable point to list
                if point is not None and ros_robot.goal_is_reachable(point):
                    reachable_points.append(point)

        # Create flag for whether any of the points were successfully processed by ROS
        success = False
        # Try to send one of the reachable points until one is successful
        for pt in reachable_points:
            fcol, frow = pt
            print("Attempting to send frontier: (" + str(fcol) + ", " + str(frow) + ")")
            
            x, y = point

            # Redefine the angle so the robot does not have to spin
            # as much when getting new frontier
            angle = math.atan2(y-robot_pos[1], x-robot_pos[0])
            angle -= math.pi/2.0
            if ros_robot.send_goal((x,y), angle):
                success = True
                # Reset num wanders
                num_wanders = 0
                break
            else:
                print("ROS navigation stack could not plan path to frontier. Robot may be stuck, busy, or dealing with noise!")

        # Only safe wander when there are no reachable (or, implicitly, no frontier cells)
        # or when no reachable points were successfully processed by ROS
        robot_pos = ros_robot.get_pos()
        if len(reachable_points) == 0 or not success:
            print("No frontiers available or no intermediate points reachable.")
            randoms = range(1, 9)
            random.shuffle(randoms)
            # Test eight random points on radius
            for i, rand in enumerate(randoms):
                # If none of the random walk points were reachable and successfully 
                # processed by ROS break and end
                # Map is finished
                if i == len(randoms) - 1:
                    print("Nowhere else to wander.")
                    num_wanders = 3
                    break
                
                # Define random walk pose vals
                angle = rand * math.pi / 4.0
                step_size = 0.15 # in meters
                tx, ty = step_size * math.cos(angle) + robot_pos[0], step_size * math.sin(angle) + robot_pos[1]
                tangle = angle - math.pi / 2.0

                # Check if random walk point is reachable and successfully processed by ROS
                # print("Checking random walk point: " + "(" + str(tx) + ", " + str(ty) + ")")
                if ros_robot.goal_is_reachable((tx, ty)):
                    print("Attempting to safe wander to goal [{}]: {}".format(num_wanders, (tx, ty)))
                    if ros_robot.send_goal((tx, ty), tangle):
                        break

            num_wanders += 1
            continue

        if rospy.is_shutdown():
            sys.exit(0)  

    print("Finished map.")   
    sys.exit(0)

if __name__ == '__main__':
    main()

