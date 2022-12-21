from email.quoprimime import unquote
import enum
from importlib.machinery import all_suffixes
from constants import c
from robot import Robot
from graphics import GraphicsWindow
import numpy as np
import copy 
import time
import math

class Env:
    def __init__(self, filename, frontier_algo, show_graphics):
        # list of robots
        self.robots = []
        # initial grid: g_0
        self.grid = np.array([[]])
        # dynamic grid: g(t)
        self.dynamic_grid = np.array([[]])
        # grid for printing
        self.viz_grid = None
        # frontier algorithms for different experiments
        self.frontier_algo = frontier_algo
        # boolean for showing graphics
        self.show_graphics = show_graphics
        # initialize the grid
        self.read_Grid(filename)
    
    # funtion to update grid at each timestep: g(t+1)
    def update_grid(self): 

        frontier = None

        for i, rob in enumerate(self.robots):

            stationary = True

            if stationary:
                rob.sense()
                rob.get_frontier()

            for y in range(len(self.grid)):
                for x in range(len(self.grid[y])):
                    pt = (x, y)
                    if pt == rob.pos:
                        self.viz_grid[y][x] = "R"
                        self.dynamic_grid[y][x] = c.ROBOT
                    elif pt == rob.frontier:
                        self.viz_grid[y][x] = "F"
                        self.dynamic_grid[y][x] = c.FREE
                    elif pt in rob.path and self.dynamic_grid[y][x] != c.ROBOT:
                        self.viz_grid[y][x] = "."
                        self.dynamic_grid[y][x] = c.NEXT_PATH
                    elif pt in rob.prev_path:
                        self.viz_grid[y][x] = "*"
                        self.dynamic_grid[y][x] = c.PREV_PATH
                    elif pt in rob.seen:
                        if rob.seen[pt] == c.FREE:
                            if self.viz_grid[y][x] == "-" or i == 0:
                                self.viz_grid[y][x] = " "
                                self.dynamic_grid[y][x] = c.FREE
                        elif rob.seen[pt] == c.WALL:
                            self.viz_grid[y][x] = "W"
                            self.dynamic_grid[y][x] = c.WALL
                    elif i == 0:
                        self.viz_grid[y][x] = "-"
                        self.dynamic_grid[y][x] = c.UNEXPLORED

            f = rob.explore()
            if frontier == None:
                frontier = f
                 
            self.print_grid()
        return frontier
    

    def update_grid_task_allocation(self): 

        frontier = None

        unique_frontiers = []
        all_utilities = []

        for i, rob in enumerate(self.robots):

            rob.sense() 
            rob.get_frontier()
            frontiers = rob.get_all_frontiers()
            def share_common(f1, f2):
                for p in f1:
                    if p in f2:
                        return True
                return False

            def contained(uniques, frontier):
                for f in uniques:
                    if share_common(f, frontier):
                        return True
                return False
            
            for f in frontiers:
                if len(unique_frontiers) == 0  or not contained(unique_frontiers, f):
                    unique_frontiers.append(f)

        def dist(x,y): # manhattan distance
            return abs(x[0]-y[0]) + abs(x[1]-y[1])

        def get_min_dist(frontier, pos):
            return min(map(lambda p: dist(pos, p), frontier))
        
        for i, rob in enumerate(self.robots):
            costs = np.array(list(map(lambda f: get_min_dist(f, rob.pos), unique_frontiers))) # cost of a frontier is its distance to this robot
            benefits = np.array(list(map(lambda f: len(f), unique_frontiers))) # benefit of a frontier is its size
            utilities = costs - benefits
            all_utilities.append(utilities)
        
        all_utilities = np.array(all_utilities)
        all_utilities = - all_utilities

        
        # now do task allocation
        rows, cols = linear_sum_assignment(all_utilities)
        
        for row,col in zip(rows, cols):
            robot = self.robots[row]
            frontier = unique_frontiers[col]

            robot.frontier = frontier[0]
            robot.path = robot.create_path(robot.frontier)
        
        for i, rob in enumerate(self.robots):
            for y in range(len(self.grid)):
                for x in range(len(self.grid[y])):
                    pt = (x, y)
                    if pt == rob.pos:
                        self.viz_grid[y][x] = "R"
                        self.dynamic_grid[y][x] = c.ROBOT
                    elif pt == rob.frontier:
                        self.viz_grid[y][x] = "F"
                        self.dynamic_grid[y][x] = c.FREE
                    elif pt in rob.path and self.dynamic_grid[y][x] != c.ROBOT:
                        self.viz_grid[y][x] = "."
                        self.dynamic_grid[y][x] = c.NEXT_PATH
                    elif pt in rob.prev_path:
                        self.viz_grid[y][x] = "*"
                        self.dynamic_grid[y][x] = c.PREV_PATH
                    elif pt in rob.seen:
                        if rob.seen[pt] == c.FREE:
                            if self.viz_grid[y][x] == "-" or i == 0:
                                self.viz_grid[y][x] = " "
                                self.dynamic_grid[y][x] = c.FREE
                        elif rob.seen[pt] == c.WALL:
                            self.viz_grid[y][x] = "W"
                            self.dynamic_grid[y][x] = c.WALL
                    elif i == 0:
                        self.viz_grid[y][x] = "-"
                        self.dynamic_grid[y][x] = c.UNEXPLORED

            f = rob.explore()
            if frontier == None:
                frontier = f
                
            self.print_grid()
        return frontier
        
    # function to read initial grid from the layout files
    def read_Grid(self, filename):
        # empty grid & empty robot list
        grid = []
        robot_pos = []

        # read through filename.txt
        with open(filename) as f:
            int_grid = np.genfromtxt(filename, dtype=int)
            int_grid[int_grid >= c.WALL_THRESH] = c.WALL
            int_grid[int_grid == c.ROBOT_INT] = c.ROBOT

            robot_pos = np.where(int_grid == c.ROBOT)
            robot_pos = list(zip(robot_pos[1], robot_pos[0]))

            self.grid = int_grid
            self.dynamic_grid = copy.deepcopy(int_grid)
            self.viz_grid = [["-" for _ in row] for row in int_grid] 

            # instantate robot list
            global_seen = dict()
            for x, y in robot_pos:
                x, y = int(x), int(y)
                # print("ROBOT X, Y", x, y)
                self.robots.append(Robot(x, y, int_grid, global_seen, self.frontier_algo))

    # function to print the current grid g(t)
    def print_grid(self): 
        for row in self.viz_grid:
            for col in row:
                print(col, end='')
            print()
        print()

    # function to run simulations & display grid graphics
    def run_simulation(self):
        timestep = 0
        while True:
            frontier = self.update_grid()
            # print(g.dynamic_grid) 
            if self.show_graphics: 
                graphics = GraphicsWindow(self.dynamic_grid)
                algo = ''
                if self.frontier_algo == c.WAYFRONT:
                    algo = 'WFD Algorithm'
                elif self.frontier_algo == c.GREEDY:
                    algo = 'Greedy Algorithm'
                else:
                    algo = 'Random Walk Algorithm'
                graphics.show_env(timestep, algo)
            timestep += 1

            if frontier == None:
                break
        return timestep