import math
from bresenham import bresenham
from numpy import size, sort
from astar import astar
from constants import c
import random
import numpy as np

class Robot: 
    def __init__(self, x, y, grid, global_seen, frontier_algo):
        self.pos = (x, y)
        self.radius = 4
        self.side = 2 * self.radius + 1
        self.grid = grid
        self.seen = global_seen
        self.path = set()
        self.prev_path = set()
        self.frontier = None
        self.algo = frontier_algo
 
    def greedy_frontier(self):
        Q = [(self.pos, None)]
        V = set()

        while len(Q) != 0:
            n = Q.pop(0)

            n, prev = n

            if n in V: continue
            else: V.add(n)

            x, y = n

            try:
                space = self.grid[y][x] 
            except:
                continue
            
            if space != c.WALL:

                if n not in self.seen:
                    return n

                Q.append(((x - 1, y), n))
                Q.append(((x + 1, y), n))
                Q.append(((x, y - 1), n))
                Q.append(((x, y + 1), n))

                Q.append(((x - 1, y - 1), n))
                Q.append(((x + 1, y - 1), n))
                Q.append(((x - 1, y + 1), n))
                Q.append(((x + 1, y + 1), n))

            else:
                if n not in self.seen:
                    return prev

        return None
    
    def random_walk(self):

        while True:
            # Get a random direction
            delta_x, delta_y = random.choice([-1, 1]), random.choice([-1, 1])

            # Make sure this random direction is not staying in place
            while (delta_x == 0 and delta_y == 0):
                delta_x, delta_y = random.randint(-1, 1), random.randint(-1, 1)

            x, y = self.pos

            total_x = x + delta_x 
            total_y = y + delta_y

            try:
                if self.grid[total_y][total_x] != c.WALL:
                    break
            except:
                continue

        self.update_seen(total_x, total_y)  

        if len(self.seen) == self.grid.size:
            return None
        else:
            print()
            return (total_x, total_y)

# is_frontier
# arguments: p --> tuple: (x,y) 
# returns True if p is a frontier point
# p is a frontier point if it neighbors both FREE spaces and UNKNOWN points
# but is not an obstacle
    def is_frontier(self, p):
        found_free = False
        found_unknown = False
        found_obstacle = False
        col, row = p
        try: 
            found_obstacle = self.grid[row][col] == c.WALL
        except:
            pass
        for i in range(3):
            for j in range(3):
                try:
                    if self.grid[row-1 + i][col-1+j] == c.FREE:
                        found_free = True
                    if not (col-1+j, row-1 + i) in self.seen:
                        found_unknown = True
                except:
                    continue
        return found_free and found_unknown and not found_obstacle

# get_neighbors
# argument is a point in (col, row) form 
# returns the neighbors of p that are FREE (not walls or unknown)
#
    def get_neighbors(self, p):
        col, row = p
        neighbors = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                try:
                    if row - 1+ i < 0 or col -1 + j < 0:
                        continue
                    if self.grid[row-1+i][col -1 +j] != c.FREE:
                        continue
                    if (col -1 + j, row - 1 +i) not in self.seen:
                        continue
                    self.grid[row-1+i][col-1+j] # means is in grid
                    neighbors.append((col-1+j, row-1+i))
                except:
                    continue
        return neighbors

# wavefront_frontier
# no arguments
# find all the frontiers using the WFD algorithm
# starting the robot position (self.pos)
    def wavefront_frontier(self):
        Q_map = [self.pos]
        self.grid[self.pos[1]][self.pos[0]] = c.FREE

        visited = np.zeros_like(np.array(self.grid))

        self.all_frontiers = []
        num_iter = 0
        grid_size = visited.shape[0] * visited.shape[1]
        while len(Q_map) > 0:
            p = Q_map.pop(0)
            assert(self.grid[p[1]][p[0]] == c.FREE and p in self.seen)
            # print("{}/{}".format(num_iter, grid_size))
            # if num_iter % 5000 == 0:
            #     print(len(np.where(visited > 0.5, visited, visited)))
            num_iter += 1
            pcol, prow = p
            if visited[prow][pcol] > 0.5:
                continue
            visited[prow][pcol] = 1.0
            if self.is_frontier(p):
                # run inner BFS
                Q_frontier = [p]
                new_frontier = []
                # mark not visited so that it actually visits
                visited[prow][pcol] = 0.0

                while len(Q_frontier) > 0:
                    q = Q_frontier.pop(0)
                    qcol, qrow = q
                    if visited[qrow][qcol] > 0.5:
                        continue
                    visited[qrow][qcol] = 1.0
                    
                    new_frontier.append(q)
                    neighbors = self.get_neighbors(q)
                    for n in neighbors:
                        ncol, nrow = n
                        if self.is_frontier(n) and visited[nrow][ncol] < 0.5 and n in self.seen:
                            Q_frontier.append(n)
                # finished inner bfs
                if len(new_frontier) > 0:
                    # print("frontier: {}".format(new_frontier))
                    self.all_frontiers.append(new_frontier)

            for n in self.get_neighbors(p):
                ncol, nrow = n
                if visited[nrow][ncol] < 0.5 and n in self.seen:
                    Q_map.append(n)
    
        def get_centroid(frontier):
            n = float(len(frontier))
            x_acc, y_acc = 0.0, 0.0
            for p in frontier:
                x_acc += float(p[0])
                y_acc += float(p[1])
            return (int(round(x_acc/n)), int(round(y_acc/n)))
        def get_point(frontier):
            # sort by the distance to the centroid and return the first element (closest)
            c = get_centroid(frontier)
            frontier = sorted(frontier, key = lambda x: self.distance(x, c))
            return frontier[0]

        return list(map(get_point, self.all_frontiers))

# get_frontier
# find a frontier using the chosen algorithm for the simulation
# and set self.frontier and self.path
#
    def get_frontier(self):
        self.prev_path = self.path
        if self.algo is c.RANDOM_WALK:
            self.frontier = self.random_walk()
            self.path.update([self.pos, self.frontier])
        elif self.algo is c.GREEDY:
            self.frontier = self.greedy_frontier()
            self.path = self.create_path(self.frontier)
        elif self.algo is c.WAVEFRONT:
            frontiers = self.wavefront_frontier()
            frontiers = sorted(frontiers, key=lambda x: self.distance(x, self.pos))
            if len(frontiers) > 0:
                self.frontier = frontiers[0]
            else:
                self.frontier = None
            self.path = self.create_path(self.frontier)

        return self.frontier

    def distance(self, point1, point2):
        # Return Euclidean distance between two points
        return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) ** (1 / 2)


    # sort the path to frontier by length
    def sort_path(self, points, start):
        # Create new list for sorted path
        sorted_path = [start]
        if start in points:
            points.remove(start)
        # Sort points based on shortest distance between one point and another, appending as needed
        while points:
            nearest_point = min(points, key=lambda x: self.distance(sorted_path[-1], x))
            sorted_path.append(nearest_point)
            points.remove(nearest_point)
        # Return list for sorted path
        return sorted_path

    # maka a path to frontier
    def create_path(self, goal):
        print(goal)
        if goal:
            goal = (goal[1], goal[0])
            start = (self.pos[1], self.pos[0])
            path = astar(self.grid, start, goal)
            for i, pt in enumerate(path):
                path[i] = (pt[1], pt[0])
            
            path = set(path)
        else:
            path = set()

        return path

    # move towards frontier
    def explore(self):
        if self.frontier:
            # self.pos = self.frontier # teleport robot
            try:
                sorted_path = self.sort_path(self.path, self.pos)
                self.pos = sorted_path[1]
            except:
                self.pos = self.frontier
            
            return self.frontier
        else:
            print("Map finished")
            return None
    
    # Get the points of the sensing perimeter
    def raycast(self):
        x, y = self.pos

        points = []

        dist = self.radius

        num_points = self.side * 4 - 4

        base_theta = 2 * math.pi / num_points

        for i in range(num_points):
            
            theta = i * base_theta

            dx = round(dist * math.cos(theta))
            dy = round(dist * math.sin(theta))

            pt = (x + dx, y + dy)

            points.append(pt)
        
        return points
    
    # get all points on the way to the sensing perimeter that aren't blocked
    # by an obstacle and add them to seen
    def bresenham(self, p2):
        points = []

        x1, y1 = self.pos
        x2, y2 = p2

        intermediates = list(bresenham(x1, y1, x2, y2))
        for p in intermediates:
            x, y = p
            
            self.update_seen(x, y)

            points.append((x, y))
        return points

    # send rays out in all directions and add them to seen
    def sense(self):
        points = self.raycast()

        for point in points:
            self.bresenham(point)
    
    # add a grid location to the known area
    def update_seen(self, x, y):
        try:
            space = self.grid[y][x]
        except:
            return

        if space != c.WALL:
            self.seen[(x,y)] = c.FREE
        else:
            self.seen[(x,y)] = c.WALL
            