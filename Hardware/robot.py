import math
import random
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bresenham import bresenham
from astar import astar
from constants import c

FRONTIER_SIZE = 10 # Min side length // 0.9 of frontier in greedy algo
MIN_FRONTIER_DISTANCE = 5 # Minimum distance between two frontiers in greedy algo
MAX_FRONTIER_LEN = 20 # Maximum number of frontiers to be returned by greedy algo

# VirtualRobot
# This class manages all the frontier detection
# and virtual robot interactions with its map
class VirtualRobot: 
    def __init__(self, x, y, grid):
        self.pos = (x, y)
        self.grid = grid
        self.seen = dict() # seen positions on map

        # Initialize seen using known/explored values
        known = np.where(grid != c.UNEXPLORED)
        rows, cols = known
        known = list(zip(cols, rows))
        for col, row in known:     
            self.seen[(col,row)] = grid[row][col]
    
    # Determine whether a frontier is big enough
    # (whether 90% of the FRONTIER_SIZE x FRONTIER_SIZE square
    # of which it is the center is also unexplored)
    def is_big(self, pt):

        x, y = pt

        size = FRONTIER_SIZE

        half = size // 2

        new_points = total = 0.0

        for i in range(x - half, x + half):
            for j in range(y - half, y + half):
                try:
                    space = self.grid[j][i]
                except:
                    continue
            
                if (i, j) not in self.seen:
                    new_points += 1.0
                total += 1.0
        
        return new_points >= total * 0.9

    # Determine whether a point is a frontier
    # using definition in paper (if a point is a neighbor to an explored free point and unexplored point)
    def is_frontier(self, p):
        found_free = False
        found_unknown = False
        found_obstacle = False
        col, row = p
        for i in range(3):
            for j in range(3):
                try:
                    if self.grid[row - 1 + i][col - 1 + j] == c.FREE:
                        found_free = True
                    if self.grid[row - 1 + i][col - 1 + j] == c.UNEXPLORED:
                        found_unknown = True
                    if self.grid[row - 1 + i][col - 1 + j] == c.WALL:
                        found_obstacle = True
                except:
                    continue
        return found_free and found_unknown and not found_obstacle

    # Get free neighbors of a point
    def get_neighbors(self, p):
        col, row = p
        neighbors = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                try:
                    if row - 1 + i < 0 or col - 1 + j < 0:
                        continue
                    if self.grid[row - 1 + i][col - 1 + j] != c.FREE:
                        continue
                    neighbors.append((col - 1 + j, row - 1 + i))
                except:
                    continue
        return neighbors

    # Greedy frontier detection algorithm
    # Return list (max size = MAX_FRONTIER_LEN) of 
    # nearest unexplored points that are sufficiently 
    # large and at least MIN_FRONTIER_DISTANCE apart
    def get_frontier_greedy(self):
        Q = [self.pos]
        V = set()

        found_frontiers = []

        # Execute BFS
        while len(Q) != 0 and len(found_frontiers) < MAX_FRONTIER_LEN:
            
            n = Q.pop(0)
            
            # If we already visited this loc, skip
            if n in V: continue
            else: V.add(n)

            x, y = n

            try:
                # Make sure the point is in bounds
                space = self.grid[y][x] 
            except:
                continue
            
            if space <= c.FREE:

                # If n is not in self.seen, it is unknown
                # If it is in a sufficiently large unknown space, and far enough from
                # already found frontiers, add it to the frontier list
                if n not in self.seen and self.is_big(n):
                    min_dist = 1e9
                    
                    for f in found_frontiers:
                        if self.distance(f, n) < min_dist:
                            min_dist = self.distance(f,n)

                    if min_dist > MIN_FRONTIER_DISTANCE:
                        found_frontiers.append(n)

                # Add neighbors to the end of the queue
                if x > 0:
                    Q.append((x - 1, y))
                Q.append((x + 1, y))
                
                if y > 0:
                    Q.append((x, y - 1))
                Q.append((x, y + 1))

            else:
                continue
  
        return found_frontiers

    # WFD algorithm
    def get_frontier_WFD(self):
        # Run outer BFS
        Q_map = [self.pos]
        visited = np.zeros_like(np.array(self.grid))
        all_frontiers = [] # will contain lists of tuples that comprise frontiers
        num_iter = 0
        grid_size = visited.shape[0] * visited.shape[1]
        while len(Q_map) > 0:
            p = Q_map.pop(0)
            num_iter += 1
            pcol, prow = p
            if visited[prow][pcol] > 0.5:
                continue
            visited[prow][pcol] = 1.0
            if self.is_frontier(p):
                # Run inner BFS
                Q_frontier = [p]
                new_frontier = []
                # Mark point as not visited so it will be visited later
                visited[prow][pcol] = 0.0

                # Once we find a frontier point, run BFS
                # on all neighboring points to find the whole frontier
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
                        if self.is_frontier(n) and visited[nrow][ncol] < 0.5:
                            Q_frontier.append(n)
                # Append found frontier to list
                if len(new_frontier) > 0:
                    all_frontiers.append(new_frontier)

            for n in self.get_neighbors(p):
                ncol, nrow = n
                if visited[nrow][ncol] < 0.5:
                    Q_map.append(n)
        for f in all_frontiers:
            for p in f:
                pcol, prow = p
                self.grid[prow][pcol] = 5000
        
        # Get centroid of frontier, which is defined by a list of tuples
        def get_centroid(frontier):
            n = float(len(frontier))
            x_acc, y_acc = 0.0, 0.0
            for p in frontier:
                x_acc += float(p[0])
                y_acc += float(p[1])
            return (int(round(x_acc/n)), int(round(y_acc/n)))

        # Get point nearest to centroid of a given frontier
        # Necessary for irregularly shaped frontiers
        def get_point(frontier):
            c = get_centroid(frontier)
            frontier = sorted(frontier, key=lambda x: self.distance(x, c))
            return frontier[0] # return closest point to the centroid

        # Sort all frontiers from biggest to smallest
        all_frontiers = sorted(all_frontiers, key= lambda f: len(f), reverse = True)
        # Reduce list of frontiers to list of their nearest-to-centroid points
        return map(get_point, all_frontiers)

    # Compute Euclidean distance between two points
    def distance(self, point1, point2):
        return math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))
