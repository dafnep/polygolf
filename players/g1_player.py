import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import shape, Polygon, LineString , Point
from sympy import Point2D
import math
import matplotlib.pyplot as plt
import constants
import heapq



class Cell:
    def __init__(self, point, man_dist, est_stk, actual_cost, previous):
        man_wt = 1
        stk_wt = 2
        self.point = point
        self.heuristic_cost = man_wt * man_dist + stk_wt * est_stk
        self.actual_cost = actual_cost
        self.previous = previous
    
    def total_cost(self):
        return self.heuristic_cost + self.actual_cost

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(self.point)
    


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.centers =[]
        self.centerset =set()
        self.centers2 = []
        self.target = (0,0)
        self.turns = 0
        self.map = None
        self.map_shapely = None
        self.unit = 20
        self.ex_strokes = []
        self.man_dist = []

    def point_inside_polygon(self,poly, p) -> bool:
    # http://paulbourke.net/geometry/polygonmesh/#insidepoly
        n = len(poly)
        inside = False
        p1 = poly[0]
        for i in range(1, n + 1):
            p2 = poly[i % n]
            if min(p1.y, p2.y) < p.y <= max(p1.y, p2.y) and p.x <= max(p1.x, p2.x) and p1.x != p2.y:
                xints = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                if p1.x == p2.x or p.x <= xints:
                    inside = not inside
            p1 = p2
        return inside
        
    def segmentize_map(self, golf_map ):
        self.logger.info("Segmentizing")
        area_length = self.unit
        std_dev = 1
        beginx = area_length/2
        beginy = area_length/2
        endx = constants.vis_width
        endy = constants.vis_height
        node_centers = []
        node_centers2 =[]
        count = 0
        for i in range(int(beginx), int(endx), area_length):
            tmp = []    
            for j in range(int(beginy), int(endy), area_length):
                representative_point = Point2D(i,j)
                #maybe if its not in the polygon check points around in order to use those
                if self.point_inside_polygon(golf_map.vertices,sympy.geometry.Point2D(i , j)):
                    tmp.append(representative_point)
                    node_centers.append(representative_point)
                    self.centerset.add((i,j))
                    count += 1
                else:
                    tmp.append(None)
            self.logger.info(f"Segmentized Row {i}")
            node_centers2.append(tmp)

        self.logger.info(f"Cells {count}")
        # ex_strokes: expected number of strokes to reach cell, using 1 std. dev.
        # man_dist: manhattan distance within the polygon from the target
        ex_strokes = [[100 for _ in node_centers2[0]] for _ in node_centers2]
        man_dist = [[-1 for _ in node_centers2[0]] for _ in node_centers2]
        self.logger.info("Calculating Man Distance")
        # BFS through grid to populate manhattan distance and calculate estimated number of strokes from target
        tx = int(self.target[0] / self.unit)
        ty = int(self.target[1] / self.unit)
        man_dist[tx][ty] = 0
        current_points = [(tx,ty)]
        movement = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        count2 = 0
        while not len(current_points) == 0:
            current = current_points.pop(0)
            for move in movement:
                next_point = (current[0] + move[0], current[1] + move[1])
                x = next_point[0]
                y = next_point[1]
                if len(man_dist) > x >= 0 and len(man_dist[0]) > y >= 0 and node_centers2[x][y] is not None \
                        and man_dist[x][y] == -1:
                    self.logger.info(next_point)
                    man_dist[x][y] = man_dist[current[0]][current[1]] + 1
                    # number of strokes approximated as along straight line distance from target
                    ex_strokes[x][y] = \
                        int(np.linalg.norm(np.array((self.unit * (tx - x), self.unit*(ty - y))).astype(float)
                                           / (200 + std_dev * (200 / self.skill)))) + 1
                    current_points.append(next_point)
                elif node_centers2[x][y] is None:
                    man_dist[x][y] = np.infty
                    ex_strokes[x][y] = np.infty
            count2 += 1
            if (100 * count2 / count) % 10 == 0:
                self.logger.info(f"% of Nodes = {count2 / count}")
        final_man_dist = []
        final_ex_strokes = []
        for col in man_dist:
            col = [np.infty if i == -1 else i for i in col]
            final_man_dist.append(col)
        for col in ex_strokes:
            col = [np.infty if i == -1 else i for i in col]
            final_ex_strokes.append(col)
        self.centers = node_centers
        self.centers2 = node_centers2
        self.ex_strokes = final_ex_strokes
        self.man_dist = final_man_dist
        self.logger.info(list(zip(*final_man_dist)))

    def get_manhattan_distance(self, point):
        x  = int(point[0] / self.unit)
        y = int(point[1] / self.unit)
        return self.man_dist[x][y]

    def get_est_strokes(self, point):
        x  = int(point[0] / self.unit)
        y = int(point[1] / self.unit)
        return self.ex_strokes[x][y]

    def sector(self, center, start_angle, end_angle, radius):
        def polar_point(origin_point, angle,  distance):
            return [origin_point.x + math.sin(math.radians(angle)) * distance, origin_point.y + math.cos(math.radians(angle)) * distance]
        steps=10
        if start_angle > end_angle:
            t = start_angle
            start_angle = end_angle
            end_angle = t
        else:
            pass
        step_angle_width = (end_angle-start_angle) / steps
        sector_width = (end_angle-start_angle) 
        segment_vertices = []
        segment_vertices.append(polar_point(center, 0,0))
        segment_vertices.append(polar_point(center, start_angle,radius))
        for z in range(1, steps):
            segment_vertices.append((polar_point(center, start_angle + z * step_angle_width,radius)))
        segment_vertices.append(polar_point(center, start_angle+sector_width,radius))
        return Polygon(segment_vertices)

        
    def is_safe(self, d, angle, start_point):
        #to do add confidence bounds
        angle_2std = ((1/(2*self.skill)))
        distance_2std = (d/self.skill)
        min_distance = d-distance_2std
        max_distance = d+(d*0.1)+distance_2std
        begin_line1 = (start_point.x + (min_distance)*math.cos(angle - angle_2std ), start_point.y + (min_distance)*math.sin(angle -angle_2std ))
        begin_line2 = (start_point.x + (min_distance)*math.cos(angle + angle_2std), start_point.y + (min_distance)*math.sin(angle + angle_2std))
        end_line1 = (start_point.x + (max_distance)*math.cos(angle - angle_2std ), start_point.y + (max_distance)*math.sin(angle - angle_2std))
        end_line2 = (start_point.x + (max_distance)*math.cos(angle + angle_2std ), start_point.y + (max_distance)*math.sin(angle + angle_2std))
        L1 = LineString([Point(begin_line1), Point(end_line1)])
        L2 = LineString([Point(begin_line2), Point(end_line2)])
        check1 = L1.within(self.map_shapely)
        check2 = L2.within(self.map_shapely)
        if (check1 &   check2):
            return 1
        else:
            return 0



    def is_neighbour(self, curr_loc, target_loc):
        current_point = curr_loc
        target_point = target_loc
        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)
        max_dist = 200 + self.skill
        required_dist = np.linalg.norm(current_point - target_point)
        angle = sympy.atan2(target_point[1] - current_point[1], target_point[0] - current_point[0])
        #is reachable
        if (np.linalg.norm(current_point - target_point) < max_dist):
            #is safe to land
            #if(Point2D(self.target).equals(Point2D(target_loc))):
                #return 1
            if (self.is_safe(required_dist,angle,Point2D(curr_loc))):
                return 1
            else:
                return 0
            #return 1
        else:
            return 0

    def adjacent_cells(self, point, closedSet):
        neighbours = []
        if self.is_neighbour(point, self.target):
            print('target close!')
            neighbours.append(self.target)
        for center in self.centers:
            if center.equals(Point2D(point)):
                continue
            if tuple(center) in closedSet:
                continue
            if self.is_neighbour(point, center):
                neighbours.append( tuple(center))
        return neighbours

  
    def aStar( self, current, end):
        cur_loc = tuple(current)
        current = Cell(cur_loc, self.get_manhattan_distance(cur_loc), self.get_est_strokes(cur_loc), 0.0 , cur_loc)
        openSet = set()
        node_dict = {}
        node_dict[(cur_loc)] = 0.0
        openHeap = []
        closedSet = set()
        openSet.add(cur_loc)
        openHeap.append(current)
        while openSet:
            next_pointC = heapq.heappop(openHeap)
            next_point = next_pointC.point
            #reached the goal
            if np.linalg.norm(np.array(self.target).astype(float) - np.array(next_point).astype(float)) <= 5.4 / 100.0:
                while next_pointC.previous.point != cur_loc:
                    next_pointC = next_pointC.previous
                return next_pointC.point
            openSet.remove(next_point)
            closedSet.add(next_point)
            neighbours = self.adjacent_cells(next_point, closedSet)
            for n in neighbours :
                if n not in closedSet:
                    cell = Cell(n, self.get_manhattan_distance(n), self.get_est_strokes(n), next_pointC.actual_cost +1 , next_pointC)
                    if n not in openSet and (next_pointC.actual_cost +1 <=10 - self.turns):
                        if (n not in node_dict or cell.total_cost() < node_dict(n)):
                            openSet.add(n)
                            node_dict[n] = cell.total_cost()
                            heapq.heappush(openHeap, cell )
        return []

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based n current game state returns the distance and angle, the shot must be played 

        Args:
            score (int): Your total score including current turn
            golf_map (sympy.Polygon): Golf Map polygon
            target (sympy.geometry.Point2D): Target location
            curr_loc (sympy.geometry.Point2D): Your current location
            prev_loc (sympy.geometry.Point2D): Your previous location. If you haven't played previously then None
            prev_landing_point (sympy.geometry.Point2D): Your previous shot landing location. If you haven't played previously then None
            prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None

        Returns:
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """
        if (prev_loc == None):
            self.target = tuple(target)
            self.map = golf_map
            shape_map = golf_map.vertices
            self.map_shapely = Polygon(shape_map)
            self.segmentize_map(golf_map)
            self.logger.info("Initialziation Done")

        required_dist = 0
        angle = 0
        if curr_loc.distance(target) <= 200 + self.skill:
            required_dist = curr_loc.distance(target)
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        else:
            next_point = self.aStar(curr_loc, target)
            required_dist = curr_loc.distance(next_point)
            angle = sympy.atan2(next_point[1] - curr_loc.y, next_point[0] - curr_loc.x)
        #angle2  = math.degrees(angle)
        #a =  self.positionSafety( distance, angle2, curr_loc.evalf(), golf_map)
        """ if (next_point[1] == self.target[1] and next_point[0] == self.target[0]):
            if(required_dist>20):
                required_dist = 0.9*required_dist"""

        self.turns = self.turns + 1
        return (required_dist, angle)