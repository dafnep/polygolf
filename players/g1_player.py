import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import shape, Polygon, LineString, Point
from sympy import Point2D
import math
import constants
import heapq
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, point, target, step_cost, h_cost=None, parent=None):
        """
        Cell class for the A star algorithm
        :param point:   A tuple of float (float, float) that indicate the current location
        :param target:   A tuple of float (float, float) that indicate the target location
        :param step_cost:    the step cost (forward cost)
        :param h_cost:   the heuristic cost
        :param parent:    the parent of the current point
        """
        self.point = point
        self.h_cost = h_cost if h_cost else np.linalg.norm(
            np.array(point).astype(float) - np.array(target).astype(float))
        self.step_cost = step_cost
        self.parent = parent

    def total_cost(self):
        return self.h_cost + self.step_cost

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(self.point)

    def __repr__(self):
        return "Cell(point: {} actual_cost: {} h_cost: {} total_cost: {} parent: {})".format(self.point,
                                                                                             self.step_cost,
                                                                                             self.h_cost,
                                                                                             self.total_cost(),
                                                                                             self.parent)


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.skill = skill
        self.max_dist = constants.max_dist + self.skill
        print(self.max_dist)
        self.rng = rng
        self.logger = logger
        self.center_points = None  # A list of np array points
        self.turns = 0
        self.map = None
        self.map_shapely = None
        self.bounds = None
        self.prev_pt = None
        self.prev_ang = None
        self.iteration = 5
        self.cell_length = 5
        self.is_greedy = False

    def segmentize_map(self, cell_length: float = 5.0):
        """
        Segmentize the map into grids of size = cell_length

        :param cell_length: the length of the cell
        :return: None
        """
        (min_x, min_y, max_x, max_y) = self.bounds
        x_num_cells = (max_x - min_x) // self.cell_length
        y_num_cells = (max_y - min_y) // self.cell_length
        x_coords, y_coords = np.meshgrid(np.linspace(float(min_x), float(max_x), x_num_cells),
                                         np.linspace(float(min_y), float(max_y), y_num_cells))
        print(x_num_cells, x_num_cells)
        center_points = []
        m, n = x_coords.shape
        for i in range(m):
            for j in range(n):
                center_p = np.array([x_coords[i, j], y_coords[i, j]]).astype(float)
                if self.map_shapely.contains(Point(center_p)):
                    center_points.append(center_p)
        self.center_points = center_points
        print(len(self.center_points))

    def is_landing_area_safe(self, distance: float, angle: float, start_point: Point, alpha: float = 2.0) -> bool:
        """
        Create the possible landing area shot from the start_point with a distance and angle

        :param distance:  distance to aim
        :param angle:     angle to aim
        :param start_point:  starting point
        :param alpha:    confidence level
        :return:  a boolean value indicating whether the landing area is safe
        """
        angle_std = alpha * (1 / (2 * self.skill))
        distance_std = alpha * distance / self.skill
        shortest_p1 = (start_point.x + (distance - distance_std) * math.cos(angle - angle_std),
                       start_point.y + (distance - distance_std) * math.sin(angle - angle_std))
        shortest_p2 = (start_point.x + (distance - distance_std) * math.cos(angle + angle_std),
                       start_point.y + (distance - distance_std) * math.sin(angle + angle_std))
        furthest_p1 = (start_point.x + (distance + distance_std) * math.cos(angle - angle_std),
                       start_point.y + (distance + distance_std) * math.sin(angle - angle_std))
        furthest_p2 = (start_point.x + (distance + distance_std) * math.cos(angle + angle_std),
                       start_point.y + (distance + distance_std) * math.sin(angle + angle_std))
        poly = Polygon([shortest_p1, shortest_p2, furthest_p2, furthest_p1])

        return poly.intersection(self.map_shapely).area / poly.area > 0.7

    def is_landing_pt_safe(self, iteration: int, curr_loc: Point, distance: float, angle: float) -> bool:
        """
        Check if a point is safe by simulation
        :param iteration:  number of iterations we want to try this distance and angle
        :param curr_loc:   Current location
        :param distance:   Aimed distance
        :param angle:      Aimed angle
        :return:   True if it landed successfully iteration of times.
        """

        valid_cnt = 0
        for _ in range(iteration):
            actual_distance = self.rng.normal(distance, distance / self.skill)
            actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

            if self.max_dist >= distance >= constants.min_putter_dist:
                landing_point = Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                      curr_loc.y + actual_distance * np.sin(actual_angle))
                final_point = Point(
                    curr_loc.x + (1. + constants.extra_roll) * actual_distance * sympy.cos(actual_angle),
                    curr_loc.y + (1. + constants.extra_roll) * actual_distance * sympy.sin(actual_angle))
            else:
                landing_point = curr_loc
                final_point = Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                    curr_loc.y + actual_distance * np.sin(actual_angle))

            segment_land = LineString([landing_point, final_point])
            if segment_land.within(self.map_shapely):
                valid_cnt += 1

        return valid_cnt == iteration

    def is_nei_valid(self, curr_loc: Tuple[float, float], target_loc: Tuple[float, float]) -> bool:
        """
        Check if the neighbour point is within the maximum reachable distance

        :param curr_loc:  the current location
        :param target_loc:   the target location
        :return:  return True if it is valid
        """
        current_point = np.array(curr_loc).astype(float)
        target_point = np.array(target_loc).astype(float)
        required_dist = np.linalg.norm(current_point - target_point)
        if required_dist <= self.max_dist:
            # Take care of the 10% extra rolling
            roll_factor = 1. + constants.extra_roll if required_dist > 20 else 1.0
            required_dist /= roll_factor
            angle = np.arctan2(target_point[1] - current_point[1], target_point[0] - current_point[0])
            if self.is_landing_pt_safe(self.iteration, Point(curr_loc), required_dist, angle):
                return True
            else:
                return False
        return False

    def get_nei(self, curr_loc: Tuple[float, float]) -> Tuple[float, float]:
        """
        Given the current cell, yield its reachable neighbours
        :param target:  the location of the target cell
        :param curr_loc:  the location of the current cell
        :return:  Neighbour points
        """
        for center_point in self.center_points:
            # Skip if it's the same point
            if np.linalg.norm(center_point - np.array(curr_loc).astype(float)) < 0.0001:
                continue
            if self.is_nei_valid(curr_loc, center_point):
                yield tuple(center_point)

    def aStar(self, curr_loc: Point2D, end_loc: Point2D) -> Tuple[float, float]:
        """
        A-star to find the if there is a path to the goal. If there is a path, return the next point on the path

        :param curr_loc:  the current location
        :param end_loc:  the end location (target location)
        :return:    a Point
        """
        start_loc_np, end_loc_np = tuple(curr_loc), tuple(end_loc)
        heap = [Cell(start_loc_np, target=end_loc_np, step_cost=0.0)]
        best_cost = {start_loc_np: heap[0].total_cost()}
        while heap:
            next_cell = heapq.heappop(heap)
            next_pt = next_cell.point
            # Goal test
            dist = np.linalg.norm(np.array(next_pt).astype(float) - np.array(end_loc_np).astype(float))
            print("dist", dist)
            if dist <= 5.4 / 100.0:
                # Backtrack to the parent
                print(next_cell, "next cell")
                while next_cell.parent.point != start_loc_np:
                    next_cell = next_cell.parent
                return next_cell.point
            for nei_pt in self.get_nei(next_pt):
                nei_cell = Cell(nei_pt, end_loc_np, next_cell.step_cost + 1.0, parent=next_cell)
                if nei_pt not in best_cost or best_cost[nei_pt] > nei_cell.total_cost():
                    best_cost[nei_pt] = nei_cell.total_cost()
                    heapq.heappush(heap, nei_cell)

        return None

    def greedy(self, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D) -> Tuple[np.float, np.float]:

        """
        Default greedy algorithm
        :param target:  Target location
        :param curr_loc:  Current location
        :return:
        """

        curr_loc = np.array(curr_loc, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        required_dist = np.linalg.norm(curr_loc-target)
        # required_dist = curr_loc.distance(target)
        roll_factor = 1. + constants.extra_roll
        if required_dist < 20:
            roll_factor = 1.0
        # distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
        # angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        distance = min(200.0 + self.skill, required_dist / roll_factor)
        angle = np.arctan2(target[1] - curr_loc[1], target[0] - curr_loc[0])
        return distance, angle


    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
             curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D,
             prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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
        if not self.map_shapely:
            self.bounds = golf_map.bounds
            self.map_shapely = Polygon(golf_map.vertices)
            self.map = golf_map
            self.segmentize_map()
            pts = np.array(self.center_points)
            self.center_points.append(np.array(target).astype(float))
            np.save("points.npy", pts)

        if not prev_admissible:
            self.iteration += 3

        if self.is_greedy:
            return self.greedy(target, curr_loc)

        next_point = self.aStar(curr_loc, target)
        print(next_point, "next point")
        required_dist = np.linalg.norm(np.array(next_point).astype(float) - np.array(curr_loc).astype(float))
        print(required_dist, "require dist")
        curr_pt = np.array(curr_loc).astype(float)
        angle = np.arctan2(next_point[1] - curr_pt[1], next_point[0] - curr_pt[0])
        self.turns = self.turns + 1
        return required_dist, angle
