"""
Contains representation of state for each module.
"""

class LitterState(object):
    """
    Representation of state for the litter module.
    """
    def __init__(self, nearest_litter_x_dist, nearest_litter_y_dist):
        """
        Create the litter state.

        :param nearest_litter_x_dist: Signed x distance to nearest litter.
        :param nearest_litter_y_dist: Signed y distance to nearest litter.
        """
        self.nearest_litter_x_dist = nearest_litter_x_dist
        self.nearest_litter_y_dist = nearest_litter_y_dist

    def __str__(self):
        return "LitterState(Nearest:" + str(self.nearest_litter_x_dist) + ", " + str(self.nearest_litter_y_dist) + ")"

    def __hash__(self):
        return hash((self.nearest_litter_x_dist, self.nearest_litter_y_dist))

    def __eq__(self, other):
        return (self.nearest_litter_x_dist == other.nearest_litter_x_dist) and (self.nearest_litter_y_dist == other.nearest_litter_y_dist)

class LocalObstacleState(object):
    """
    Representation of state for the obstacle module.
    """
    def __init__(self, obs_x_plus, obs_y_plus, obs_x_min, obs_y_min):
        """
        Create the obstacle state.

        :param obs_x_plus: True if there is an obstacle to the right of the agent, false if there is no obstacle.
        :param obs_y_plus: True if there is an obstacle below the agent, false if there is no obstacle.
        :param obs_x_min: True if there is an obstacle to the left of the agent, false if there is no obstacle.
        :param obs_y_min: True if there is an obstacle above the agent, false if there is no obstacle.
        """
        self.obs_x_plus = obs_x_plus
        self.obs_y_plus = obs_y_plus
        self.obs_x_min = obs_x_min
        self.obs_y_min = obs_y_min

    def __str__(self):
        return "LocObstState(xMin:" + str(self.obs_x_min) + ",xPlus:" + str(self.obs_x_plus) + ",yMin:" + str(self.obs_y_min) + ",yPlus:" + str(self.obs_y_plus) + ")"

    def __hash__(self):
        return hash((self.obs_x_min, self.obs_x_plus, self.obs_y_plus, self.obs_y_min))

    def __eq__(self, other):
        return (self.obs_x_min == other.obs_x_min) and (self.obs_y_min == other.obs_y_min) and \
               (self.obs_x_plus == other.obs_x_plus) and (self.obs_y_plus == other.obs_y_plus)

class ObstacleState(object):
    """
    Representation of state for the obstacle module.
    """
    def __init__(self, nearest_obstacle_x_dist, nearest_obstacle_y_dist):
        """
        Create the obstacle state.

        :param nearest_litter_x_dist: Signed x distance to nearest obstacle.
        :param nearest_litter_y_dist: Signed y distance to nearest obstacle.
        """
        self.nearest_obstacle_x_dist = nearest_obstacle_x_dist
        self.nearest_obstacle_y_dist = nearest_obstacle_y_dist

    def __str__(self):
        return "ObstacleState(Nearest:" + str(self.nearest_obstacle_x_dist) + ", " + str(self.nearest_obstacle_y_dist) + ")"

    def __hash__(self):
        return hash((self.nearest_obstacle_x_dist, self.nearest_obstacle_y_dist))

    def __eq__(self, other):
        return (self.nearest_obstacle_x_dist == other.nearest_obstacle_x_dist) and (self.nearest_obstacle_y_dist == other.nearest_obstacle_y_dist)

class SidewalkState(object):
    """
    Representation of state for the sidewalk module.
    """
    def __init__(self, distance_to_nearest_sidewalk_center):
        """
        Create the sidewalk state.

        :param distance_to_nearest_sidewalk_center: Signed distance to sidewalk center.
        """
        self.distance_to_nearest_sidewalk_center = distance_to_nearest_sidewalk_center

    def __str__(self):
        return "SidewalkState(DistToSidewalkCenter:" + str(self.distance_to_nearest_sidewalk_center) + ")"

    def __hash__(self):
        return int(self.distance_to_nearest_sidewalk_center)

    def __eq__(self, other):
        return (self.distance_to_nearest_sidewalk_center == other.distance_to_nearest_sidewalk_center)

class ForthState(object):
    """
    Representation of state for the forth module.
    """
    def __init__(self, distance_to_nearest_goal):
        """
        Create the forth state.

        :param distance_to_nearest_goal: Distance to nearest goal
        """
        self.distance_to_nearest_goal = distance_to_nearest_goal

    def __str__(self):
        return "ForthState(DistanceToNearestGoal:" + str(self.distance_to_nearest_goal) + ")"

    def __hash__(self):
        return hash(self.distance_to_nearest_goal)

    def __eq__(self, other):
        return (self.distance_to_nearest_goal == other.distance_to_nearest_goal)

class StuckState(object):
    """
    Representation of the stuck state.
    """
    def __init__(self, last_four_actions):
        self.last_four_actions = last_four_actions

    def __str__(self):
        return "StuckState(Last4Actions:" + str(self.last_four_actions) + ")"

    def __hash__(self):
        last_actions = self.last_four_actions[:]
        while (len(last_actions) < 4):
            last_actions.append(-1)
        return hash((last_actions[0], last_actions[1], last_actions[2], last_actions[3]))

    def __eq__(self, other):
        if (len(self.last_four_actions) != len(other.last_four_actions)):
            return False
        else:
            for i in range(len(self.last_four_actions)):
                if (self.last_four_actions[i] != other.last_four_actions[i]):
                    return False
            return True

class CombinedState(object):
    """
    State that is the combination of states for all modules.
    """

    def __init__(self, sidewalk_state, litter_state, obstacle_state, forth_state, stuck_state):
        """
        Create the combined state.

        :param sidewalk_state:  State for sidewalk module.
        :param litter_state:    State for litter module.
        :param obstacle_state:  State for stuck module.
        :param forth_state:     State for forth module.
        :param stuck_state:     State for stuck module.
        """
        self.sidewalk_state = sidewalk_state
        self.litter_state = litter_state
        self.obstacle_state = obstacle_state
        self.forth_state = forth_state
        self.stuck_state = stuck_state