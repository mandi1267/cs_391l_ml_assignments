"""
This defines the grid world.

More specifically, given sidewalk bounds, grid bounds, and litter and obstacle positions, it defines the transition
function, actions, and functions for retrieving module-specific state.
Does not define the rewards of the MDP
"""

import copy
from states import *
from grid_rep import *

class GridWorld(object):
    """
    Class representing a world and the state and necessary history of the agent within the world.
    """

    def __init__(self, init_obstacle_litter_map, agent_start):
        """
        Create the grid world.

        :param init_obstacle_litter_map:    Initial litter and obstacle state, grid boundaries, and sidewalk boundaries
                                            (in the form of a GridRepresentation object).
        :param agent_start:                 Initial agent position (Position object).
        """
        self.init_obstacle_litter_map = init_obstacle_litter_map

        # Create a copy so that we can keep the initial state and the most up to date state
        self.current_obstacle_litter_map = init_obstacle_litter_map.copy()
        self.agent_start = agent_start
        self.agent_position = copy.deepcopy(agent_start)
        self.trajectory = []
        self.trajectory.append(self.agent_position)
        self.num_obstacles_hit = 0
        self.off_sidewalk_times = 0
        self.num_steps = 0
        self.num_litters_pickedup = 0
        self.local_obstacle_state_cache = {}

        self.last_four_actions = []

    def copy(self):
        """
        Make a deep copy of this grid world using the initial state and agent start position.

        :return: Deep copy of this grid world using the initial state and agent start position.
        """
        return GridWorld(self.init_obstacle_litter_map.copy(), self.agent_start)

    def moveAgent(self, action):
        """
        Have the agent take the given action. Actions can be 1 (up), 2 (down), 3 (left), 4(right). Action will not
        succeed if it would take the agent into an obstacle cell or off the grid.

        :param action: Action to take.

        :return: Type of the cell that the agent tried to go into (0 for empty cell, 1 for litter, 2 for obstacle).
        Does not include sidewalk/off sidewalk since litter may be on or off sidewalk.
        """

        # Actions can be 1 (up), 2 (down), 3 (left), 4(right)
        self.last_four_actions.append(action)
        if (len(self.last_four_actions) > 4):
            self.last_four_actions.pop(0)

        (cell_contains_obs, new_pos) = self.getAgentPosAfterAction(action)

        self.agent_position = new_pos
        self.trajectory.append(self.agent_position)

        action_result = self.current_obstacle_litter_map.visitCell(self.agent_position)
        if (action_result == self.current_obstacle_litter_map.litter_num):
            self.num_litters_pickedup += 1

        if (not self.onSidewalk()):
            self.off_sidewalk_times += 1

        self.num_steps += 1

        if (cell_contains_obs):
            self.num_obstacles_hit += 1
            return self.current_obstacle_litter_map.obstacle_num
        else:
            return action_result


    def getAgentPosAfterAction(self, action):
        """
        Get the position of the agent after a hypothetical action.

        :param action:  Action to consider. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Position of the agent after the action (doesn't move when agent would hit an obstacle or wall
        boundaries).
        """

        potential_new_pos = Position(self.agent_position.x, self.agent_position.y)
        if (action == 1):
            if (self.agent_position.y > 0):
                potential_new_pos.y = self.agent_position.y - 1
        elif (action == 2):
            # Assuming we terminate after getting to goal (y = max -1), so we can automatically advance without checking
            # bounds
            potential_new_pos.y = self.agent_position.y + 1
        elif (action == 3):
            if (self.agent_position.x > 0):
                potential_new_pos.x = self.agent_position.x - 1
        elif (action == 4):
            if (self.agent_position.x < (self.current_obstacle_litter_map.bounds_x - 1)):
                potential_new_pos.x = self.agent_position.x + 1
        else:
            print("Invalid action " + str(action))
            return Position(-1, -1)

        # Check if an obstacle would've prevented successful execution of the action
        cell_contains_obs = self.current_obstacle_litter_map.getEntryAtCell(potential_new_pos) == self.current_obstacle_litter_map.obstacle_num
        if (cell_contains_obs):
            potential_new_pos = Position(self.agent_position.x, self.agent_position.y)

        return (cell_contains_obs, potential_new_pos)

    def getObstacleState(self, agent_pos=None):
        """
        Get the obstacle state (version that uses distance to closest).

        :param agent_pos: Position of the agent to get the nearest obstacle for. If none, use the agent's current
        position.

        :return: Obstacle state containing relative position of nearest obstacle.
        """
        if (agent_pos == None):
            agent_pos = self.agent_position

        # Get x and y to nearest obstacle
        nearest_obs = self.current_obstacle_litter_map.getPosOfNearestObjOfType(self.current_obstacle_litter_map.obstacle_num, agent_pos)
        if (nearest_obs.x < self.current_obstacle_litter_map.bounds_x):
            return ObstacleState(nearest_obs.x - agent_pos.x, nearest_obs.y - agent_pos.y)
        else:
            return ObstacleState(nearest_obs.x, nearest_obs.y)

    def getLitterState(self, agent_pos=None):
        """
        Get the litter state.

        :param agent_pos: Position of the agent to get the nearest litter for. If none, use the agent's current
        position.

        :return: Litter state containing relative position of nearest litter.
        """
        if (agent_pos == None):
            agent_pos = self.agent_position
        # Get x and y to nearest litter
        nearest_litter = self.current_obstacle_litter_map.getPosOfNearestObjOfType(self.current_obstacle_litter_map.litter_num, agent_pos)
        if (nearest_litter.x < self.current_obstacle_litter_map.bounds_x):
            return LitterState(nearest_litter.x - agent_pos.x, nearest_litter.y - agent_pos.y)
        else:
            return LitterState(nearest_litter.x, nearest_litter.y)

    def getSidewalkState(self, agent_pos=None):
        """
        Get the sidewalk state.

        :param agent_pos: Position of the agent to get the state for. If none, use the agent's current
        position.

        :return: Sidewalk state containing signed distance to sidewalk center.
        """
        if (agent_pos == None):
            agent_pos = self.agent_position
        return SidewalkState(self.current_obstacle_litter_map.sidewalk_center_x - agent_pos.x)

    def getForthState(self, agent_pos=None):
        """
        Get the forth state.

        :param agent_pos: Position of the agent to get the state for. If none, use the agent's current
        position.

        :return: Forth state containing distance to the end of the sidewalk.
        """
        if (agent_pos == None):
            agent_pos = self.agent_position
        return ForthState(self.current_obstacle_litter_map.bounds_y - agent_pos.y)

    def atGoal(self, agent_pos=None):
        """
        Determine if the agent is as at a goal cell.

        :param agent_pos: Position of the agent check against the goal. If none, use the agent's current
        position.

        :return: True if the agent is at a goal cell, false if it is not at a goal cell.
        """
        if (agent_pos == None):
            agent_pos = self.agent_position
        if (agent_pos.y == (self.current_obstacle_litter_map.bounds_y - 1)):
            return True
        return False

    def onSidewalk(self, agent_pos=None):
        """
        Determine if the agent is on the sidewalk.

        :param agent_pos: Position of the agent check against the sidewalk. If none, use the agent's current
        position.

        :return: Determine if the agent is on the sidewalk
        """
        if (agent_pos == None):
            agent_pos = self.agent_position
        return self.current_obstacle_litter_map.onSidewalk(agent_pos)

    def getLastFourActDistMoved(self):
        """
        Get the distance the agent has moved in the last 4 actions.

        :return: Distance the agent has moved in the last 4 actions.
        """
        if (len(self.trajectory) < 5):
            init_pos = self.agent_start
        else:
            init_pos = self.trajectory[-5]
        curr_pos = self.trajectory[-1]
        traveled_dist = abs(init_pos.x - curr_pos.x) + abs(init_pos.y - curr_pos.y)
        return traveled_dist

    def getStuckState(self):
        """
        Get stuck state (the last 4 actions taken).

        :return: the stuck state.
        """
        return StuckState(self.last_four_actions[:])

    def getLocalObstacleState(self):
        """
        Get the local obstacle state (if there are obstacles neighboring the agent).

        :return: The local obstacle state.
        """

        lookup_pos = (self.agent_position.x, self.agent_position.y)
        if (lookup_pos in self.local_obstacle_state_cache):
            return self.local_obstacle_state_cache[lookup_pos]
        obs_num = self.current_obstacle_litter_map.obstacle_num
        xmin_pos = Position(self.agent_position.x - 1, self.agent_position.y)
        xplus_pos = Position(self.agent_position.x + 1, self.agent_position.y)
        ymin_pos = Position(self.agent_position.x, self.agent_position.y - 1)
        yplus_pos = Position(self.agent_position.x, self.agent_position.y + 1)

        obs_at_xmin = False
        obs_at_xplus = False
        obs_at_ymin = False
        obs_at_yplus = False

        if (not self.current_obstacle_litter_map.posOffBoard(xmin_pos)):
            obs_at_xmin = (self.current_obstacle_litter_map.getEntryAtCell(xmin_pos) == obs_num)
        if (not self.current_obstacle_litter_map.posOffBoard(xplus_pos)):
            obs_at_xplus = (self.current_obstacle_litter_map.getEntryAtCell(xplus_pos) == obs_num)
        if (not self.current_obstacle_litter_map.posOffBoard(ymin_pos)):
            obs_at_ymin = (self.current_obstacle_litter_map.getEntryAtCell(ymin_pos) == obs_num)
        if (not self.current_obstacle_litter_map.posOffBoard(yplus_pos)):
            obs_at_yplus = (self.current_obstacle_litter_map.getEntryAtCell(yplus_pos) == obs_num)
        loc_obs_state = LocalObstacleState(obs_x_min=obs_at_xmin, obs_x_plus=obs_at_xplus, obs_y_min=obs_at_ymin, obs_y_plus=obs_at_yplus)
        self.local_obstacle_state_cache[lookup_pos] = loc_obs_state
        return loc_obs_state

