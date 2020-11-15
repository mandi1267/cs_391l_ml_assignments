"""
Contains objects for computing rewards for each module
"""

from grid_world import *
class RewardEvaluator(object):
    """
    Defines interface for the reward evaluator.
    """

    def __init__(self):
        pass

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        pass

class ExecutionOnlyRewardEvaluator(RewardEvaluator):
    """
    Reward evaluator only takes the action (has a constant 0 reward).
    """
    def __init__(self):
        super(RewardEvaluator, self).__init__()

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """

        grid_world.moveAgent(action)
        return 0

class SidewalkRewardEvaluator(RewardEvaluator):
    """
    Computes rewards for the sidewalk module.
    """

    def __init__(self, off_sidewalk_penalty):
        """
        Create the sidewalk reward evaluator.

        :param off_sidewalk_penalty: Penalty for going off the sidewalk (should be negative).
        """
        super(RewardEvaluator, self).__init__()
        self.off_sidewalk_penalty = off_sidewalk_penalty

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        grid_world.moveAgent(action)
        if (grid_world.onSidewalk()):
            return 0.0
        return self.off_sidewalk_penalty

class LitterRewardEvaluator(RewardEvaluator):
    """
    Computes rewards for the sidewalk module.
    """

    def __init__(self, litter_reward):
        """
        Create the litter reward evaluator.

        :param litter_reward: Reward for picking up litter.
        """
        super(RewardEvaluator, self).__init__()
        self.litter_reward = litter_reward

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        action_result = grid_world.moveAgent(action)
        if (action_result == grid_world.current_obstacle_litter_map.litter_num):
            return self.litter_reward
        return 0.0

class ObstacleRewardEvaluator(RewardEvaluator):
    """
    Computes rewards for the obstacle module.
    """

    def __init__(self, obstacle_penalty):
        """
        Create the obstacle reward evaluator.

        :param obstacle_penalty: Penalty for hitting an obstacle (should be negative).
        """
        super(RewardEvaluator, self).__init__()
        self.obstacle_penalty = obstacle_penalty

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        action_result = grid_world.moveAgent(action)
        if (action_result == grid_world.current_obstacle_litter_map.obstacle_num):
            return self.obstacle_penalty
        return 0.0

class ForthRewardEvaluator(RewardEvaluator):
    """
    Computes rewards for the obstacle module.
    """

    def __init__(self, goal_reward, forward_movement_reward):
        """
        Create the forth reward evaluator.

        :param goal_reward:             Reward for reaching a goal cell.
        :param forward_movement_reward: Amount to reward for a forward movement, and penalize for a backward movement
        (negated for action 1, positive for action 2).
        """
        super(RewardEvaluator, self).__init__()
        self.goal_reward = goal_reward
        self.forward_movement_reward = forward_movement_reward

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        prev_agent_pos = grid_world.agent_position
        grid_world.moveAgent(action)
        pos_diff = grid_world.agent_position.y - prev_agent_pos.y
        action_reward = pos_diff * self.forward_movement_reward
        if (grid_world.atGoal()):
            return action_reward + self.goal_reward
        return action_reward

class StuckRewardEvaluator(RewardEvaluator):
    """
    Computes rewards for the stuck module.
    """

    def __init__(self, stuck_penalty):
        """
        Create the stuck reward evaluator.

        :param stuck_penalty: Penalty for getting stuck (being in the same place after 4 moves). Should be negative.
        """
        super(RewardEvaluator, self).__init__()
        self.stuck_penalty = stuck_penalty

    def takeActionAndGetReward(self, grid_world, action):
        """
        Take the given action in the grid world and compute the reward for doing so.

        :param grid_world:  Grid world to move agent within based on action.
        :param action:      Action to take. Actions can be 1 (up), 2 (down), 3 (left), 4(right).

        :return: Reward for the action.
        """
        grid_world.moveAgent(action)
        distance_moved = grid_world.getLastFourActDistMoved()
        if (distance_moved == 0):
            return self.stuck_penalty
        return 0

