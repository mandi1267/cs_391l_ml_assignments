"""
Amanda Adkins
Assignment 5

This file contains the main code for executing assignment 5, including learning the q values for each module and executing various test runs.
"""

from q_learning import *
from grid_world import *
import joblib
import datetime
import math
from rewarder import *
from state_retrievers import *

from plotting import *

def getForwardMovementReward():
    """
    Return the reward for moving forward. Also serves as the penalty for moving backward (0 reward for moving left/right).

    :return: The reward for moving forward/penalty for moving backward.
    """
    return 0.01

def getOffSidewalkPenalty():
    """
    Get the penalty for going off the sidewalk.

    :return: Penalty for going off the sidewalk.
    """
    return -1.0

def getLitterReward():
    """
    Get the reward for picking up a piece of litter.

    :return: Reward for picking up a piece of litter.
    """
    return 3.0

def getObstaclePenalty():
    """
    Get the penalty for running into an obstacle.

    :return: Penalty for running ingo an obstacle.
    """
    return -20.0

def getGoalReward():
    """
    Get the reward for reaching a goal cell (maximum y value on the board).

    :return: Reward for reaching a goal cell.
    """
    return getForwardMovementReward()

def getEpsilon():
    """
    Get the epsilon value. This gives the probability that during q-learning for each module that a random action
    should be taken, instead of the one with the best q value so far.

    :return: Epsilon value, which gives the proportion of time that a random action should be taken during q-learning.
    """
    return 0.25

def getGoalDiscountFactor():
    """
    Get the discount factor to use in the forth module when learning q values.

    :return: Discount factor to use in the forth module.
    """
    return 0.2

def getObstacleDiscountFactor():
    """
    Get the discount factor to use in the obstacle module when learning q values.

    :return: Discount factor to use in the obstacle module.
    """
    return 0.1

def getLitterDiscountFactor():
    """
    Get the discount factor to use in the litter module when learning q values.

    :return: Discount factor to use in the litter module.
    """
    return 0.5

def getSidewalkDiscountFactor():
    """
    Get the discount factor to use in the sidewalk module when learning q values.

    :return: Discount factor to use in the sidewalk module.
    """
    return 0.1

def getGoalLearningRate():
    """
    Get the learning rate to use in the forth module when learning q values.

    :return: Learning rate to use in the forth module.
    """
    return 0.1

def getObstacleLearningRate():
    """
    Get the learning rate to use in the forth module when learning q values.

    :return: Learning rate to use in the forth module.
    """
    return 0.1

def getLitterLearningRate():
    """
    Get the learning rate to use in the litter module when learning q values.

    :return: Learning rate to use in the litter module.
    """
    return 0.1

def getSidewalkLearningRate():
    """
    Get the learning rate to use in the sidewalk module when learning q values.

    :return: Learning rate to use in the sidewalk module.
    """
    return 0.1

def getStuckPenalty():
    """
    Get the penalty for getting "stuck", which is defined as being in the same location after 4 actions. This could
    result from moving in a square or from oscillating back and forth between two locations.

    :return: Penalty for getting stuck.
    """
    return -10

def getStuckLearningRate():
    """
    Get the learning rate to use in the stuck module when learning q values.

    :return: Learning rate to use in the stuck module.
    """
    return 0.1

def getStuckDiscountFactor():
    """
    Get the discount factor to use in the stuck module when learning q values.

    :return: Discount factor to use in the stuck module.
    """
    return 0.1

def trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world=None):
    """
    Train the given module (learn the q table).

    :param reward_evaluator:    Object that returns rewards given the grid world configuration after an action.
    :param state_retriever:     Object that retrieves the module-specific state.
    :param max_steps:           Maximum number of steps that the agent can take in a particular episode. An episode
                                terminates when the agent reaches the goal or when it has taken this many actions.
    :param discount_factor:     Discount factor to use in updating the q values.
    :param learning_rate:       Learning rate to use in updating the q values.
    :param grid_world:          Grid world to train on. If none, a grid world is randomly generated for each episode.

    :return: Learned q table for the module.
    """

    num_epochs = 50000
    possible_actions = [1, 2, 3, 4]
    q_table = QTable(possible_actions)
    epsilon = getEpsilon()

    for i in range(num_epochs):
        if ((i % 1000) == 0):
            print("Epoch " + str(i))
        if (grid_world == None):
            grid_world_copy = randomlyGenerateGridWorld()
        else:
            grid_world_copy = grid_world.copy()
        (actions, q_table, run_results) = executeRun(grid_world_copy, q_table, reward_evaluator, state_retriever,
                                                     epsilon, max_steps, possible_actions, discount_factor,
                                                     learning_rate, True, plot_trajectory=False)
    return q_table


def trainLitterModule(grid_world, max_steps):
    """
    Train the litter module (learn the q table for the litter module).

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the litter module.
    """
    litter_reward = getLitterReward()
    discount_factor = getLitterDiscountFactor()
    learning_rate = getLitterLearningRate()

    reward_evaluator = LitterRewardEvaluator(litter_reward)
    state_retriever = LitterStateRetriever()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_litter_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump((litter_reward, discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def trainObstacleModule(grid_world, max_steps):
    """
    Train the obstacle module (learn the q table for the obstacle module). This version uses the distance to the
    nearest obstacle as the state.

    This version isn't actually used as the final result.

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the obstacle module.
    """
    obstacle_penalty = getObstaclePenalty()
    discount_factor = getObstacleDiscountFactor()
    learning_rate = getObstacleLearningRate()

    reward_evaluator = ObstacleRewardEvaluator(obstacle_penalty)
    state_retriever = ObstacleStateRetriever()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_obstacle_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump((obstacle_penalty, discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def trainLocalObstacleModule(grid_world, max_steps):
    """
    Train the obstacle module (learn the q table for the obstacle module). This version uses the presence/absence of
    obstacles in neighboring cells as the state.

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the obstacle module.
    """
    obstacle_penalty = getObstaclePenalty()
    discount_factor = getObstacleDiscountFactor()
    learning_rate = getObstacleLearningRate()

    reward_evaluator = ObstacleRewardEvaluator(obstacle_penalty)
    state_retriever = LocalObstacleStateRetreiver()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_local_obstacle_rand_world" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump((obstacle_penalty, discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def trainSidewalkModule(grid_world, max_steps):
    """
    Train the sidewalk module (learn the q table for the sidewalk module).

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the sidewalk module.
    """

    off_sidewalk_penalty = getOffSidewalkPenalty()
    discount_factor = getSidewalkDiscountFactor()
    learning_rate = getSidewalkLearningRate()

    reward_evaluator = SidewalkRewardEvaluator(off_sidewalk_penalty)
    state_retriever = SidewalkStateRetriever()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_sidewalk_rand_world" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump((off_sidewalk_penalty, discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def trainForthModule(grid_world, max_steps):
    """
    Train the forth module (learn the q table for the forth module).

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the forth module.
    """

    goal_reward = getGoalReward()
    discount_factor = getGoalDiscountFactor()
    learning_rate = getGoalLearningRate()
    reward_evaluator = ForthRewardEvaluator(goal_reward, getForwardMovementReward())
    state_retriever = ForthStateRetriever()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_forth_" + datetime.datetime.now().replace(microsecond=0).isoformat() + ".pkl"
    joblib.dump((goal_reward, getForwardMovementReward(), discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def trainStuckModule(grid_world, max_steps):
    """
    Train the stuck module (learn the q table for the stuck module).

    :param grid_world: Grid world to operate on. If none, grid worlds will be randomly generated for each episode.
    :param max_steps: Maximum number of steps to take in each episode.

    :return: Q Values for the stuck module.
    """

    stuck_penalty = getStuckPenalty()
    discount_factor = getStuckDiscountFactor()
    learning_rate = getStuckLearningRate()
    reward_evaluator = StuckRewardEvaluator(stuck_penalty)
    state_retriever = StuckStateRetriever()

    # Learn the q values
    q_values = trainModule(reward_evaluator, state_retriever, max_steps, discount_factor, learning_rate, grid_world)

    # Output them to a file so we don't have to constantly relearn
    new_fpath = "assign_5_q_results_stuck_" + datetime.datetime.now().replace(
        microsecond=0).isoformat() + ".pkl"
    joblib.dump((stuck_penalty, discount_factor, learning_rate, q_values), new_fpath)

    return q_values

def runTrajectory(grid_world, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals, sidewalk_q_vals, stuck_q_vals,
                  forth_weight, litter_weight, obstacle_weight, sidewalk_weight, stuck_weight):
    """
    Execute a single test run.

    :param grid_world:      Grid world to test on.
    :param max_steps:       Maximum number of steps that the agent can take before the run ends
    :param forth_q_vals:    Q table for the forth module.
    :param litter_q_vals:   Q table for the litter module.
    :param obstacle_q_vals: Q table for the obstacles module.
    :param sidewalk_q_vals: Q table for the sidewalk module.
    :param stuck_q_vals:    Q table for the stuck module.
    :param forth_weight:    Weight to use for the forth module (multiply q value times this when computing total
                            q value).
    :param litter_weight:   Weight to use for the litter module (multiply q value times this when computing total
                            q value).
    :param obstacle_weight: Weight to use for the obstacle module (multiply q value times this when computing total
                            q value).
    :param sidewalk_weight: Weight to use for the sidewalk module (multiply q value times this when computing total
                            q value).
    :param stuck_weight:    Weight to use for the stuck module (multiply q value times this when computing total
                            q value).

    :return: Results of the run (how many litters, how many obstacles hit, etc).
    """

    epsilon = 0 # Ignored in test mode anyway
    learning_rate = 0 # Ignored in test mode anyway
    discount_factor = 0 # Ignored in test mode anyway

    # Create a q table object that combines each modules q values using the provided weights
    aggregate_q_vals = TestTimeQTable(forth_q_vals, litter_q_vals, obstacle_q_vals, sidewalk_q_vals, stuck_q_vals,
                                      forth_weight, litter_weight, obstacle_weight, sidewalk_weight, stuck_weight)
    reward_evaluator = ExecutionOnlyRewardEvaluator() # Ignored in test mode anyway
    state_retriever = FullStateRetriever()

    # Compute a string of weights to use in the plot title
    weights_str = "forth:" + str(aggregate_q_vals.forth_weight) + ",Sidewalk:" + str(aggregate_q_vals.sidewalk_weight) +\
                  ",Obstacle:" + str(aggregate_q_vals.obstacle_weight) +",Litter:" +\
                  str(aggregate_q_vals.litter_weight) + ",Stuck:" + str(aggregate_q_vals.stuck_weight)

    # Execute the run
    actions, updated_q, run_results = executeRun(grid_world, aggregate_q_vals, reward_evaluator, state_retriever,
                                                 epsilon, max_steps, [1, 2, 3, 4], discount_factor, learning_rate,
                                                 False, plot_trajectory=True, custom_plot_title=weights_str,
                                                 plot_full_trajectory=True, plot_at_times=[i for i in range(max_steps)])

    return run_results

    # Plot actions?

def getQResultsFromFiles():
    """
    Get the q results from a file for each module.

    :return: Tuple containing the litter q values, obstacle q values, sidewalk q values, forth q values, and
    stuck q values.
    """

    forth_file = "assign_5_q_results_forth_rand_world_minimal_goal2020-11-11T00:03:31.pkl"
    litter_file = "assign_5_q_results_litter_half_discount_rand_world2020-11-11T09:03:34.pkl"
    obstacle_file = "assign_5_q_results_local_obstacle_rand_world2020-11-11T00:47:45.pkl"
    sidewalk_file = "assign_5_q_results_sidewalk_rand_world2020-11-10T20:01:51.pkl"
    stuck_file = "assign_5_q_results_stuck_rand_world2020-11-10T22:58:38.pkl"

    (goal_reward, forward_movement_reward, forth_discount_factor, forth_learning_rate, forth_q_values) = joblib.load(forth_file)
    (off_sidewalk_penalty, sidewalk_discount_factor, sidewalk_learning_rate, sidewalk_q_values) = joblib.load(sidewalk_file)
    (obstacle_penalty, obs_discount_factor, obs_learning_rate, obs_q_values) = joblib.load(obstacle_file)
    (litter_reward, litter_discount_factor, litter_learning_rate, litter_q_values) = joblib.load(litter_file)
    (stuck_penalty, stuck_discount_factor, stuck_learning_rate, stuck_q_values) = joblib.load(stuck_file)

    return (litter_q_values, obs_q_values, sidewalk_q_values, forth_q_values, stuck_q_values)


def runTrajectoryWithWeightEval(grid_world, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals, sidewalk_q_vals,
                                stuck_q_vals):

    """
    Execute test runs using different configured weights.

    :param grid_world:      Grid world to use. If None, a grid world should be randomly generated for each run.
    :param max_steps:       Maximum number of steps per run.
    :param forth_q_vals:    Q table for the forth module.
    :param litter_q_vals:   Q table for the litter module.
    :param obstacle_q_vals: Q table for the obstacle module.
    :param sidewalk_q_vals: Q table for the sidewalk module.
    :param stuck_q_vals:    Q table for the stuck module.
    """

    # Run forth only
    if (grid_world == None):
        print("Randomly generating")
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()

    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals,
                                sidewalk_q_vals, stuck_q_vals,
                                1, 0, 0, 0, 0)

    # Run litter only
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()

    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals,
                                sidewalk_q_vals, stuck_q_vals,
                                0, 1, 0, 0, 0)

    # Run obstacle only
    if (grid_world == None):
        print("Randomly generating")
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()

    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals,
                                sidewalk_q_vals, stuck_q_vals,
                                0, 0, 1, 0, 0)

    # Run sidewalk only
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()
    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals,
                                sidewalk_q_vals, stuck_q_vals,
                                0, 0, 0, 1, 0)

    # Run stuck only
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()
    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals,
                                sidewalk_q_vals, stuck_q_vals,
                                0, 0, 0, 0, 1)

    # Run with a configuration that weights the sidewalk higher than litter
    litter_weight = 1
    sidewalk_weight = 2
    forth_weight = 1
    stuck_weight = 0.06
    obstacle_weight = 20

    print("Weights: forth:" + str(forth_weight) + ",Sidewalk:" + str(sidewalk_weight) +
          ",Obstacle:" + str(obstacle_weight) + ",Litter:" + str(litter_weight) +
          ",stuck:" + str(stuck_weight))
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()
    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals,
                                obstacle_q_vals, sidewalk_q_vals, stuck_q_vals, forth_weight,
                                litter_weight, obstacle_weight, sidewalk_weight, stuck_weight)

    # Run with a configuration that weights the litter highly
    litter_weight = 1
    sidewalk_weight = 1
    forth_weight = 1
    stuck_weight = 0.06
    obstacle_weight = 20

    print("Weights: forth:" + str(forth_weight) + ",Sidewalk:" + str(sidewalk_weight) +
          ",Obstacle:" + str(obstacle_weight) + ",Litter:" + str(litter_weight) +
          ",stuck:" + str(stuck_weight))
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()
    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals,
                                obstacle_q_vals, sidewalk_q_vals, stuck_q_vals, forth_weight,
                                litter_weight, obstacle_weight, sidewalk_weight, stuck_weight)

    # Run with a configuration that weights the forth module highly
    litter_weight = 1
    sidewalk_weight = 2
    forth_weight = 30
    stuck_weight = 0.1
    obstacle_weight = 20

    print("Weights: forth:" + str(forth_weight) + ",Sidewalk:" + str(sidewalk_weight) +
          ",Obstacle:" + str(obstacle_weight) + ",Litter:" + str(litter_weight) +
          ",stuck:" + str(stuck_weight))
    if (grid_world == None):
        grid_world_to_use = randomlyGenerateGridWorld()
    else:
        grid_world_to_use = grid_world.copy()
    run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals,
                                obstacle_q_vals, sidewalk_q_vals, stuck_q_vals, forth_weight,
                                litter_weight, obstacle_weight, sidewalk_weight, stuck_weight)

    # # Do a sweep of many parameter combinations and save the results
    # run_num = 0
    # total_run_num = len(litter_weights) * len(sidewalk_weights) * len(obstacle_weights) * len(forth_weights) * len(stuck_weights)
    # results_data = []

    # for obstacle_weight in obstacle_weights:
    #     for stuck_weight in stuck_weights:
    #         for forth_weight in forth_weights:
    #             for sidewalk_weight in sidewalk_weights:
    #                 for litter_weight in litter_weights:
    #                     for meta_weight in meta_direction_weights:
    #                         print("Weights: forth:" + str(forth_weight * meta_weight) + ",Sidewalk:" + str(sidewalk_weight) +
    #                               ",Obstacle:" + str(obstacle_weight) +",Litter:" + str(litter_weight) +
    #                               ",stuck:" + str(stuck_weight * meta_weight))
    #
    #                         print("Run " + str(run_num) + " of " + str(total_run_num))
    #                         if (grid_world == None):
    #                             grid_world_to_use = randomlyGenerateGridWorld()
    #                         else:
    #                             grid_world_to_use = grid_world.copy()
    #                         run_results = runTrajectory(grid_world_to_use, max_steps, forth_q_vals, litter_q_vals,
    #                                                     obstacle_q_vals, sidewalk_q_vals, stuck_q_vals, forth_weight * meta_weight,
    #                                                     litter_weight, obstacle_weight, sidewalk_weight, stuck_weight * meta_weight)
    #                         results_data.append((run_results, litter_weight, sidewalk_weight, obstacle_weight, forth_weight * meta_weight,
    #                                              stuck_weight * meta_weight))
    #                         run_num += 1
    #
    #             joblib.dump(results_data, run_results_file_name)

def trainAllModules(max_steps, grid_world=None):
    """
    Train all of the modules (learn their q values).

    :param max_steps:   Maximum number of steps per episode in training.
    :param grid_world:  Grid world to use. If none, a grid world should be randomly generated for each episode.

    :return: Tuple of the q values for the litter module, obstacle module, sidewalk module, forth module, and stuck
    module, in that order.
    """

    print("Training forth module")
    forth_q_vals = trainForthModule(grid_world, max_steps)
    print("Training sidewalk module")
    sidewalk_q_vals = trainSidewalkModule(grid_world, max_steps)
    print("Training litter module")
    litter_q_vals = trainLitterModule(grid_world, max_steps)
    print("Training obstacle module")
    # obstacle_q_vals = trainObstacleModule(grid_world, max_steps)
    obstacle_q_vals = trainLocalObstacleModule(grid_world, max_steps)
    print("Training stuck module")
    stuck_q_vals = trainStuckModule(grid_world, max_steps)

    return (litter_q_vals, obstacle_q_vals, sidewalk_q_vals, forth_q_vals, stuck_q_vals)

def getGridWorld():
    """
    Get the hand crafted grid world.

    :return: Hand crafted grid world.
    """
    bounds_x = 21
    bounds_y = 24
    sidewalk_min_x = 5
    sidewalk_max_x = 15

    litter_positions = [(0, 10), (0, 17), (0, 21), (2, 5), (2, 13), (2, 20), (3, 1), (3, 8), (4, 11), (4, 14), (4, 21),
                        (5, 20), (6, 3), (6, 8), (6, 12), (7, 21), (8, 6), (8, 16), (9, 11), (10, 3), (10, 7),
                        (10, 19), (11, 21), (12, 3), (12, 10), (12, 15), (12, 20), (13, 6), (14, 2), (14, 21),
                        (15, 11), (16, 14), (16, 16), (16, 19), (17, 2), (17, 4), (17, 9), (17, 21), (18, 7), (18, 14),
                        (18, 17), (19, 11), (19, 20), (20, 5)]

    litter_positions = [Position(x, y) for (x, y) in litter_positions]

    obstacle_positions = [(0, 4), (2, 15), (3, 7), (3, 18), (4, 4), (4, 10), (5, 5), (5, 18), (6, 11), (6, 23), (7, 8),
                          (7, 16), (9, 9), (9, 12), (9, 22), (10, 8), (10, 16), (11, 7), (12, 6), (12, 11), (12, 14),
                          (12, 23), (13, 12), (13, 19), (15, 9), (16, 3), (16, 7), (16, 12), (17, 15), (18, 22),
                          (19, 10), (19, 19)]
    obstacle_positions = [Position(x, y) for (x, y) in obstacle_positions]

    grid_rep = GridRepresentation(litter_positions, obstacle_positions, sidewalk_min_x, sidewalk_max_x, bounds_x, bounds_y)
    agent_start = Position(10, 0)

    grid_world = GridWorld(grid_rep, agent_start)
    return grid_world

def randomlyGenerateGridWorld():
    """
    Randomly generate a grid world. X bounds will be in range [11,30], y bounds will be in range [10,30]. Agent will
    be placed randomly in a cell with y=0. Obstacles will cover between 5% and 20% of the board and litter will cover
    between 5% and 50% of the board. The sidewalk is 11 cells wide (x dimension) and will be randomly placed.

    :return: Randomly generated grid world.
    """
    min_bounds_x = 11
    max_bounds_x = 30
    min_bounds_y = 10
    max_bounds_y = 30

    bounds_x = random.randint(min_bounds_x, max_bounds_x)
    bounds_y = random.randint(min_bounds_y, max_bounds_y)

    min_obstacles = bounds_x * bounds_y * 0.05
    max_obstacles = bounds_x * bounds_y * 0.2
    num_obstacles = random.randint(math.floor(min_obstacles), math.floor(max_obstacles))

    min_litter = bounds_x * bounds_y * 0.05
    max_litter = bounds_x * bounds_y * 0.5
    num_litter = random.randint(math.floor(min_litter), math.floor(max_litter))

    agent_start_x = random.randint(0, bounds_x - 1)

    obstacles_by_y = {obs_y:[] for obs_y in range(bounds_y)}
    num_obstacles_added = 0
    attempts = 0

    while (num_obstacles > num_obstacles_added):
        obs_y = random.randint(0, bounds_y - 1)
        if (len(obstacles_by_y[obs_y]) >= (bounds_x / 2)):
            attempts += 1
            continue
        obs_x = random.randint(0, bounds_x - 1)
        if (obs_y == 0) and (obs_x == agent_start_x):
            attempts += 1
            continue
        if (obs_x in obstacles_by_y[obs_y]):
            attempts += 1
            continue
        obstacles_by_y[obs_y].append(obs_x)
        num_obstacles_added += 1

    obstacle_positions = []
    for obs_y in obstacles_by_y:
        for obs_x in obstacles_by_y[obs_y]:
            obstacle_positions.append(Position(obs_x, obs_y))

    num_litter_added = 0
    litter_by_y = {litter_y:[] for litter_y in range(bounds_y)}
    while (num_litter != num_litter_added):
        litter_y = random.randint(0, bounds_y - 1)
        litter_x = random.randint(0, bounds_x - 1)
        if (litter_y == 0) and (litter_x == agent_start_x):
            continue
        if (litter_x in obstacles_by_y[litter_y]):
            continue
        if (litter_x in litter_by_y[litter_y]):
            continue
        litter_by_y[litter_y].append(litter_x)
        num_litter_added += 1

    min_sidewalk = random.randint(0, bounds_x - 11)
    max_sidewalk = min_sidewalk + 10

    litter_positions = []
    for litter_y in litter_by_y:
        for litter_x in litter_by_y[litter_y]:
            litter_positions.append(Position(litter_x, litter_y))

    grid_rep = GridRepresentation(litter_positions, obstacle_positions, min_sidewalk, max_sidewalk, bounds_x, bounds_y)
    grid_world = GridWorld(grid_rep, Position(agent_start_x, 0))
    return grid_world

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_max_steps = 1000
    test_max_steps = 150

    for i in range(3):
        plotGridWorld(randomlyGenerateGridWorld())

    (litter_q_vals, obstacle_q_vals, sidewalk_q_vals, forth_q_vals, stuck_q_vals) = trainAllModules(train_max_steps)
    # (litter_q_vals, obstacle_q_vals, sidewalk_q_vals, forth_q_vals, stuck_q_vals) = getQResultsFromFiles()

    # Test on the same grid world so it is easier to compare results
    grid_world = getGridWorld()
    runTrajectoryWithWeightEval(grid_world, test_max_steps, forth_q_vals, litter_q_vals, obstacle_q_vals, sidewalk_q_vals, stuck_q_vals)

