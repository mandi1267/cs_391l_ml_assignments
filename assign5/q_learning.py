"""
This contains functions and classes related to Q-learning.
"""
from grid_world import *
import random
from plotting import *

class QTable(object):
    """
    Contains the Q-values for each state-action pair.
    """

    def __init__(self, action_options):
        """
        Create a new q table (value for each state-action pair is initially 0).

        :param action_options: Options for actions to take.
        """
        # Contains dictionary mapping state to another dictionary of action to q value
        self.state_to_action_dict = {}
        self.action_options = action_options

    def getQValueForStateActionPair(self, state, action):
        """
        Get the stored q value for the given state and action.

        :param state:   State to get q value for.
        :param action:  Action to get q value for.

        :return: Q value for the given state-action pair
        """
        if (state in self.state_to_action_dict):
            action_dict_for_state = self.state_to_action_dict[state]
            if (action in action_dict_for_state):
                return action_dict_for_state[action]
        return 0.0

    def updateQValue(self, discount_factor, learning_rate, reward, state, action, new_state):
        """
        Update the q value for the given state-action pair.

        :param discount_factor: Discount factor to use in updating the q value.
        :param learning_rate:   Learning rate to use in updating the q value.
        :param reward:          Reward gained from taking the given action in the given state.
        :param state:           State that the action was taken in and that the q value should be updated for.
        :param action:          Action that was taken.
        :param new_state:       State that the agent is in after the action was taken.
        """
        # Get the q value before
        prev_q_value = self.getQValueForStateActionPair(state, action)
        max_q_value_for_new_state = -1 * float("inf")
        # Get the maximum value for the state that the agent is in after taking the action (iterate over all possible
        # actions)
        for candidate_action in self.action_options:
            q_value_for_new_state = self.getQValueForStateActionPair(new_state, candidate_action)
            if (q_value_for_new_state > max_q_value_for_new_state):
                max_q_value_for_new_state = q_value_for_new_state
        # Compute the new q value and update the q table
        new_q_value = prev_q_value + learning_rate * (reward + (discount_factor * max_q_value_for_new_state) - prev_q_value)
        if (not (state in self.state_to_action_dict)):
            self.state_to_action_dict[state] = {}
        self.state_to_action_dict[state][action] = new_q_value

    def __str__(self):
        total_str = ""
        for state in self.state_to_action_dict:
            state_str = str(state)
            should_print = False
            for action, value in self.state_to_action_dict[state].items():
                if (value != 0.0):
                    should_print = True
            if (should_print):
                total_str += "State="
                total_str += state_str
                total_str += "Actions:"
                total_str += str(self.state_to_action_dict[state])
        return total_str


class TestTimeQTable(object):
    """
    Construct the q table to use in deciding actions when combining multiple modules.
    """

    def __init__(self, forth_module_q_table, litter_module_q_table, obstacle_module_q_table, sidewalk_module_q_table,
                 stuck_module_q_table, forth_weight, litter_weight, obstacle_weight, sidewalk_weight, stuck_weight):
        """
        Create the q table that combines q tables from each module based on the given weights.

        :param forth_module_q_table:    Q table learned for the forth module.
        :param litter_module_q_table:   Q table learned for the litter module.
        :param obstacle_module_q_table: Q table learned for the obstacle module.
        :param sidewalk_module_q_table: Q table learned for the sidewalk module.
        :param stuck_module_q_table:    Q table learned for the forth module.
        :param forth_weight:            Weight to multiply the forth module's q values by.
        :param litter_weight:           Weight to multiply the litter module's q values by.
        :param obstacle_weight:         Weight to multiply the obstacle module's q values by.
        :param sidewalk_weight:         Weight to multiply the sidewalk module's q values by.
        :param stuck_weight:            Weight to multiply the stuck module's q values by.
        """

        self.forth_module_q_table = forth_module_q_table
        self.litter_module_q_table = litter_module_q_table
        self.obstacle_module_q_table = obstacle_module_q_table
        self.sidewalk_module_q_table = sidewalk_module_q_table
        self.stuck_module_q_table = stuck_module_q_table
        self.forth_weight = forth_weight
        self.litter_weight = litter_weight
        self.obstacle_weight = obstacle_weight
        self.sidewalk_weight = sidewalk_weight
        self.stuck_weight = stuck_weight

    def getQValueForStateActionPair(self, state, action):
        """
        Get the q value for the state-action pair. Combines values from all modules based on the weights.

        :param state:   State to get q value for.
        :param action:  Action to get q value for.

        :return: Q value for the state-action pair
        """
        # print("Action " + str(action))

        # The state retriever at test time must construct a meta-state object with these 4 substates
        sidewalk_q = self.sidewalk_module_q_table.getQValueForStateActionPair(state.sidewalk_state, action)
        litter_q = self.litter_module_q_table.getQValueForStateActionPair(state.litter_state, action)
        obstacle_q = self.obstacle_module_q_table.getQValueForStateActionPair(state.obstacle_state, action)
        forth_q = self.forth_module_q_table.getQValueForStateActionPair(state.forth_state, action)
        stuck_q = self.stuck_module_q_table.getQValueForStateActionPair(state.stuck_state, action)

        # print("Sidewalk: " + str(sidewalk_q) + ", " + str(self.sidewalk_weight * sidewalk_q))
        # print("Litter: " + str(litter_q) + ", " + str(self.litter_weight * litter_q))
        # print("Obstacle: " + str(obstacle_q) + ", " + str(self.obstacle_weight * obstacle_q))
        # print("Forth: " + str(forth_q) + ", " + str(self.forth_weight * forth_q))
        # print("Stuck: " + str(stuck_q) + ", " + str(self.stuck_weight * stuck_q))

        return (self.sidewalk_weight * sidewalk_q) + (self.litter_weight * litter_q) + \
               (self.obstacle_weight * obstacle_q) + (self.forth_weight * forth_q) + (self.stuck_weight * stuck_q)

    def updateQValue(self, discount_factor, learning_rate, reward, state, action, new_state):
        # We're not updating this when testing (as opposed to training)
        pass

class RunResults():
    def __init__(self, num_steps, num_obstacles_hit, num_off_sidewalk, num_litters, max_possible_steps):
        """
        Results of a run.

        :param num_steps:           Number of steps to get to the goal.
        :param num_obstacles_hit:   Number of obstacles hit along the way.
        :param num_off_sidewalk:    Number of times the agent was in a position off the sidewalk.
        :param num_litters:         Number of litters picked up.
        :param max_possible_steps:  Maximum possible steps that could've been taken (num_steps will equal this if the
                                    agent never got to the goal).
        """
        self.num_steps = num_steps
        self.num_obstacles_hit = num_obstacles_hit
        self.num_off_sidewalk = num_off_sidewalk
        self.num_litters = num_litters
        self.max_possible_steps = max_possible_steps

    def __str__(self):
        return "RunResults(num_litter:" + str(self.num_litters) + ",obs_hit:" + str(self.num_obstacles_hit) + \
               ",off_sidewalk:" + str(self.num_off_sidewalk) + ",steps/max_steps:" + str(self.num_steps) + "/" \
               + str(self.max_possible_steps) + ")"

def executeRun(grid_world, prev_q_values, reward_evaluator, state_retriever, epsilon, max_steps,
               action_options, discount_factor, learning_rate, training, plot_trajectory=False, custom_plot_title=None,
               plot_full_trajectory=False, plot_at_times=[]):

    """
    Execute one run (training or test) in the grid world. Runs until max steps have been hit or the agent is in a goal
    cell.

    :param grid_world:              Grid world defining state of agent and world.
    :param prev_q_values:           Q values from previous runs.
    :param reward_evaluator:        Object to execute moves and compute rewards.
    :param state_retriever:         Object used to extract the state to use in the q table from the grid world.
    :param epsilon:                 Epsilon value. If this is 0, the agent will always take the best action from the
                                    q table. If this is 1, the agent will always take random actions.
    :param max_steps:               Maximum number of steps that can be taken before a run is terminated.
    :param action_options:          Possible actions.
    :param discount_factor:         Discount factor for updating the q value.
    :param learning_rate:           Learning rate for updating the q value.
    :param training:                True if this is a training run (for learning q values), false if it is a testing
                                    run (for evaluating performance of q value).
    :param plot_trajectory:         True if the agent's position and the world state should be plotted at each timestep.
    :param custom_plot_title:       Custom title for the plot.
    :param plot_full_trajectory:    True if the full trajectory should be plotted (agent start to current agent
                                    position).
    :param plot_at_times:           Timesteps for which we should plot the q values with.

    :return: Results of the run.
    """
    total_steps = 0
    new_q_vals = prev_q_values
    actions = []
    prev_action = None

    # Plot the initial state of the grid world
    if (plot_trajectory):
        plotGridWorld(grid_world, total_steps, show_duration=0.1, display_action=prev_action,
                      custom_title=custom_plot_title)

    while ((not grid_world.atGoal()) and (total_steps < max_steps)):

        world_state = state_retriever.getState(grid_world)
        if ((training) and (random.uniform(0, 1) < epsilon)):
            # Choose an action at random
            action = random.choice(action_options)
            q_val_for_actions = {}
        else:

            # Take the action with the best q value
            q_val_for_actions = {}
            max_action_val = -1 * float("inf")
            max_action = -1
            for candidate_action in action_options:
                q_val_for_actions[candidate_action] = new_q_vals.getQValueForStateActionPair(world_state, candidate_action)
                if (q_val_for_actions[candidate_action] > max_action_val):
                    max_action_val = q_val_for_actions[candidate_action]
                    max_action = candidate_action
            action = max_action

        # Plot the given q values
        if (plot_trajectory):
            if (total_steps in plot_at_times):
                plotGridWorld(grid_world, total_steps, show_duration=4, display_action=action,
                              custom_title=custom_plot_title, q_values_for_actions=q_val_for_actions)

        # Compute the reward and next state and update the q value
        reward = reward_evaluator.takeActionAndGetReward(grid_world, action)
        new_state = state_retriever.getState(grid_world)
        new_q_vals.updateQValue(discount_factor, learning_rate, reward, world_state, action, new_state)

        actions.append(action)
        prev_action=action
        total_steps += 1
        if (plot_trajectory):
            if (not (total_steps in plot_at_times)):
                plotGridWorld(grid_world, total_steps, show_duration=0.01, display_action=prev_action,
                              custom_title=custom_plot_title)

    # Plot the full path taken by the agent
    if (plot_trajectory):
        if (plot_full_trajectory):
            plotGridWorld(grid_world, total_steps, show_duration=1, custom_title=custom_plot_title,
                          trajectory=(grid_world.trajectory, actions))
    run_results = RunResults(grid_world.num_steps, grid_world.num_obstacles_hit, grid_world.off_sidewalk_times,
                             grid_world.num_litters_pickedup, max_steps)
    print(run_results)
    return (actions, new_q_vals, run_results)



