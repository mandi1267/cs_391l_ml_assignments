"""
This file contains a function for plotting the grid world and several other pieces of data related to a run, if
provided.
"""
from grid_world import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.patches as mpatches

def plotGridWorld(grid_world, timestep=None, show_duration=None, display_action=None, custom_title=None,
                  trajectory=None, q_values_for_actions=None):
    """
    Plot the grid world with the given options.

    :param grid_world:              Grid world containing litter, obstacle, sidewalk, and agent locations. If the
                                    trajectory is provided, the initial litter configuration will be used. If the
                                    trajectory is not provided, the final litter state will be used.
    :param timestep:                Timestep. Used in title. If none, no timestep will be displayed.
    :param show_duration:           Amount of time to show plot. If none, plot will be displayed until it is closed.
    :param display_action:          Numeric action to be displayed. If none, no action will be displayed.
    :param custom_title:            Custom title to be displayed. If none, only the timestep/display action (if
                                    provided) will be displayed in the title. If provided, it will be added to a second
                                    line of the title.
    :param trajectory:              Trajectory of the agent represented as a sequence of its positions (Position
                                    objects) on the board. If none, don't display the trajectory, just the current state.
    :param q_values_for_actions:    Q values for the possible actions that the agent can take in its current state.
                                    If provided, will display 4 numbers around the current agent position. If not
                                    provided, no q values will be plotted.
    """

    x_bounds = grid_world.current_obstacle_litter_map.bounds_x
    y_bounds = grid_world.current_obstacle_litter_map.bounds_y

    litter_num = grid_world.current_obstacle_litter_map.litter_num
    obstacle_num = grid_world.current_obstacle_litter_map.obstacle_num
    no_content = grid_world.current_obstacle_litter_map.no_content
    agent_num = 3
    sidewalk_num = 4

    grid_rep_to_plot = grid_world.current_obstacle_litter_map
    if (trajectory != None):
        grid_rep_to_plot = grid_world.init_obstacle_litter_map

    # Determine what color to make each cell
    data = np.zeros((x_bounds, y_bounds))
    for x_val in range(x_bounds):
        for y_val in range(y_bounds):
            point_pos = Position(x_val, y_val)
            cell_val = grid_rep_to_plot.getEntryAtCell(point_pos)
            if ((grid_world.agent_position.x == x_val) and (grid_world.agent_position.y == y_val)):
                data[x_val, y_val] = agent_num
            elif (cell_val != no_content):
                data[x_val, y_val] = cell_val
            elif (grid_world.onSidewalk(point_pos)):
                data[x_val, y_val] = sidewalk_num
            else:
                data[x_val, y_val] = no_content

    color_map_list = ['white', 'yellow', 'purple', 'green', 'gray']
    color_map = pltcolors.ListedColormap(color_map_list)
    data = np.transpose(data)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=color_map)

    # Plot the trajectory if provided (lines from prev cell to the next cell for each action)
    if (trajectory != None):
        positions = trajectory[0]
        actions = trajectory[1]
        for i in range(len(actions)):
            start_pos = (positions[i].x, positions[i].y)
            action = actions[i]
            dx = 0
            dy = 0
            if (action == 1):
                dy = -1
            elif (action == 2):
                dy = 1
            elif (action == 3):
                dx = -1
            elif (action == 4):
                dx = 1
            else:
                print("Invalid action " + str(action))
                continue

            ax.arrow(start_pos[0], start_pos[1], dx, dy)

    ax.set_xticks(np.arange(x_bounds + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(y_bounds + 1) - 0.5, minor=True)
    ax.grid(True, which="minor")

    # Add the legend
    no_content_patch = mpatches.Patch(color=color_map_list[no_content], label='Empty')
    obstacle_patch = mpatches.Patch(color=color_map_list[obstacle_num], label='Obstacle')
    litter_patch = mpatches.Patch(color=color_map_list[litter_num], label='Litter')
    sidewalk_patch = mpatches.Patch(color=color_map_list[sidewalk_num], label='Sidewalk')
    agent_path = mpatches.Patch(color=color_map_list[agent_num], label='Current Agent Position')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., handles=[no_content_patch, obstacle_patch, litter_patch, sidewalk_patch, agent_path])

    # Add the q values to the plot if provided
    if (q_values_for_actions != None):
        agent_pos = grid_world.agent_position

        ax.annotate("{:.4f}".format(q_values_for_actions[1]), xy=(agent_pos.x, agent_pos.y - 1), ha='center')
        ax.annotate("{:.4f}".format(q_values_for_actions[2]), xy=(agent_pos.x, agent_pos.y + 1), ha='center')
        ax.annotate("{:.4f}".format(q_values_for_actions[3]), xy=(agent_pos.x - 2, agent_pos.y), ha='center')
        ax.annotate("{:.4f}".format(q_values_for_actions[4]), xy=(agent_pos.x + 2, agent_pos.y), ha='center')

    # Create the title
    title_str = ""
    if(timestep != None):
        title_str += "Grid at t=" + str(timestep)
        if (display_action != None):
            title_str += ", Action="
            title_str += str(display_action)

    if custom_title != None:
        if title_str != "":
            title_str += "\n"
        title_str += custom_title

    if (title_str != ""):
        plt.title(title_str)
    if (show_duration == None):
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(show_duration)
        plt.close()