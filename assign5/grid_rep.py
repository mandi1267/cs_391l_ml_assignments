"""
This defines the grid representation (grid world without agent positions).
"""

import copy

class Position(object):
    """
    Object for representing the position.
    """
    def __init__(self, x, y):
        """
        Create the position.

        :param x: X component of the position.
        :param y: Y component of the position.
        """
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class GridRepresentation(object):
    """
    Object for representing the grid state (excluding agent information).
    """

    def __init__(self, litter_positions, obstacle_positions, sidewalk_min_x, sidewalk_max_x, bounds_x, bounds_y):
        """
        Create the grid representation (grid world except agent state).
        Litter and obstacles are assumed not to be colocated.

        :param litter_positions:    List of litter positions.
        :param obstacle_positions:  List of obstacle positions.
        :param sidewalk_min_x:      Minimum x position of the sidewalk.
        :param sidewalk_max_x:      Maximum x position of the sidewalk.
        :param bounds_x:            X bounds of the sidewalk.
        :param bounds_y:            Y bounds of the sidewalk.
        """

        # Assumes no litter and obstacles colocated
        # min x and max x are inclusive

        self.litter_positions = litter_positions
        self.obstacle_positions = obstacle_positions
        self.sidewalk_min_x = sidewalk_min_x
        self.sidewalk_max_x = sidewalk_max_x
        self.litter_num = 1
        self.obstacle_num = 2
        self.no_content = 0

        # First index gives x
        # Second index gives y
        empty_row = [self.no_content for i in range(bounds_y)]
        empty_mat = [empty_row[:] for i in range(bounds_x)]

        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

        self.sidewalk_center_x = ((sidewalk_max_x - sidewalk_min_x) / 2.0) + sidewalk_min_x
        self.litter_obs_grid = empty_mat
        for litter_pos in litter_positions:
            if (litter_pos.x < self.bounds_x) and (litter_pos.y < self.bounds_y):
                self.litter_obs_grid[litter_pos.x][litter_pos.y] = self.litter_num
            else:
                print("litter not in board range")

        for obstacle_pos in obstacle_positions:
            if (obstacle_pos.x < self.bounds_x) and (obstacle_pos.y < self.bounds_y):
                if (self.litter_obs_grid[obstacle_pos.x][obstacle_pos.y] != self.litter_num):
                    self.litter_obs_grid[obstacle_pos.x][obstacle_pos.y] = self.obstacle_num
                else:
                    print("Tried to place litter and obstacle at " + str(obstacle_pos.x) + ", " + str(obstacle_pos.y))
            else:
                print("obstacle not in board range")

    def copy(self):
        """
        Deep copy the grid representation.

        :return: copy of the grid representation.
        """
        return copy.deepcopy(self)

    def getEntryAtCell(self, cell_pos):
        """
        Get the entry at the given cell position.

        :param cell_pos: Cell position to get contents of.

        :return: Value of the given cell (0 = nothing, 1 = litter, 2 = obstacle). Does not contain sidewalk state.
        """
        return self.litter_obs_grid[cell_pos.x][cell_pos.y]

    def visitCell(self, cell_pos):
        """
        Visit the cell (removing litter if there is litter in the cell).

        :param cell_pos: Cell position to visit.

        :return: Value of the given cell (0 = nothing, 1 = litter, 2 = obstacle). Does not contain sidewalk state.
        """
        entry_val = self.getEntryAtCell(cell_pos)
        if (entry_val == self.litter_num):
            self.litter_obs_grid[cell_pos.x][cell_pos.y] = self.no_content
        return entry_val

    def posOffBoard(self, pos):
        """
        Return true if the query position is off the board, false if the position is on the board.

        :param pos: Position to check.

        :return: True if the position is off the board, false if it is on the board.
        """
        if (pos.x < 0):
            return True
        if (pos.x >= self.bounds_x):
            return True
        if (pos.y < 0):
            return True
        if (pos.y >= self.bounds_y):
            return True
        return False

    def getPosOfNearestObjOfType(self, obj_type, query_pos):
        """
        Get the position of the nearest object of the given type (1 for litter, 2 for obstacle).

        :param obj_type:    Object type (1 for litter, 2 for obstacle).
        :param query_pos:   Position to get the nearest object for.

        :return: Relative position of the nearest obstacle of the type. Will be larger than the bounds of the board if
         no object exists.
        """
        if ((obj_type == self.litter_num) or (obj_type == self.obstacle_num)):
            closest_position = Position(self.bounds_x + 1, self.bounds_y + 1)
            closest_dist = float("inf")
            for x_val in range(self.bounds_x):
                for y_val in range(self.bounds_y):
                    if (self.litter_obs_grid[x_val][y_val] == obj_type):
                        pos_dist = (abs(x_val - query_pos.x) + abs(y_val - query_pos.y))
                        if (closest_dist > pos_dist):
                            closest_dist = pos_dist
                            closest_position = Position(x_val, y_val)

            return closest_position
        else:
            print("Invalid object type " + str(obj_type))
            return Position(None, None)

    def onSidewalk(self, agent_pos):
        """
        Get if the agent is on the sidewalk.

        :param agent_pos: Agent position to check.

        :return: True if the agent is on the sidewalk, false if it is off the sidewalk.
        """
        if ((agent_pos.x >= self.sidewalk_min_x) and (agent_pos.x <= self.sidewalk_max_x)):
            return True
        return False