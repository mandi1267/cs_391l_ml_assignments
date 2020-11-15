"""
File containing objects to retrieve the states for each module.
"""

from grid_world import *
from states import *

class StateRetriever(object):
    """
    Interface for retrieving states.
    """

    def __init__(self):
        pass

    def getState(self, grid_world):
        """
        Get the state from the grid world.

        :param grid_world: Grid world to retreive the state from.

        :return: State from the grid world.
        """
        pass

class LitterStateRetriever(StateRetriever):
    """
    Object for retrieving the litter state.
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getLitterState()

class ObstacleStateRetriever(StateRetriever):
    """
    Object for retrieving the obstacle state (distance to nearest obstacle).
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getObstacleState()

class SidewalkStateRetriever(StateRetriever):
    """
    Object for retrieving the sidewalk state.
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getSidewalkState()

class ForthStateRetriever(StateRetriever):
    """
    Object for retrieving the forth state.
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getForthState()

class StuckStateRetriever(StateRetriever):
    """
    Object for retrieving the stuck state.
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getStuckState()

class LocalObstacleStateRetreiver(StateRetriever):
    """
    Object for retrieving the local obstacle state (configuration of obstacles immediately around the agent).
    """
    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return grid_world.getLocalObstacleState()

class FullStateRetriever(StateRetriever):
    """
    Object for retrieving the full state (for all modules).
    """

    def __init__(self):
        super(StateRetriever, self).__init__()

    def getState(self, grid_world):
        return CombinedState(grid_world.getSidewalkState(), grid_world.getLitterState(), grid_world.getLocalObstacleState(), #grid_world.getObstacleState(),
                             grid_world.getForthState(), grid_world.getStuckState())
