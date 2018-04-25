import numpy as np
import abc
import util
from game import Agent, Action
import copy
from game_state import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board

        return np.sum(getChainedValueTable(board))


def getChainedValueTable(board):
    t = np.zeros(board.shape)
    table = np.zeros(board.shape)
    raw,col = board.shape
    for x in range(raw):
        for y in range(col):
            table[x,y] = getChainedValue(x,y,board,t)
    return table



def getChainedValue (x, y , board, table):
    if table[x,y]>0:
        return table[x,y]

    tileValue = board[x,y]
    value = 0
    adjacentTiles = getAdjacentTiles(x,y,board)
    if tileValue != 0:
        for adjTileValue,adjTileCoor in adjacentTiles:
            value2 = 0
            if tileValue == adjTileValue*2:
                value2 = getChainedValue(adjTileCoor[0],adjTileCoor[1],board,table)
            elif tileValue == adjTileValue:
                value2 = adjTileValue*2
            elif tileValue == adjTileValue*4:
                value2 = adjTileValue
            if value2>value:
                value = value2
        value += tileValue
    table[x,y] = value
    return value


def getAdjacentTiles(x,y,board):
    deltas = np.array([[0,-1],[0,1],[-1,0],[1,0]])
    return [(board[tuple(np.array([x,y])+delta)],tuple(np.array([x,y])+delta)) for delta in deltas if
             np.all(np.array([x,y])+delta<[4,4]) and
             np.all(np.array([x,y])+delta>=[0,0])]


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.get_score(game_state, action) for action in legal_moves]
        best_score = max(scores)
        print(best_score)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
        return legal_moves[chosen_index]


    def get_score(self,current_game_state,action):
        successor_game_state = current_game_state.generate_successor(action=action)
        ret = self.minMax(successor_game_state, 0,False)
        return ret


    def minMax(self,state,depth,isMax):
        if depth == self.depth:
            ret = self.evaluation_function(state)
            #print(ret)
            return ret

        if isMax:
            legal_moves = state.get_agent_legal_actions()
            if len(legal_moves) == 0:
                return self.evaluation_function(state)
            successors = [state.generate_successor(action=action) for action in legal_moves]
            return max([self.minMax(succ,depth+1,False) for succ in successors])
        else:
            actions = state.get_opponent_legal_actions()
            if len(actions) == 0:
                return self.evaluation_function(state)
            successors = []
            for act in actions:
                newSucc = copy.deepcopy(state)
                #newSucc = GameState(state.board.shape[0],state.board.shape[1],np.copy(state.board),state.score,state.done)
                newSucc.apply_opponent_action(act)
                successors.append(newSucc)
            return min([self.minMax(succ, depth + 1, True) for succ in successors])



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()





def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
