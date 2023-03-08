# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # heavily prioritize winning and avoid losing 
        if (successorGameState.isWin()):
            return 99999 
        elif (successorGameState.isLose()):
            return -99999 

        score = 0 

        # prioritize closer food while also reducing the number of food remaining 
        newFood = newFood.asList() 
        closestFood = max([1/(manhattanDistance(newPos, foodPos)) for foodPos in newFood]) 
        score += closestFood 
        score -= len(newFood) 
        
        # run away if ghosts are close 
        for ghostState in newGhostStates: 
            if manhattanDistance(ghostState.getPosition(), newPos) < 3:
                score -= 100 
        
        # disincentivize staying in place 
        if (action == Directions.STOP): score -= 100 
        
        # disicentivize stalling 
        score += successorGameState.getScore() 

        return score 

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # get legal moves 
        legalMoves = gameState.getLegalActions(0) 

        # score each action 
        scores = self.getValue(gameState, 0, 0) 
        bestScore = max(scores) 
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def getValue(self, gameState: GameState, depth: int, agentIndex: int): 
        if (depth >= self.depth or gameState.isWin() or gameState.isLose()): 
            # terminal node at depth limit or game end 
            return self.evaluationFunction(gameState) 
        elif agentIndex == 0: 
            # pacman 
            return self.getMaxValue(gameState, depth) 
        else:
            # ghost 
            return self.getMinValue(gameState, depth, agentIndex) 

    def getMaxValue(self, gameState: GameState, depth: int): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(0) 

        if depth == 0: 
            # at root 
            scores = [] 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(0, action) 
                score = self.getValue(successor, depth, 1) 
                scores.append(score) 
            return scores 
        else:
            scores = [self.getValue(gameState.generateSuccessor(0, action), depth, 1) for action in legalMoves] 
            bestScore = max(scores) 
            return bestScore 

    def getMinValue(self, gameState: GameState, depth:int, agentIndex: int): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(agentIndex) 

        if agentIndex == gameState.getNumAgents() - 1: 
            # last ghost to move this depth 
            scores = [self.getValue(gameState.generateSuccessor(agentIndex, action), depth+1, 0) for action in legalMoves] 
            bestScore = min(scores) 
            return bestScore 
        else:
            # middle ghosts, next to move is ghost 
            scores = [self.getValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1) for action in legalMoves] 
            bestScore = min(scores) 
            return bestScore 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState): 
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        bestAction, _ = self.getAlphaBetaValue(gameState, 0, 0, -float('inf'), float('inf')) 
        return bestAction 

    def getAlphaBetaValue(self, gameState: GameState, depth: int, agentIndex: int, alpha, beta): 
        # terminal node at depth limit or game end 
        if (depth >= self.depth or gameState.isWin() or gameState.isLose()): 
            return self.evaluationFunction(gameState) 
        elif agentIndex == 0: 
            # pacman 
            return self.getMaxValue(gameState, depth, alpha, beta) 
        else: 
            # ghost 
            return self.getMinValue(gameState, depth, agentIndex, alpha, beta) 

    def getMaxValue(self, gameState: GameState, depth: int, alpha, beta): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(0) 

        if depth == 0: 
            # at root 
            bestScore = -float('inf') 
            bestAction = None 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(0, action) 
                score = self.getAlphaBetaValue(successor, depth, 1, alpha, beta) 
                if score > bestScore: 
                    bestScore = score 
                    bestAction = action 
                alpha = max(alpha, bestScore) 
            return bestAction, bestScore 
        else: 
            bestScore = -float('inf') 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(0, action) 
                score = self.getAlphaBetaValue(successor, depth, 1, alpha, beta) 
                bestScore = max(bestScore, score) 
                if bestScore > beta: 
                    return bestScore 
                alpha = max(alpha, bestScore) 
            return bestScore 

    def getMinValue(self, gameState: GameState, depth: int, agentIndex: int, alpha, beta): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(agentIndex) 

        if agentIndex == gameState.getNumAgents() - 1: 
            # last ghost 
            bestScore = float('inf') 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(agentIndex, action) 
                score = self.getAlphaBetaValue(successor, depth+1, 0, alpha, beta) 
                bestScore = min(bestScore, score) 
                if bestScore < alpha: 
                    return bestScore 
                beta = min(beta, bestScore) 
            return bestScore 
        else:
            # middle ghosts 
            bestScore = float('inf') 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(agentIndex, action) 
                score = self.getAlphaBetaValue(successor, depth, agentIndex+1, alpha, beta) 
                bestScore = min(bestScore, score) 
                if bestScore < alpha: 
                    return bestScore 
                beta = min(beta, bestScore) 
            return bestScore 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # get legal moves 
        legalMoves = gameState.getLegalActions(0) 

        # score each action 
        scores = self.getValue(gameState, 0, 0)  
        bestScore = max(scores) 
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def getValue(self, gameState: GameState, depth: int, agentIndex: int): 
        if (depth >= self.depth or gameState.isWin() or gameState.isLose()): 
            # terminal node at depth limit or game end 
            return self.evaluationFunction(gameState) 
        elif agentIndex == 0: 
            # pacman 
            return self.getMaxValue(gameState, depth) 
        else: 
            # ghost 
            return self.getExpectedValue(gameState, depth, agentIndex) 

    def getMaxValue(self, gameState: GameState, depth: int): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(0) 

        if depth == 0: 
            # at root 
            scores = [] 
            for action in legalMoves: 
                successor = gameState.generateSuccessor(0, action) 
                score = self.getValue(successor, depth, 1) 
                scores.append(score) 
            return scores 
        else:
            scores = [self.getValue(gameState.generateSuccessor(0, action), depth, 1) for action in legalMoves] 
            bestScore = max(scores) 
            return bestScore 

    def getExpectedValue(self, gameState: GameState, depth:int, agentIndex: int): 
        # get all legal moves 
        legalMoves = gameState.getLegalActions(agentIndex) 

        if agentIndex == gameState.getNumAgents() - 1: 
            # last ghost to move this depth 
            scores = [self.getValue(gameState.generateSuccessor(agentIndex, action), depth+1, 0) for action in legalMoves] 
            expectedScore = sum(scores) * 1/len(scores) 
            return expectedScore 
        else:
            # middle ghosts, next to move is ghost 
            scores = [self.getValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1) for action in legalMoves] 
            expectedScore = sum(scores) * 1/len(scores) 
            return expectedScore 

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    score = 0 

    # prioritize closer food while also reducing the number of food remaining 
    currentFood = currentFood.asList() 
    if currentFood: 
        closestFood = max([1/(manhattanDistance(currentPos, foodPos)) for foodPos in currentFood]) 
    else:
        closestFood = 0 
    score += closestFood 
    score -= len(currentFood) * 10 
        
    # run away if ghosts are close 
    for i in range(len(currentGhostStates)): 
        ghostState = currentGhostStates[i] 
        ghostScaredTime = currentScaredTimes[i] 
        ghostDistance = manhattanDistance(ghostState.getPosition(), currentPos)

        if ghostScaredTime == 0 and ghostDistance < 3:
            score -= 100 

    return score 

# Abbreviation
better = betterEvaluationFunction

