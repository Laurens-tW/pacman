# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='HybridReflexAgent1', second='HybridReflexAgent2', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class HybridReflexAgent1(ReflexCaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.threshold = 18
        self.food = 0
        self.maxfood = 2
        self.is_defensive = True
        self.last_food_count = None


    def pacman_on_own_side(self,game_state):
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_position(self.index)
        if self.red:
            return my_pos[0] < mid_x
        else:
            return my_pos[0] > mid_x


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        #Food from current state and the next
        curr_food_list = self.get_food(game_state).as_list()
        food_list = self.get_food(successor).as_list()

        my_state = game_state.get_agent_state(self.index)
        my_scared_timer = my_state.scared_timer

        #initialize last_food_count
        if self.last_food_count is None:
            self.last_food_count = len(curr_food_list)

        #if agent ate a pellet, add to own food count
        food_eaten = self.last_food_count - len(curr_food_list)
        if food_eaten > 0:
            self.food += food_eaten

        #update global food count
        self.last_food_count = len(curr_food_list)



        # if self.is_defensive:
        #     print("agent 2 is defensive")
        # else:
        #     print("agent 2 is offensive")



        # defensive features
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()


        if self.pacman_on_own_side(game_state):
            self.food = 0

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            if min(dists) < 10:
                features['flee'] = 1

        #computes distance to start position
        disttostart = self.get_maze_distance(my_pos, self.start)
        if my_state.is_pacman:
            features['distance_to_start'] = disttostart
        else:
            features['distance_to_start'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        #offensive features

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        opponents = self.get_opponents(game_state)
        max_scared_time = 0
        closest_scared_ghost_dist = float('inf')
        for opponent in opponents:
            ghost_state = game_state.get_agent_state(opponent)
            if not ghost_state.is_pacman:  # spook
                scared_timer = ghost_state.scared_timer
                max_scared_time = max(max_scared_time, scared_timer)
                if scared_timer > 5:
                    ghost_pos = ghost_state.get_position()
                    if ghost_pos:
                        dist = self.get_maze_distance(my_pos, ghost_pos)
                        closest_scared_ghost_dist = min(closest_scared_ghost_dist, dist)
                        features['distance_to_scared_ghost'] = closest_scared_ghost_dist
                else:
                    features['distance_to_scared_ghost'] = 0
                    defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

                    if len(defenders) > 0:
                        dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
                        mindist = min(dists)
                        if mindist < 3:
                            features['ghost_really_close'] = 3 - mindist
                            features['getoutofthere'] = 1
                        elif mindist < 8:
                            features['getoutofthere'] = 1
                            features['ghost_close'] = mindist
                        else:
                            features['ghost_far'] = mindist

        features['scared_ghost_time'] = max_scared_time

        #agent acts defensive when: -total game score is high enough,
        #                           -it is holding enough pellets (to bring them home)
        #it turns offensive when the enemy eats a power pellet
        if (self.get_score(game_state) >= self.threshold or self.food > self.maxfood) and my_scared_timer == 0:
            self.is_defensive = True
        elif self.pacman_on_own_side(game_state) and len(invaders) > 0:
            self.is_defensive = True
        else:
            self.is_defensive = False


        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        my_scared_timer = my_state.scared_timer
        if self.is_defensive and my_state.is_pacman:
            #defensive weights
            return {#'num_invaders': -1000,
                    'on_defense': 100,
                    'invader_distance': -500,
                    'stop': -200,
                    'reverse': -1,
                    'distance_to_start': -30,
                    'ghost_really_close': -500,
                    'ghost_close': 100,
                    'getoutofthere': -100,
                    'distance_enemy_most_food': -100
                    }
        elif self.is_defensive and not my_state.is_pacman:
            if my_scared_timer > 0:
                return {'flee': -500,
                        'stop': -200,
                        'reverse': -1
                        }
            else:
                return {
                    'num_invaders': -5000,
                    'on_defense': 100,
                    'invader_distance': -500,
                    'stop': -200,
                    'reverse': -1,
                    'distance_enemy_most_food': -50
                    #'distance_to_start': -30,
                    #'ghost_really_close': -100,
                    #'ghost_close': 10
                    #'getoutofthere': -100
                }
        elif not self.is_defensive and not my_state.is_pacman:
            #offensive weights
            return {'successor_score': 100,
                    'distance_to_food': -2,
                    'ghost_really_close': -200,
                    'ghost_close': 50,
                    'stop': -200,
                    'ghost_far': 20,
                    'getoutofthere': -100,
                    'distance_to_scared_ghost': -500
                    }
        else:
            return {'successor_score': 100,
                    'distance_to_food': -2,
                    'ghost_really_close': -2000,
                    'ghost_close': 200,
                    'stop': -200,
                    'ghost_far': 500,
                    'getoutofthere': -100,
                    'distance_to_scared_ghost': -500
                    }


class HybridReflexAgent2(ReflexCaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.threshold = 10
        self.food = 0
        self.maxfood = 3
        self.is_defensive = True
        self.last_food_count = None


    def pacman_on_own_side(self,game_state):
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_position(self.index)
        if self.red:
            return my_pos[0] < mid_x
        else:
            return my_pos[0] > mid_x


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        #Food from current state and the next
        curr_food_list = self.get_food(game_state).as_list()
        food_list = self.get_food(successor).as_list()

        my_state = game_state.get_agent_state(self.index)
        my_scared_timer = my_state.scared_timer

        #initialize last_food_count
        if self.last_food_count is None:
            self.last_food_count = len(curr_food_list)

        #if agent ate a pellet, add to own food count
        food_eaten = self.last_food_count - len(curr_food_list)
        if food_eaten > 0:
            self.food += food_eaten

        #update global food count
        self.last_food_count = len(curr_food_list)



        # if self.is_defensive:
        #     print("agent 2 is defensive")
        # else:
        #     print("agent 2 is offensive")



        # defensive features
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()


        if self.pacman_on_own_side(game_state):
            self.food = 0

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            if min(dists) < 10:
                features['flee'] = 1

        #computes distance to start position
        disttostart = self.get_maze_distance(my_pos, self.start)
        if my_state.is_pacman:
            features['distance_to_start'] = disttostart
        else:
            features['distance_to_start'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        #offensive features

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        opponents = self.get_opponents(game_state)
        max_scared_time = 0
        closest_scared_ghost_dist = float('inf')
        for opponent in opponents:
            ghost_state = game_state.get_agent_state(opponent)
            if not ghost_state.is_pacman:  # spook
                scared_timer = ghost_state.scared_timer
                max_scared_time = max(max_scared_time, scared_timer)
                if scared_timer > 5:
                    ghost_pos = ghost_state.get_position()
                    if ghost_pos:
                        dist = self.get_maze_distance(my_pos, ghost_pos)
                        closest_scared_ghost_dist = min(closest_scared_ghost_dist, dist)
                        features['distance_to_scared_ghost'] = closest_scared_ghost_dist
                else:
                    features['distance_to_scared_ghost'] = 0
                    defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

                    if len(defenders) > 0:
                        dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
                        mindist = min(dists)
                        if mindist < 3:
                            features['ghost_really_close'] = 3 - mindist
                            features['getoutofthere'] = 1
                        elif mindist < 8:
                            features['ghost_close'] = mindist
                            features['getoutofthere'] = 1
                        else:
                            features['ghost_far'] = mindist

        features['scared_ghost_time'] = max_scared_time

        #agent acts defensive when: -total game score is high enough,
        #                           -it is holding enough pellets (to bring them home)
        #it turns offensive when the enemy eats a power pellet
        if (self.get_score(game_state) >= self.threshold or self.food > self.maxfood) and my_scared_timer == 0:
            self.is_defensive = True
        elif self.pacman_on_own_side(game_state) and len(invaders) > 0:
            self.is_defensive = True
        else:
            self.is_defensive = False



        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        my_scared_timer = my_state.scared_timer
        if self.is_defensive and my_state.is_pacman:
            #defensive weights
            return {#'num_invaders': -1000,
                    'on_defense': 100,
                    'invader_distance': -500,
                    'stop': -200,
                    'reverse': -1,
                    'distance_to_start': -30,
                    'ghost_really_close': -500,
                    'ghost_close': 100,
                    'getoutofthere': -100,
                    'distance_enemy_most_food': -100
                    }
        elif self.is_defensive and not my_state.is_pacman:
            if my_scared_timer > 0:
                return {'flee': -500,
                        'stop': -200,
                        'reverse': -1
                        }
            else:
                return {
                    'num_invaders': -5000,
                    'on_defense': 100,
                    'invader_distance': -500,
                    'stop': -200,
                    'reverse': -1,
                    'distance_enemy_most_food': -50
                    #'distance_to_start': -30,
                    #'ghost_really_close': -100,
                    #'ghost_close': 10
                    #'getoutofthere': -100
                }
        elif not self.is_defensive and not my_state.is_pacman:
            #offensive weights
            return {'successor_score': 100,
                    'distance_to_food': -2,
                    'ghost_really_close': -200,
                    'ghost_close': 50,
                    'stop': -200,
                    'ghost_far': 20,
                    'getoutofthere': -100,
                    'distance_to_scared_ghost': -500
                    }
        else:
            return {'successor_score': 100,
                    'distance_to_food': -2,
                    'ghost_really_close': -2000,
                    'ghost_close': 200,
                    'stop': -200,
                    'ghost_far': 500,
                    'getoutofthere': -100,
                    'distance_to_scared_ghost': -500
                    }
