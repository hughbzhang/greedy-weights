import numpy as np
import math
import os
import functools
import itertools
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import random
from IPython import embed
from tqdm import tqdm
import scipy.stats as ss

from game import Game

EPSILON = 1e-10

# Given a cce return its marginal distribution
def marginals_from_cce(cce):
    num_players = len(cce.shape)
    marginals = []
    for player in range(num_players):

        # Sum over all axes except the player axes

        list_without_player = tuple(list(range(0, player)) + list(range(player + 1,
                                                                  num_players)))
        marginals.append(np.sum(cce, axis=list_without_player))

    return marginals
    
def marginals_from_raw_equilibrium(equilibrium, weights):
    num_players = equilibrium.shape[-2]
    return np.sum(np.multiply(np.expand_dims(weights, axis=tuple(range(1,
                                                                     num_players+1))),
                              equilibrium), axis=0) / np.sum(weights)

def final_payoff(equilibrium, game, weights):
    return average_payoff(equilibrium, game, weights)[-1]

def average_payoff(equilibrium, game, weights):

    average = np.zeros(game.num_players)
    payoffs = []
    total_weight = 0
    total_payoff = 0

    for moves, weight in zip(equilibrium, weights):
        total_payoff += game.payoff(moves) * weight
        total_weight += weight
        payoffs.append(np.sum(total_payoff) / total_weight)

    return payoffs

# This code is only sound for a 2PZS game
def compute_exploitability_over_time(equilibrium, game, weights, num_players, num_moves):

    converted_game = Game(num_players, num_moves, False)
    converted_game.matrix = game

    exploitability = []
    mixed_strategy = np.zeros((num_players, num_moves))
    total_weight = 0
    for t, play in tqdm(enumerate(equilibrium)):
        for player in range(num_players):
            mixed_strategy[player] += weights[t] * play[player]

        total_weight += weights[t]

        exploitability.append(compute_exploitability(converted_game, np.copy(mixed_strategy) / (total_weight)))

    return exploitability

def pairwise_cramer(equilibrium, weights):

    num_players = equilibrium[0].shape[0]
    num_moves = equilibrium[0].shape[1]
    weights = np.array(weights) * (len(weights) / sum(weights))
    cramer_v = 0

    for p1 in range(num_players):
        for p2 in range(p1+1, num_players):
            count = np.ones((num_moves, num_moves))
            for moves, weight in zip(equilibrium, weights):

                p1_move = np.random.choice(num_moves, p=moves[p1, :])
                p2_move = np.random.choice(num_moves, p=moves[p2, :])
                count[p1_move][p2_move] += weight

            cramer_v += cramers_corrected_stat(count)

    return cramer_v

#https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def blackwells(regrets):
    pos_regrets = np.copy(regrets)
    pos_regrets[pos_regrets < 0] = EPSILON
    return pos_regrets / np.sum(pos_regrets)

def find_stationary(Q):

    repeated_matrix = np.linalg.matrix_power(Q, 20)
    stationary = repeated_matrix[0, :]

    return stationary

def regret(game, last_moves, player):

    regrets = np.zeros(game.num_moves)
    player_utility = game.payoff(last_moves)[player]
    alt_play = np.copy(last_moves)
    for alt_move in range(game.num_moves):

        alt_play[player] = np.zeros(game.num_moves)
        alt_play[player][alt_move] = 1
        alt_utility = game.payoff(alt_play)[player]

        regrets[alt_move] = (alt_utility - player_utility)

    return regrets

# Given a candidate CE, what is the internal regret
def internal_regret_from_ce(game, ce):
    assert abs(np.sum(ce) - 1.) < np.sqrt(EPSILON), "Not a probability distribution {}".format(np.sum(ce))

    payoff = game.payoff_from_outcome_matrix(np.expand_dims(ce, axis=-1))
    regret = 0.0
    for player in range(game.num_players):
        for base_action in range(game.num_moves):
            for action in range(game.num_moves):
                if action == base_action:
                    continue

                new_ce = copy.deepcopy(ce)

                new_ind = [slice(None)] * len(ce.shape)
                old_ind = [slice(None)] * len(ce.shape)
                new_ind[player] = action
                old_ind[player] = base_action
                new_ce[tuple(new_ind)] += new_ce[tuple(old_ind)]
                new_ce[tuple(old_ind)] = 0
                new_ce = np.stack([new_ce for _ in range(game.num_players)], axis=-1)

                if regret < game.payoff_from_outcome_matrix(new_ce)[player] - payoff[player]:
                    regret = game.payoff_from_outcome_matrix(new_ce)[player] - payoff[player]

    return regret

# Given a candidate CCE, what is the external
def external_regret_from_cce(game, cce):
    assert abs(np.sum(cce) - 1.) < np.sqrt(EPSILON), "Not a probability distribution {}".format(np.sum(cce))

    payoff = game.payoff_from_outcome_matrix(np.expand_dims(cce, axis=-1))
    regret = 0.0
    w_player, w_action = None, None
    for player in range(game.num_players):
        for action in range(game.num_moves):
            new_outcome_matrix = np.zeros(cce.shape)

            ind = [slice(None)] * len(cce.shape)
            ind[player] = action
            new_outcome_matrix[tuple(ind)] = np.sum(cce, axis=player)
            new_cce = np.expand_dims(new_outcome_matrix, axis=-1)

            if regret < game.payoff_from_outcome_matrix(new_cce)[player] - payoff[player]:
                regret = game.payoff_from_outcome_matrix(new_cce)[player] - payoff[player]
                w_player = player
                w_action = action

    return regret, w_player, w_action

# If you played the marginals instead of the correlated strategy, how exploitable are you?
def compute_exploitability(game, mixed_strategy, output_type="max"):

    best_regret = 0
    all_payoff = game.payoff(mixed_strategy)

    for player in range(game.num_players):
        for alt_move in range(game.num_moves):

            alt_strat = np.copy(mixed_strategy)
            alt_strat[player] = np.zeros(game.num_moves)
            alt_strat[player][alt_move] = 1
            
            if output_type == "max":
                best_regret = max(best_regret, game.payoff(alt_strat)[player] - all_payoff[player])
            elif output_type == "sum":
                best_regret += max(game.payoff(alt_strat)[player] - all_payoff[player], 0)
            else:
                raise Exception("Not implemented")

    return best_regret

def convert_to_outcome_matrix(game, equilibrium, weights):

    assert len(equilibrium.shape) == 3, "Not the correct input format for raw equilibrium"

    num_actions = equilibrium.shape[-1]
    num_players = equilibrium.shape[-2]

    outcome_matrix = np.zeros([num_actions] * num_players)
    for i in range(equilibrium.shape[0]):
        local_outcome_matrix = game.marginals_to_outcome_matrix(equilibrium[i])[..., 0]
        outcome_matrix += weights[i] * local_outcome_matrix

    return outcome_matrix / np.sum(weights)

def check_cce(game, cce, regret, weights):
    outcome_matrix = convert_to_outcome_matrix(game, cce, weights)
    return external_regret_from_cce(game, outcome_matrix)[0] <= regret

def check_ce(game, ce, regret, weights):
    outcome_matrix = convert_to_outcome_matrix(game, ce, weights)
    return internal_regret_from_ce(game, outcome_matrix) <= regret
