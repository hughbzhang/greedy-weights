import numpy as np
import json
import math
import os
import functools
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from IPython import embed
from tqdm import tqdm
import argparse

# Global constants
EPSILON = 1e-10
RUN_TESTS = False

class Game:
    def __init__(self, num_players, num_moves, zero_sum, special_type=None):

        self.num_players = num_players
        self.num_moves = num_moves
        self.zero_sum = zero_sum
        if special_type is not None:

            if special_type == "stag_hunt":

                assert num_players == 2 and not zero_sum and num_moves == 2, "Options wrong for stag hunt"

                self.matrix = np.array(
                [
                    [
                        [1, 1],  # MM
                        [1, -5], # MH
                    ],
                    [
                        [-5, 1], # HM
                        [3, 3],  # HH
                    ]
                ])


            elif special_type == "super_stag_hunt":

                assert num_players == 2 and not zero_sum and num_moves == 2, "Options wrong for stag hunt"

                self.matrix = np.array(
                [
                    [
                        [1, 1],  # MM
                        [1, -20], # MH
                    ],
                    [
                        [-20, 1], # HM
                        [3, 3],  # HH
                    ]
                ])


            elif special_type == "two_center":

                assert num_players == 3 and zero_sum and num_moves == 2, "Options configured wrong"

                self.matrix = np.zeros((2, 2, 2, 3))

                self.matrix[:, :, :, 2] = np.ones((2, 2, 2))
                self.matrix[0, 0, 0, :]  = [1, 1, 0]
                self.matrix[1, 1, 1, :]  = [1, 1, 0]
            else:
                raise Exception("Unknown special type {}".format(special_type))
        else:
            self.matrix = np.random.random([num_moves] * num_players + [num_players])

            if zero_sum:
                # Rewards sum to constant to ensure that all utilities are between 0 and 1
                self.matrix[...,-1] = num_players - 1 - np.sum(self.matrix[...,0:-1], axis=-1)
                self.matrix /= (num_players - 1)

        self.reset_queries()


    def reset_queries(self):
        self.queries = np.zeros(self.matrix.shape)

    def marginals_to_outcome_matrix(self, moves):
        # All hail StackOverflow https://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors
        return np.repeat(np.expand_dims(functools.reduce(np.multiply, np.ix_(*moves)), -1), self.num_players, axis=-1)

    def payoff_from_outcome_matrix(self, outcome_matrix):
        # Sum everything along the last axis
        payoffs =  np.sum(outcome_matrix * self.matrix, axis=tuple(range(self.matrix.ndim - 1)))
        return payoffs.astype(float)

    def payoff(self, moves):

        #assert abs(np.sum(moves) - self.num_players) < EPSILON, moves

        # Check if pure strategies
        if all(abs(np.max(moves, axis=-1) - 1) < EPSILON):
            self.queries[tuple(np.argmax(moves, axis=1))] = 1

            return self.matrix[tuple(np.argmax(moves, axis=1))].astype(float)

        outcome_matrix = self.marginals_to_outcome_matrix(moves)
        return self.payoff_from_outcome_matrix(outcome_matrix)

if __name__ == "__main__":
    main()
