import numpy as np
import time
import json
import seaborn as sns
import scipy.special
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

import util
import er
import optim
from game import Game

# Global constants
USE_TIME = False
bars = np.linspace(0, 100, 500)

RUN_REGRET_TESTS = False
EPSILON = 1e-15
time_to_plot, regrets_to_plot = {}, defaultdict(lambda: [])

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Finalize and save the plot with the relevant experimental parameters
def finalize_plot(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, method, ax, game_reps):

    minlength = 1e10
    for name in regrets_to_plot:
        for i in range(len(regrets_to_plot[name])):
            minlength = min(minlength, len(regrets_to_plot[name][i]))

    for name in regrets_to_plot:
        if "Greedy" in name:
            linewidth = 3
        else:
            linewidth = 1
        if USE_TIME:
            for i in range(len(regrets_to_plot[name])):
                regrets_to_plot[name][i] = regrets_to_plot[name][i][:minlength]
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(bars[:len(avg)], avg, label=name, linewidth=linewidth, yerr=std)
        else:
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(np.arange(1, len(avg)+1), avg, label=name, linewidth=linewidth, yerr=std)

    fpath = "figs/{}_{}".format(experiment_name, args.use_time)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    plt.xscale("log"), plt.yscale("log")
    plt.plot(4/iterations, label="O(1/sqrt(t))")

    game_type = "zero" if zero_sum else "general"
    strategy = "pure" if use_pure_strategies else "mixed"
    game_name = "Internal Regret Minimization for {}-player {}-action {}-sum games".format(game.num_players, game.num_moves, game_type)

    plt.title(game_name)
    if USE_TIME:
        plt.xlabel("Seconds (logscale)")
    else:
        plt.xlabel("Iterations (logscale)")
    plt.ylabel("Regret (logscale)")
    plt.legend(loc=3, prop={'size': 8})
    fname = "{}/{}P-{}A-{}-{}-{}I-{}S-INTERNAL-{}".format(fpath, game.num_players, game.num_moves, game_type, strategy, iterations, seed, method)
    print(fname)
    plt.savefig(fname + ".pdf", dpi=1000)

    dumped = json.dumps(regrets_to_plot, cls=NumpyEncoder)

    with open(fname + ".json", "w+") as f:
        json.dump(dumped, f)

# Save the regrets to be plotted
def plot_regrets(equilibrium, regrets, game, weights, name, time):

    time = np.array(time) - min(time)
    if RUN_REGRET_TESTS:
        # After the discounting, you can't run this any more because of numerical precision
        all_regrets = manual_regret(equilibrium, game, weights)
        #assert np.max(np.abs(regrets - all_regrets)) < 1e-10, "Potentially a mistake calculating regret (or over/underflow in the testing function)"

    #regrets = er.manual_regret(equilibrium, game, weights)

    if USE_TIME:

        marker = 0
        results = []

        for r, t in zip(regrets, time):

            while marker < len(bars) and t > bars[marker]:
                results.append(r)
                marker += 1

            if marker >= len(bars):
                break

        results = np.array(results)
        regrets_to_plot[name].append(np.copy(results))

    else:

        regrets_to_plot[name].append(np.copy(regrets))

# Compare dynamic weights with various different floors
def sweep_floor(game, use_pure_strategies, iterations, method, seed):

    uniform, step = np.ones(iterations-1), np.arange(1, iterations)

    for floor in [100, 10, 4, 2, 1, 0.5]:
        exact_equilibrium, exact_regrets, exact_weights, exact_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, dynamic=True, use_potential_function=True, floor=True, floor_amount=floor)
        plot_regrets(exact_equilibrium, exact_regrets, game, exact_weights, "Floor is {}% of average".format(1/floor * 100), exact_time)

# Compare dynamic weights with generic CFR w/o any additional gadgets
def solo_compare(game, use_pure_strategies, iterations, method, seed):

    uniform, step = np.ones(iterations-1), np.arange(1, iterations)
    exact_equilibrium, exact_regrets, exact_weights, exact_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, dynamic=True, use_potential_function=True)
    cfr_equilibrium, cfr_regrets, _, cfr_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method)
    plot_regrets(exact_equilibrium, exact_regrets, game, exact_weights, "Dynamic Weights", exact_time)
    plot_regrets(cfr_equilibrium, cfr_regrets, game, uniform, "Regret Matching", cfr_time)

# Sweep through all regret minimization methods except dynamic weights
def full_sweep(game, use_pure_strategies, iterations, method, seed):

    # Weighting schemes
    uniform, step = np.ones(iterations-1), np.arange(1, iterations)

    plot_cfr, plot_plus, plot_linear, plot_dynamic = True, True, True, True

    if plot_cfr:
        cfr_equilibrium, cfr_regrets, _, cfr_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method)
        alt_cfr_equilibrium, alt_cfr_regrets, _, alt_cfr_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, alternating=True)
        opt_cfr_equilibrium, opt_cfr_regrets, _, opt_cfr_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, optimism=True)

        plot_regrets(cfr_equilibrium, cfr_regrets, game, uniform, "RM", cfr_time)
        plot_regrets(alt_cfr_equilibrium, alt_cfr_regrets, game, uniform, "RM w/ Alternating Updates", alt_cfr_time)
        plot_regrets(opt_cfr_equilibrium, opt_cfr_regrets, game, uniform, "RM w/ Optimism", opt_cfr_time)

    if plot_plus:

        plus_equilibrium, plus_regrets, _, plus_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, cfrplus=True)
        alt_plus_equilibrium, alt_plus_regrets, _, alt_plus_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, alternating=True, cfrplus=True)
        opt_plus_equilibrium, opt_plus_regrets, _, opt_plus_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, optimism=True, cfrplus=True)

        plot_regrets(plus_equilibrium, plus_regrets, game, step, "RM+", plus_time)
        plot_regrets(alt_plus_equilibrium, alt_plus_regrets, game, step, "RM+ w/ Alternating Updates", alt_plus_time)
        plot_regrets(opt_plus_equilibrium, opt_plus_regrets, game, step, "RM+ w/ Optimism", opt_plus_time)

    if plot_linear:

        linear_equilibrium, linear_regrets, _, linear_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, linear=True)
        alt_linear_equilibrium, alt_linear_regrets, _, alt_linear_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, alternating=True, linear=True)
        opt_linear_equilibrium, opt_linear_regrets, _, opt_linear_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, method=method, optimism=True, linear=True)

        plot_regrets(linear_equilibrium, linear_regrets, game, step, "Linear RM", linear_time)
        plot_regrets(alt_linear_equilibrium, alt_linear_regrets, game, step, "Linear RM w/ Alterating Updates", alt_linear_time)
        plot_regrets(opt_linear_equilibrium, opt_linear_regrets, game, step, "Linear RM w/ Optimism", opt_linear_time)

    if plot_dynamic:
        pot_equilibrium, pot_regrets, pot_weights, pot_time = internal_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=int(iterations/3), method=method, dynamic=True, use_potential_function=True)

        plot_regrets(pot_equilibrium, pot_regrets, game, pot_weights, "Greedy Weights (ours)", pot_time)

# Compare variations of dynamic weights to each other
def compare_dynamic(game, use_pure_strategies, iterations, method, seed):

    pot_equilibrium, pot_regrets, pot_weights, pot_time = internal_regret(game, use_pure_strategies=True, iterations=iterations, method=method, dynamic=True, use_potential_function=True)
    floor_equilibrium, floor_regrets, floor_weights, floor_time = internal_regret(game, use_pure_strategies=True, iterations=iterations, method=method, dynamic=True, use_potential_function=True, floor=True)

    plot_regrets(pot_equilibrium, pot_regrets, game, pot_weights, "Dynamic", pot_time)
    plot_regrets(floor_equilibrium, floor_regrets, game, floor_weights, "Dynamic (floor)", floor_time)

# Update regrets given regret minimization settings and the last moves played
def update_regrets(SWAP, game, last_moves, weight=1, cfrplus=False, dilution=1, only_player=None, payback={}, optimism=False):

    if bool(payback):
        assert optimism, "Payback should only be nonempty if optimism is True"

    SWAP *= 1./dilution
    new_payback = {}

    for player in range(game.num_players):

        if only_player is not None and player != only_player:
            continue

        player_regret = weight * util.regret(game, last_moves, player)

        for base_move in range(game.num_moves):

            if (player, base_move) in payback:
                SWAP[player, base_move, :] -= payback[player, base_move]

            SWAP[player, base_move, :] += (last_moves[player][base_move] * player_regret)

            # Optimism adds it twice and then subtracts it out later
            if optimism:
                SWAP[player, base_move, :] += (last_moves[player][base_move] * player_regret)
                new_payback[player, base_move] = (last_moves[player][base_move] * player_regret)
    
    if cfrplus:
        SWAP[SWAP<0] = 0

    return SWAP, new_payback

# Computes the SWAP regret manually to double check for bugs
def manual_regret(equilibrium, game, weights):

    SWAP = np.zeros((game.num_players, game.num_moves, game.num_moves))
    regret_over_time = []
    total_weight = 0

    for t, moves in tqdm(enumerate(equilibrium)):

        utility = game.payoff(moves)
        total_weight += weights[t]

        for player in range(game.num_players):

            player_regrets = util.regret(game, moves, player)
            for base_move in range(game.num_moves):
                SWAP[player, base_move, :] += weights[t] * moves[player][base_move] * player_regrets
        
        cur_regret = total_regrets(SWAP, game)
        regret_over_time.append(cur_regret / total_weight)
        
    return np.array(regret_over_time)

# Variation of regret matching which we use in the dynamic weights paper
def low_inertia(SWAP, player, last_move, game, explore=0):
    regrets = np.array([SWAP[(player, last_move, A)] for A in range(game.num_moves)])
    regrets[regrets <= 0] = 0
    regrets[last_move] = EPSILON
    regrets += explore
    regrets /= np.sum(regrets)
    return regrets

# Implements Hart and Mas Colell 2000 Regret Matching
def regret_matching(SWAP, player, last_move, VISITS):
    regrets = np.array([SWAP[(player, last_move, A)] for A in range(num_moves)])
    INERTIA = num_moves - 1
    regrets[regrets <= 0] = 0
    regrets /= (VISITS[player, last_move] * INERTIA)
    regrets[last_move] += EPSILON

    assert np.sum(regrets) <= 1, "Regrets were calculated wrong"

    regrets[last_move] = 1 - np.sum(regrets)
    return regrets

 # Each player chooses their moves based on past regrets
def compute_next_moves(SWAP, method, use_pure_strategies, last_moves, game, explore):

    for player in range(game.num_players):

        if use_pure_strategies:
            if method == "hart":
                last_move = np.argmax(last_moves[player])
                probs = low_inertia(SWAP, player, last_move, game, explore=explore)
            else:
                probs = blum_mansour(SWAP, player, game)

            last_moves[player, :] = np.zeros(game.num_moves)
            last_moves[player][np.random.choice(game.num_moves, p=probs)] = 1
        else:
            assert args.method == "blum", "Regret matching not implemented for mixed strategies"
            probs = blum_mansour(SWAP, player, game)
            last_moves[player, :] = probs

    return last_moves

# Computes internal regret
def internal_regret(game,
                    iterations,
                    method,
                    use_pure_strategies,
                    seed=None,
                    # Turn on dynamic weights
                    dynamic=False,
                    # Optimistic updates from Syrgkanis+ 2015
                    optimism=False,
                    # CFR+ from Tammelin 2014
                    cfrplus=False,
                    # Linear CFR from Brown & Sandholm 2019
                    linear=False,
                    # Alternating updates from Tammelin 2014
                    alternating=False,
                    # use the potential function instead of minimizing the regrets
                    use_potential_function=True,
                    # use the max regret as the function to minimize
                    use_max=False,
                    # defunct parameters that controlled finding optimal CE
                    reward=False,
                    alpha=0,
                    # use optim.py to dtermine the optimal weight
                    use_optimize=True,
                    # don't print the progress bar
                    silent=False,
                    # floor the weights
                    floor=False,
                    # How much to floor the weights?
                    floor_amount=10):

    np.random.seed(seed)
    CLOCK = [time.time()]

    # First move is uniform random by default
    if not use_pure_strategies:
        last_moves = np.ones((game.num_players, game.num_moves)) / game.num_moves
    else:
        last_moves = np.zeros((game.num_players, game.num_moves))
        for p in range(game.num_players):
            last_moves[p][np.random.randint(0, game.num_moves)] = 1

    # SWAP keeps track of all the regrets. Payback is responsible for "paying back" the optimistic regrets (only if using optimism).
    SWAP, payback = update_regrets(np.zeros((game.num_players, game.num_moves, game.num_moves)), game, last_moves)
    EQUILIBRIUM_SWAP, _ = update_regrets(np.zeros((game.num_players, game.num_moves, game.num_moves)), game, last_moves)

    # Keeps track of the regret over time.
    REGRET = [total_regrets(SWAP, game)]

    # ALL_MOVES stores the history of plays, which you need to actually compute the correlated equilibrium
    ALL_MOVES = [np.copy(last_moves)]
    total_weight, WEIGHTS = 1, np.array(1.)

    # Used only for the reward dynamic weighting
    past_utility = np.sum(game.payoff(last_moves))

    if not silent:
        print("Starting regret minimization")

    for t in tqdm(range(2, iterations), disable=silent):

        last_moves = compute_next_moves(SWAP, method, use_pure_strategies, last_moves, game, alpha/t)
 
        if dynamic:
            best_w = optimize(SWAP, game, last_moves, total_weight, t)
            #best_w = find_best_weight(dynamic_range, SWAP, game, last_moves, total_weight, t, use_potential_function=use_potential_function, use_max=use_max, exploit=exploit, MIXED_STRAT=MIXED_STRAT)

            if floor:
                best_w = max(best_w, total_weight/(t*floor_amount))

        elif reward:
            cur_utility = np.sum(game.payoff(last_moves))
            raw_logits = (alpha * t * np.array([past_utility / total_weight, cur_utility]))
            best_w = t ** scipy.special.softmax(raw_logits)[1]
        else:
            best_w = 1

        if linear:
            dilution = t / (t-1)
        else:
            dilution = 1

        if alternating:
            only_player = t % game.num_players
        else:
            only_player = None

        if best_w == np.inf:
            best_w = 1
            dilution = 1e12
            
        SWAP, payback = update_regrets(SWAP, game, last_moves, weight=best_w, cfrplus=cfrplus, dilution=dilution, only_player=only_player, payback=payback, optimism=optimism)
        total_weight = total_weight / dilution + best_w
        WEIGHTS *= 1./dilution
        WEIGHTS = np.append(WEIGHTS, best_w)

        past_utility += best_w * np.sum(game.payoff(last_moves))

        if not cfrplus:
            EQUILIBRIUM_SWAP, _ = update_regrets(EQUILIBRIUM_SWAP, game, last_moves, weight=best_w, dilution=dilution)
            REGRET.append(total_regrets(EQUILIBRIUM_SWAP, game) / total_weight)
        else:
            EQUILIBRIUM_SWAP, _ = update_regrets(EQUILIBRIUM_SWAP, game, last_moves, weight=t, dilution=dilution)
            REGRET.append(total_regrets(EQUILIBRIUM_SWAP, game) / (t*(t+1)/2))

        ALL_MOVES.append(np.copy(last_moves))

        # Normalize after each step.
        total_weight *= (t / total_weight)
        WEIGHTS *= (t / total_weight)
        CLOCK.append(time.time())


    return np.array(ALL_MOVES), np.array(REGRET), WEIGHTS, CLOCK
    CLOCK = [time.time()]

# Dynamically choose the weights to minimize something
def optimize(        SWAP,
                     game,
                     last_moves, 
                     total_weight,
                     timestep):

    NEW_SWAP, _  = update_regrets(np.zeros(SWAP.shape), game, last_moves, weight=1)

    R = np.array([SWAP[player][a][b] for player in range(game.num_players) for a in range(game.num_moves) for b in range(game.num_moves)])
    r = np.array([NEW_SWAP[player][a][b] for player in range(game.num_players) for a in range(game.num_moves) for b in range(game.num_moves)])

    best_w, best_phi = optim.find_optimal_weight(R, r, total_weight)
    if False:
        SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=best_w)
        assert abs(best_phi - potential_func(SWAP_tmp, game) / ((total_weight + best_w)**2)) < EPSILON

    return best_w
    
# Dynamically choose the weights to minimize something
def find_best_weight(dynamic_range,
                     SWAP,
                     game,
                     last_moves, 
                     total_weight,
                     timestep,
                     use_potential_function=False,
                     use_max=False,
                     exploit=False,
                     MIXED_STRAT=None):
    # Sweep from weights of 1 to t^2. Using logspace is CRUCIAL for performance
    candidates = np.logspace(0, np.log(timestep**3), dynamic_range, base=np.e)
    arr = []

    for w in candidates:

        if exploit:
            # Dynamically choose the weight to minimize exploitability

            NEW_MIXED = (np.copy(MIXED_STRAT) + w * last_moves) / (total_weight + w)
            arr.append(compute_exploitability(game, NEW_MIXED))
        else:
            SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=w)
            if use_potential_function:
                # Minimize the potential function
                arr.append(potential_func(SWAP_tmp, game) / ((total_weight + w)**2))

            elif use_max:
                # Minimize max regret
                arr.append(max_regrets(SWAP_tmp, game) / (total_weight + w))

            else:
                # Minimize the regrets
                arr.append(total_regrets(SWAP_tmp, game) / (total_weight + w))

    return candidates[np.argmin(arr)]
       
# Follows the procedure here: http://www.cs.cmu.edu/~arielpro/15896/docs/notes4.pdf
def blum_mansour(SWAP, player, game):
    Q = np.zeros((game.num_moves, game.num_moves))
    for move in range(game.num_moves):

        # We add EPSILON for stability
        regrets = np.array([EPSILON + SWAP[(player, move, A)] for A in range(game.num_moves)])
        regrets[regrets < EPSILON] = EPSILON
        Q[move, :] = np.copy(regrets) / np.sum(regrets)

    # Q is the Markov matrix described in the link above
    probs = util.find_stationary(Q)
    return probs

# Compute max internal regret for any player
def max_regrets(SWAP, game):
    return max(sum(max(0, np.max(SWAP[player, move, :])) for move in range(game.num_moves)) for player in range(game.num_players))

# Compute total internal regret for all players
def total_regrets(SWAP, game):
    return sum(max(0, np.max(SWAP[player, move, :])) for player in range(game.num_players) for move in range(game.num_moves))

# Potential func, which is the squared positive regrets. Should monotonically go decrease.
def potential_func(SWAP, game):
    return np.sum(np.square(SWAP.clip(0)))
              
def main(num_players, num_moves, zero_sum, seed, iterations, use_pure_strategies, method, experiment_name, game_reps):

    assert args.sweep_floor + args.full_sweep + args.solo_compare + args.compare_dynamic == 1, "Exactly one option must be specified"

    np.random.seed(seed)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', [plt.get_cmap('tab20')(1. * i / 13) for i in range(13)])

    for i in range(game_reps):
        print("Iteration {} of {}".format(i, game_reps))
        game = Game(num_players=num_players, num_moves=num_moves, zero_sum=zero_sum)

        if args.solo_compare:
            solo_compare(game, use_pure_strategies, iterations, method, seed)
        elif args.full_sweep:
            full_sweep(game, use_pure_strategies, iterations, method, seed)
        elif args.compare_dynamic:
            compare_dynamic(game, use_pure_strategies, iterations, method, seed)
        elif args.sweep_floor:
            sweep_floor(game, use_pure_strategies, iterations, method, seed)
        else:
            raise Exception("This option does not exist")

    finalize_plot(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, method, ax, game_reps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify internal regret settings')
    parser.add_argument('--num_moves', type=int, default=3, help="Number of moves in the game, default 3")
    parser.add_argument('--num_players', type=int, default=2, help="Number of players in the game, default 2")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of iterations to run regret minimization, default 1K")
    parser.add_argument('--seed', type=int, default=42, help="Random seed, default 42")
    parser.add_argument('--zero_sum', action="store_true", default=False, help="Make the game zero sum, default false")
    parser.add_argument('--solo_compare', action="store_true", default=False, help="Compare only RM and dynamic weights")
    parser.add_argument('--sweep_floor', action="store_true", default=False, help="Sweep through the various floors")
    parser.add_argument('--compare_dynamic', action="store_true", default=False, help="Compare the variants of dynamic weights")
    parser.add_argument('--full_sweep', action="store_true", default=False, help="Sweep through all non dynamic methods")
    parser.add_argument('--pure', action="store_true", default=False, help="Use pure strategies for regret minimization, default false")
    parser.add_argument('--use_time', action="store_true", default=False,
                        help="Use time instead of iterations")
    parser.add_argument("--experiment_name", type=str, help="What is this experiment called?", default="CYBERTRASH")
    parser.add_argument("--game_reps", type=int, help="How many games to repeat for", default=1)
    parser.add_argument("--method", type=str, help="IR Method. Must be either 'hart' or 'blum'", default='hart')
    args = parser.parse_args()
    print(args)

    USE_TIME = args.use_time

    main(args.num_players, args.num_moves, args.zero_sum, args.seed, args.iterations, args.pure, args.method, args.experiment_name, args.game_reps)
