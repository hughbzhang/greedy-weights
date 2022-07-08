# External libraries
import numpy as np
import nashpy as nash
import scipy.stats as stats
from IPython import embed

# Internal Libraries
import er
import ir
import util
from game import Game

def main():

    num_players, num_moves = 2, 10
    game = Game(num_players=num_players, num_moves=num_moves, zero_sum=True)

    # Basic computation of a coarse-correlated equilibrium (cce) using external regret minimization
    cce, cce_regrets, uniform_cce_weights, _ = er.external_regret(game,
                                                iterations=1000,
                                                use_pure_strategies=True)

    assert util.check_cce(game, cce, cce_regrets[-1], uniform_cce_weights), "Not an approximate CCE"

    # Same as above, except with greedy weights this time for faster convergence
    greedy_cce, greedy_cce_regrets, greedy_cce_weights, _ = er.external_regret(game,
                                                iterations=1000,
                                                use_pure_strategies=True,
                                                dynamic=True)

    assert util.check_cce(game, greedy_cce, greedy_cce_regrets[-1], greedy_cce_weights), "Not an approximate CCE"


    # We can also compute a correlated equilibrium with internal regret minimization. The API for ir.internal_regret
    # largely mimics that of er.external_regret
    ce, ce_regrets, ce_weights, _ = ir.internal_regret(game,
                                                iterations=1000,
                                                method="hart",
                                                use_pure_strategies=True)

    assert util.check_ce(game, ce, ce_regrets[-1], ce_weights), "Not an approximate CE"

    # Finally, if the game is a 2-player zero-sum game, marginals of a coarse-correlated equilibrium are guaranteed to be Nash,
    # so we can use regret minimization to find the Nash of these games as well.

    if num_players == 2 and game.zero_sum:
        nash = util.marginals_from_raw_equilibrium(cce, uniform_cce_weights)
        exploitability = util.compute_exploitability(game, nash)
        assert exploitability < cce_regrets[-1], "Not an approximate Nash"


if __name__ == "__main__":
    main()
