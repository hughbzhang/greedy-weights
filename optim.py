import numpy as np
import math

def get_objective(R, r, w, T):
    if np.isinf(w):
        return np.sum(np.clip(r, 0, None) ** 2)
    terms = np.clip(R + r * w, 0, None)
    return np.sum((terms / (T + w)) ** 2)


def find_optimal_weight(R, r, T):
    """
    Arguments:
        R: A 1d numpy array containing concatenated total regrets for all players.
        r: A 1d numpy array containing concatenated immediate regrets for all players;
        T: current T
    
    Returns:
        w: min_w sum_a max(0, R + wr)^2 / (T + w)^2

    The function we're trying to minimize is piecewise quadratic with <=N cutpoints
    at `w = -R(a)/r(a)` for each a.
    Within each of these <=N+1 intervals, we can compute a binary vector Z of which
    actions have R(a)+wr(a)>0 . We can solve for the (possible) stationary point
    in this interval, which leads to the equation:

    1/(T+w)^3 sum_a Z(a) [ 2r(a)(R(a) + wr(a))(T+w) - 2(R(a) + wr(a))^2 ] = 0
    which rearranges to

          sum_a Z(a)[ R(a)^2 - r(a)R(a)T ]
     w  = --------------------------------
          sum_a Z(a)[ r(a)^2T - r(a)R(a) ]

    which is only valid if it's in the interval.

    The optimal w will be the minimum of w at each cutpoint, at each inner stationary point, and at the
    boundaries 0 and inf.
    """

    assert isinstance(R, np.ndarray)
    assert isinstance(r, np.ndarray)
    assert len(R.shape) == 1
    assert len(r.shape) == 1

    # import matplotlib.pyplot as plt
    # X = np.linspace(0, 4, 100)
    # Y = [get_objective(R, r, w, T) for w in X]
    # plt.plot(X, Y)
    # plt.ylim(np.min(Y), np.min(Y) * 2)
    # plt.savefig("plot.png", bbox_inches="tight", dpi=300)

    cutpoints = -R[r != 0] / r[r != 0]  # not NaN
    # print("Cutpoints", cutpoints)
    cutpoints = cutpoints[cutpoints > 0]  # w > 0

    cutpoints.sort()
    cutpoints = np.array([0] + cutpoints.tolist() + [np.inf])  # add boundaries

    to_check = cutpoints.tolist()
    for ci in range(len(cutpoints) - 1):
        start, end = cutpoints[ci], cutpoints[ci + 1]

        # sanity check!
        if end != np.inf:
            delta = 1e-3 * (end - start)
            Zs = (R + r * (start + delta)) > 0
            Ze = (R + r * (end - delta)) > 0
            assert np.all(Zs == Ze), f"{start}, {end}"

        mid = 2 * start + 1 if np.isinf(end) else (start + end) / 2
        R_mid = R + r * mid
        Z = (R_mid > 0).astype(float)
        # print(f"XXX {R_mid} {Z}")
        num = np.dot(Z, R ** 2 - r * R * T)
        denom = np.dot(Z, r ** 2 * T - r * R)
        w_star = num / denom

        # print(f"ZZZ start={start} end={end} num={num} denom={denom} w_star={w_star}")

        if w_star > start and w_star < end:
            to_check.append(w_star)

    to_check.append(1)
    #return to_check

    to_check_values = np.array([get_objective(R, r, w, T) for w in to_check])
    # print({x: y for x, y in zip(to_check, to_check_values)})
    best_idx = np.argmin(to_check_values)
    best_w, best_phi = to_check[best_idx], to_check_values[best_idx]

    return best_w, best_phi
    #assert best_phi == get_objective(
        #R, r, best_w, T
    #), f"{best_phi} {get_objective(R, r, best_w, T)}"

    for test_w in 0, best_w - 1e-2, best_w + 1e-2, best_w - 1e-4, best_w + 1e-4:
        assert best_phi <= get_objective(
            R, r, test_w, T
        ), f"{best_w} {best_phi} {test_w} : {get_objective(R, r, test_w, T)}"



if __name__ == "__main__":
    R = np.array([-5.5, 10, 3, -10, -1, 0, 1])
    r = np.array([1, -4, -0.1, 2, 0, 0, 2])
    for T in range(3, 10):
        print(f"Result (T={T}): {find_optimal_weight(R, r, T)}")

    R = np.array(
        [
            -0.26713593,
            -0.24618386,
            -0.1153491,
            0.0341592,
            0.00880994,
            0.03115524,
            0.03785481,
            0.07808526,
            -0.00153117,
            0.13818653,
            0.15653713,
            0.14047404,
            0.0049379,
            0.0,
            -0.12271792,
            -0.00667636,
            0.17494799,
            -0.00442807,
            -0.04112564,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    r = np.array(
        [
            -0.15825225,
            -0.14370357,
            -0.22585158,
            0.02448834,
            0.0656593,
            0.09458186,
            0.08456595,
            0.05372603,
            0.05115147,
            -0.04220097,
            -0.00566314,
            -0.03619601,
            0.00902815,
            0.0,
            -0.29766591,
            -0.18162435,
            0.0,
            -0.17937607,
            -0.21607363,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    print(find_optimal_weight(R, r, 1))

    R = np.array(
        [
            -2.33197328,
            -2.18423546,
            -2.05070744,
            -0.3282112,
            -0.2388985,
            -0.02019164,
            -0.03985015,
            -0.02424205,
            -0.34523432,
            -0.2283047,
            0.00961801,
            -0.1920316,
            -0.51222598,
            0.0,
            -1.81482138,
            0.00792983,
            0.0,
            -0.15911359,
            -0.07859699,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    r = np.array(
        [
            -0.42367306,
            -0.40272099,
            -0.27188623,
            -0.12237793,
            -0.14772719,
            -0.12538189,
            -0.11868232,
            -0.07845187,
            -0.1580683,
            -0.0183506,
            0.0,
            -0.01606309,
            -0.15159923,
            0.0,
            -0.11445323,
            0.0,
            0.16812611,
            -0.00414115,
            -0.02113652,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    print(find_optimal_weight(R, r, 1))
    print(find_optimal_weight(R, r, 1000))
