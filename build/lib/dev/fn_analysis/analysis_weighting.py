import numpy as np
from wsknn.weighting.weighting import linear_session_score, log_session_score, quadratic_session_score
from wsknn.weighting.weighting import linear_item_score, inv_pos_item_score, log_item_score, quadratic_item_score

import matplotlib.pyplot as plt


def analyse_weighting_output():

    arr10_length = 10
    arr100_length = 100
    arr1000_length = 1000

    arr10 = np.arange(0, arr10_length, 1)
    arr100 = np.arange(0, arr100_length, 1)
    arr1000 = np.arange(0, arr1000_length, 1)

    arrs = [(arr10_length, arr10),
            (arr100_length, arr100),
            (arr1000_length, arr1000)]

    results = {}
    for arr_tpl in arrs:
        length = arr_tpl[0]
        arr = arr_tpl[1]
        lin_res = [linear_session_score(v, length) for v in arr]
        log_res = [log_session_score(v, length) for v in arr]
        quad_res = [quadratic_session_score(v, length) for v in arr]

        results[length] = (
            lin_res, log_res, quad_res
        )

    for fnl in results.keys():
        plt.figure(figsize=(8, 12))
        plt.plot(results[fnl][0])
        plt.plot(results[fnl][1])
        plt.plot(results[fnl][2])
        plt.legend(['linear', 'log', 'quadratic'])
        plt.show()


def analyse_ranking_output():

    arr10_length = 6
    arr100_length = 10
    arr1000_length = 100

    arr10 = np.arange(1, arr10_length, 1)
    arr100 = np.arange(1, arr100_length, 1)
    arr1000 = np.arange(1, arr1000_length, 1)

    arrs = [(arr10_length, arr10),
            (arr100_length, arr100),
            (arr1000_length, arr1000)]

    results = {}
    for arr_tpl in arrs:
        length = arr_tpl[0]
        arr = arr_tpl[1]
        lin_res = [linear_item_score(v) for v in arr]
        inv_res = [inv_pos_item_score(v) for v in arr]
        log_res = [log_item_score(v) for v in arr]
        quad_res = [quadratic_item_score(v) for v in arr]

        results[length] = (
            lin_res, inv_res, log_res, quad_res
        )

    for fnl in results.keys():
        plt.figure(figsize=(12, 8))
        plt.plot(results[fnl][0])
        plt.plot(results[fnl][1])
        plt.plot(results[fnl][2])
        plt.plot(results[fnl][3])
        plt.legend(['linear', 'inverted', 'log', 'quadratic'])
        plt.show()

if __name__ == '__main__':

    # Lab test (1) - uncomment to run
    # analyse_weighting_output()

    # Summary of test:
    #   PAST elements:
    #     - LOG_FN: returns the largest weights for the past elements,
    #     - QUADRATIC_FN: returns the lowest weights for the past elements.
    #   NEAR-PRESENT elements:
    #      - LINEAR_FN: returns the largest values,
    #      - LOG_FN: returns larger values than QUADRATIC_FN for small sequences == ~10 elements.
    #   GENERAL SHAPE:
    #      - LINEAR_FN: constant change of weights,
    #      - LOG_FN: small weights up to the end of sequence, the newest elements gain the most attention, and even
    #                slightly older are not considered to be relevant,
    #      - QUADRATIC_FN: the oldest elements are penalized more than the middle elements in a sequence. Weights are
    #                      always smaller than in the LINEAR_FN but for the largest sequences weights are much
    #                      bigger than for the LOG_FN.

    ###################################################################################################################

    # Lab test (2) - uncomment to run
    analyse_ranking_output()

    # Summary of test:
    #   PAST elements:
    #     - LOG_FN: returns constant weight above 0, in practice it doesn't leave any element based on its position,
    #     - INV_FN: gradually decreases to zero but is always above the zero value, it can be treated as a function
    #               with properties between LOG_FN and LINEAR_FN or QUADRATIC_FN,
    #     - LINEAR_FN and QUADRATIC_FN: large penatlies for the past values, LINEAR_FN >= 10 gives always 0, and
    #                                   QUADRATIC_FN is very close to zero after only a few steps.
    #   NEAR-PRESENT elements:
    #      - LINEAR_FN: returns the largest values,
    #      - LOG_FN: returns the second largest values,
    #      - INV_FN: returns the third largest values,
    #   GENERAL SHAPE:
    #      - LINEAR_FN: constant change of weights, after 10th step all values are set to 0,
    #      - LOG_FN: small weights up to the end of sequence, the newest elements gain the most attention, and even
    #                slightly older are not considered to be relevant,
    #      - QUADRATIC_FN: the oldest elements are penalized more than the middle elements in a sequence. Weights are
    #                      always smaller than in the LINEAR_FN but for the largest sequences weights are much
    #                      bigger than for the LOG_FN.
    #      - INV_FN: Similar to QUADRATIC_FN but values are always larger and they tend to be > than 0.

    print('Test end')
