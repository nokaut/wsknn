from wsknn.evaluate.scores.scores import mrr_func, recall_func, precision_func


RELEVANT_ITEMS = list('abcdefgh')
RELEVANT_ITEMS_RECALL = list('abkl')
RECOMMENDATIONS = list('adk')


# TEST SCORES

def test_mrr():
    score = mrr_func(recommendations=RECOMMENDATIONS, relevant_items=RELEVANT_ITEMS)
    # 1.0
    assert score == 1


def test_precision():
    score = precision_func(recommendations=RECOMMENDATIONS, relevant_items=RELEVANT_ITEMS)
    # 0.(6)
    assert score > 0.65
    assert score < 0.67


def test_recall():
    score = recall_func(recommendations=RECOMMENDATIONS, relevant_items=RELEVANT_ITEMS_RECALL)
    # 0.5
    assert score == 0.5
