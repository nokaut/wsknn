from wsknn.evaluate.metrics import mrr_func, recall_func, precision_func


RELEVANT_ITEMS = list('abcdefgh')
RELEVANT_ITEMS_RECALL = list('abkl')
PREDICTIONS = list('adk')


def test_mrr():
    score = mrr_func(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS)
    # 1.0
    assert score == 1


def test_precision():
    score = precision_func(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS, k=len(PREDICTIONS))
    # 0.(6)
    assert score > 0.65
    assert score < 0.67


def test_recall():
    score = recall_func(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS_RECALL)
    # 0.5
    assert score == 0.5
