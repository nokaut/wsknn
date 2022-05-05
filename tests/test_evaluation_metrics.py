from wsknn.evaluate.metrics import __mrr, __recall, __precision


RELEVANT_ITEMS = list('abcdefgh')
RELEVANT_ITEMS_RECALL = list('abkl')
PREDICTIONS = list('adk')


def test_mrr():
    score = __mrr(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS)
    # 1.0
    assert score == 1


def test_precision():
    score = __precision(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS, k=len(PREDICTIONS))
    # 0.(6)
    assert score > 0.65
    assert score < 0.67


def test_recall():
    score = __recall(preds=PREDICTIONS, rel_items=RELEVANT_ITEMS_RECALL)
    # 0.5
    assert score == 0.5
