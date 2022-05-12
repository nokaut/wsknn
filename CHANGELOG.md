## Version 0.1.4.dev1:

- `predict` method changed to `recommend`, accepts only List of Lists as the input,
- refactored scoring,
- added `recommend_any` parameter to force algorithm to return k-recommendations. If there is less than k recommendations, and this parameter is set to `True`, then algorithm returns missing products from the product pool at random,
- added `.dev` suffix to package version, to pinpoint the fact that package is still unstable.

## Version 0.1.3.post1 (unstable):

- Changed API with metrics.
- Fit method takes hyperparameters.
- Changed API for precision calculation.
- Session slicing bug corrected.

## Version 0.1.3 (unstable):

- Make predictions from the List of user actions,
- Include required event in the `predict` method parameters,
- Let user decide if scores are calculated from a sliding window or fixed-length size window (the latter protects from the counting the same results multiple times).

## Version 0.1.2 (unstable):

- Load JSON Lines and gzipped JSON Lines,
- Force model to choose sessions with a specific event,
- Prec@k and Rec@k metrics added to the evaluation metrics,
- Test MRR, Precison and Recall.


## Version 0.1.1 (stable): 

- Timestamps can be floating-point numbers,
- MRR bug (ZeroDivisionError),
- Typos and descriptions in docstrings,
