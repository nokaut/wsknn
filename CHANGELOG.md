## Version 0.1.3.post1:

- Changed API with metrics.
- Fit method takes hyperparameters.
- Changed API for precision calculation.
- Session slicing bug corrected.

## Version 0.1.3:

- Make predictions from the List of user actions,
- Include required event in the `predict` method parameters,
- Let user decide if scores are calculated from a sliding window or fixed-length size window (the latter protects from the counting the same results multiple times).

## Version 0.1.2:

- Load JSON Lines and gzipped JSON Lines,
- Force model to choose sessions with a specific event,
- Prec@k and Rec@k metrics added to the evaluation metrics,
- Test MRR, Precison and Recall.


## Version 0.1.1:

- Timestamps can be floating-point numbers,
- MRR bug (ZeroDivisionError),
- Typos and descriptions in docstrings,
