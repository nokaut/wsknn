# WSKNN: k-NN recommender for session-based data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6393177.svg)](https://doi.org/10.5281/zenodo.6393177)

## Weighted session-based k-NN - Intro

Do you build a recommender system for your website? K-nearest neighbors algorithm is a good choice if you are looking for a simple, fast, and explainable solution. Weighted-session-based k-nn recommendations are close to the state-of-the-art, and we don't need to tune multiple hyperparameters and build complex deep learning models to achieve a good result.

### How does it work?

You provide two input structures as **training** data:

```
sessions : dict
               sessions = {
                   session id: (
                       [sequence of items with user interaction],
                       [timestamp of user interaction per item],
                       [sequence of weighting factors]
                   )
               }

items : dict
        items = {
            item id: (
                [sequence of sessions with an item],
                [the first timestamp of each session with an item]
            )
        }
```

And you ask a model to recommend products based on the user session:

```
user session: {session id: [[sequence of items], [sequence of timestamps]]}
```

The package is lightweight. It depends only on the `numpy` and `pyyaml`. 

Moreover, we can provide a package for non-programmers, and they can use `settings.yaml` to control a model behavior.


### Why should we use WSKNN?

- training is faster than deep learning or XGBoost algorithms, model memorizes map of session-items and item-sessions,
- recommendations are easy to control. We can change how the algorithm works in just a few lines... of text,
- as a baseline, for comparison of deep learning / XGBoost architectures,
- swift prototyping,
- easy to run in production.

The model was created along with multiple other approaches: based on RNN (GRU/LSTM), matrix factorization, and others. Its performance was always very close to the level of fine-tuned neural networks, but it was much easier and faster to train.

### What are the limitations of WSKNN?

- model memorizes session-items and item-sessions maps, and if your product base is large and you use sessions for an extended period, then the model may be too big to fit an available memory; in this case, you can 
categorize products and train a different model for each category,
- response time may be slower than from other models, especially if there are available many sessions,
- there's additional overhead related to the preparation of the input.

### Example


```python

from wsknn import fit
from wsknn.utils import load_pickled

# Load data
ITEMS = 'demo-data/items.pkl'
SESSIONS = 'demo-data/sessions.pkl'

items = load_pickled(ITEMS)
sessions = load_pickled(SESSIONS)

trained_model = fit(sessions, items)

test_session = {'unique id': [
    ['product id 1', 'product id 2'],
    ['timestamp #1', 'timestamp #2']
]}

recommendations = trained_model.predict(test_session, number_of_recommendations=3)
print(recommendations)

```

Output:

```shell
[
 ('product id 3', 0.7),
 ('product id 4', 0.33),
 ('product id 5', 0.059)
]
```


## Setup

Version 0.1 of a package can be installed with `pip`:

```shell
pip install wsknn
```

It works with Python versions greater or equal to 3.6.

## Requirements

| Package Version | Python versions | Other packages |
|-----------------|-----------------|----------------|
| 0.1             | 3.6+            | numpy, yaml    |


## Developers

- Szymon Moliński (Sales Intelligence : Digitree Group SA)

## Citation

Szymon Moliński. (2022). WSKNN - Weighted Session-based k-NN Recommendations in Python (0.1). Zenodo. https://doi.org/10.5281/zenodo.6393177

## Bibliography

### Data used in a demo example

- David Ben-Shimon, Alexander Tsikinovsky, Michael Friedmann, Bracha Shapira, Lior Rokach, and Johannes Hoerle. 2015. RecSys Challenge 2015 and the YOOCHOOSE Dataset. In Proceedings of the 9th ACM Conference on Recommender Systems (RecSys '15). Association for Computing Machinery, New York, NY, USA, 357–358. DOI:https://doi.org/10.1145/2792838.2798723

### Comparison between DL and WSKNN

- Twardowski, B., Zawistowski, P., Zaborowski, S. (2021). Metric Learning for Session-Based Recommendations. In: Hiemstra, D., Moens, MF., Mothe, J., Perego, R., Potthast, M., Sebastiani, F. (eds) Advances in Information Retrieval. ECIR 2021. Lecture Notes in Computer Science(), vol 12656. Springer, Cham. https://doi.org/10.1007/978-3-030-72113-8_43

## Funding

![Funding](./eu_funding_logos/FE_POIR_poziom_engl-1_rgb.jpg)

- Development of the package was partially based on the research project
**E-commerce Shopping Patterns Prediction System** that 
was founded under Priority Axis 1.1 of Smart Growth Operational Programme 2014-2020 for Poland
co-funded by European Regional Development Fund. Project number: `POIR.01.01.01-00-0632/18`