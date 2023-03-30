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
                       [(optional) sequence of event names],
                       [(optional) sequence of weights]
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
user session: 
    {session id:
        [[sequence of items], [sequence of timestamps], [optional event names], [optional weights]]
    }
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

trained_model = fit(sessions, items, number_of_recommendations=3)

test_session = {'unique id': [
    ['product id 1', 'product id 2'],
    ['timestamp #1', 'timestamp #2']
]}

recommendations = trained_model.recommend(test_session)
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

## Preprocessing Stage



## Setup

Version 1.x of a package can be installed with `pip`:

```shell
pip install wsknn
```

It works with Python versions greater or equal to 3.8.

## Requirements

| Package Version | Python versions | Requirements                  |
|-----------------|-----------------|-------------------------------|
| 0.1.x           | 3.6+            | numpy, pyyaml                 |
 | 1.x             | 3.8+            | numpy, pyyaml, more_itertools |

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

## Benchmarking

As a rule of thumb you should assume that you should have ~2 times more memory available than your model's memory size

- Used machine has 16GB RAM
- testing sample size - 1000 sessions
- max session length - 30 events
- min session length - 1 event
- basic data types (integers)

### Session length vs response time vs model size

|    |   sessions |   items |   mean response time |   model memory size MB |
|---:|-----------:|--------:|---------------------:|-----------------------:|
|  0 | 100000     |  100000 |           0.00501535 |                    278 |
|  1 | 200000     |  100000 |           0.00707721 |                    524 |
|  2 | 300000     |  100000 |           0.00528198 |                    769 |
|  3 | 400000     |  100000 |           0.00546341 |                   1018 |
|  4 | 500000     |  100000 |           0.00569851 |                   1264 |
|  5 | 600000     |  100000 |           0.00591904 |                   1505 |
|  6 | 700000     |  100000 |           0.00529248 |                   1764 |
|  7 | 800000     |  100000 |           0.00524046 |                   2010 |
|  8 | 900000     |  100000 |           0.00543461 |                   2250 |
|  9 |      1e+06 |  100000 |           0.00673801 |                   2495 |

### Number of items vs response time vs model size

|    |   sessions |   items |   mean response time |   model memory size MB |
|---:|-----------:|--------:|---------------------:|-----------------------:|
|  0 |     100000 |    1000 |          1.3833e-05  |                    235 |
|  1 |     100000 |   11000 |          7.047e-05   |                    250 |
|  2 |     100000 |   21000 |          0.000135771 |                    252 |
|  3 |     100000 |   31000 |          0.000257456 |                    256 |
|  4 |     100000 |   41000 |          0.000462458 |                    259 |
|  5 |     100000 |   51000 |          0.000775981 |                    262 |
|  6 |     100000 |   61000 |          0.00136349  |                    265 |
|  7 |     100000 |   71000 |          0.00211188  |                    268 |
|  8 |     100000 |   81000 |          0.00297504  |                    271 |
|  9 |     100000 |   91000 |          0.0038164   |                    276 |
| 10 |     100000 |  101000 |          0.00490628  |                    278 |
| 11 |     100000 |  111000 |          0.00580347  |                    281 |
