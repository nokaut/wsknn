Introduction
============

Does this data structure look familiar to you?

.. code-block:: python

    web_event = {
        'user_id': 'xyz',
        'event_type': 'click',
        'time': '2000-01-01 12:00:00',
        'session_id': 'a12',
        'product_id': 'c-49'
    }

This is a single event that can be traced from web services and mobile devices. Every field in this dictionary is important. We can group multiple objects of this type into a **sequence** or a **session** that contains:

- **multiple events**,
- **users' interactions with products**,
- **paths - event sequences - that leads to the transaction**,
- **timestamped actions of a user**.

We can use a set of a product interaction sequences to create a recommendation engine. A recommendation system analyzes sessions, and builds a mappings between products that are grouped with other products within users sessions. We can retrieve information about items, that will be most likely viewed or bought by the users. Finally, we can recommend products that are optimized for a user session (past interactions with products).

How does it work?
-----------------

We provide two input structures as a **training** data:

**SESSIONS**:

.. code-block:: none

               sessions = {
                   session id: (
                       [sequence of items with user interaction],
                       [timestamp of user interaction per item],
                       [(optional) sequence of event names],
                       [(optional) sequence of weights]
                   )
               }

**ITEMS / PRODUCTS**:

.. code-block:: none

        items = {
            item id: (
                [sequence of sessions with an item],
                [the first timestamp of each session with an item]
            )
        }

And we may ask a model to recommend products based on user session:

**USER SESSION**

.. code-block:: none

    {session id:
        [[sequence of items], [sequence of timestamps], [optional event names], [optional weights]]
    }

The package is lightweight. It depends only on the ``numpy`` and ``pyyaml``.

Moreover, we can provide a package for non-programmers, and they can use ``settings.yaml`` to control a model behavior.


Why should we use WSKNN?
------------------------

- training is **faster** than deep learning or XGBoost algorithms, model memorizes map of session-items and item-sessions,
- recommendations are **easy to control**. We can change how the algorithm works with s few settings,
- as a **baseline**, for comparison of deep learning / XGBoost architectures,
- **fast prototyping**,
- easy to **run in production**.

We have developed and tested this model along with other techniques: based on *RNN* (*GRU/LSTM*), *matrix factorization*, and custom deep learning architectures. A performance of ``WSKNN`` model was always very close to the level of a fine-tuned and custom neural network. On the other hand, this algorithm is much easier to build, control, understand, and run in production.

What are the limitations of WSKNN?
----------------------------------

- model *memorizes session-items and item-sessions maps*, and if your product base is large and you use sessions for an extended period, then the model may be too big to fit an available memory; in this case, you can categorize products and train a different model for each category,
- *response time may be slower than from other models*, especially if there are available many sessions,
- there's *additional overhead related to the preparation of the input*. But this is related to the every other model, except simple Markov Models.

Setup
-----

The package can be installed with ``pip``:

.. code-block:: bash

    pip install wsknn


It works with Python versions greater or equal to 3.6.

Requirements
------------

+-----------------+-----------------+-------------------------------------------+
| Package Version | Python versions | Requirements                              |
+=================+=================+===========================================+
| >=0.1           | 3.6+            | ``numpy``, ``pyyaml``                     |
| 1.x+            | 3.8+            | ``numpy``, ``pyyaml``, ``more_itertools`` |
+-----------------+-----------------+-------------------------------------------+

Developers
----------

- Szymon Moliński (Sales Intelligence : Digitree Group SA), Github: @SimonMolinsky

Citation
--------

Szymon Moliński. (2022). WSKNN - Weighted Session-based k-NN Recommendations in Python (0.1). Zenodo. https://doi.org/10.5281/zenodo.6393177

Bibliography
------------

Data used in a demo example
...........................

- David Ben-Shimon, Alexander Tsikinovsky, Michael Friedmann, Bracha Shapira, Lior Rokach, and Johannes Hoerle. 2015. RecSys Challenge 2015 and the YOOCHOOSE Dataset. In Proceedings of the 9th ACM Conference on Recommender Systems (RecSys '15). Association for Computing Machinery, New York, NY, USA, 357–358. DOI:https://doi.org/10.1145/2792838.2798723

Comparison between DL and WSKNN
...............................

- Twardowski, B., Zawistowski, P., Zaborowski, S. (2021). Metric Learning for Session-Based Recommendations. In: Hiemstra, D., Moens, MF., Mothe, J., Perego, R., Potthast, M., Sebastiani, F. (eds) Advances in Information Retrieval. ECIR 2021. Lecture Notes in Computer Science(), vol 12656. Springer, Cham. https://doi.org/10.1007/978-3-030-72113-8_43


Benchmarking
------------

As a rule of thumb you should assume that you should have ~2 times more memory available than your model's memory size

- Used machine has 16GB RAM
- testing sample size - 1000 sessions
- max session length - 30 events
- min session length - 1 event
- basic data types (integers)

Session length vs response time vs model size
.............................................

+---+----------+--------+--------------------+----------------------+
|   | sessions | items  | mean response time | model memory size MB |
+===+==========+========+====================+======================+
| 0 | 100000   | 100000 | 0.00501535         | 278                  |
| 1 | 200000   | 100000 | 0.00707721         | 524                  |
| 2 | 300000   | 100000 | 0.00528198         | 769                  |
| 3 | 400000   | 100000 | 0.00546341         | 1018                 |
| 4 | 500000   | 100000 | 0.00569851         | 1264                 |
| 5 | 600000   | 100000 | 0.00591904         | 1505                 |
| 6 | 700000   | 100000 | 0.00529248         | 1764                 |
| 7 | 800000   | 100000 | 0.00524046         | 2010                 |
| 8 | 900000   | 100000 | 0.00543461         | 2250                 |
| 9 | 1e+06    | 100000 | 0.00673801         | 2495                 |
+---+----------+---------+-------------------+----------------------+


Number of items vs response time vs model size
..............................................

+----+------------+---------+----------------------+------------------------+
|    |   sessions |   items |   mean response time |   model memory size MB |
+====+============+=========+======================+========================+
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
+----+------------+---------+----------------------+------------------------+

Funding
-------

..  image:: ../../eu_funding_logos/FE_POIR_poziom_engl-1_rgb.jpg
    :width: 400
    :alt: Funding Bodies logos


Development of the package was partially based on the research project
**E-commerce Shopping Patterns Prediction System** that
was founded under Priority Axis 1.1 of Smart Growth Operational Programme 2014-2020 for Poland
co-funded by European Regional Development Fund. Project number: `POIR.01.01.01-00-0632/18`