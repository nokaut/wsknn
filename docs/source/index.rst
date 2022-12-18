.. wsknn documentation master file, created by
   sphinx-quickstart on Wed Apr 13 10:28:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Weighted session-based k-NN - Intro
===================================

Do you build a **recommender system** for your website? The K-nearest neighbors algorithm is a good choice if you are looking for a simple, fast, and explainable solution. Weighted-session-based k-nn recommendations are close to the state-of-the-art methods. We don't need to tune multiple hyperparameters and build complex deep learning models to achieve a good result.

Example
-------

**Input**:

.. code-block:: python

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

   recommendations = trained_model.recommend(test_session, number_of_recommendations=3)
   print(recommendations)

**Output**:

.. code-block:: shell

   >>> [
   ...  ('product id 3', 0.7),
   ...  ('product id 4', 0.33),
   ...  ('product id 5', 0.059)
   ... ]

Contents
--------

.. toctree::
   :maxdepth: 1

   introduction
   api
   contribution


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
