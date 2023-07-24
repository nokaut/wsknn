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

   import numpy as np
   from wsknn import fit
   from wsknn.utils import load_gzipped_pickle

   # Load data
   ITEMS = 'demo-data/recsys-2015/parsed_items.pkl.gz'
   SESSIONS = 'demo-data/recsys-2015/parsed_sessions.pkl.gz'

   items = load_gzipped_pickle(ITEMS)
   sessions = load_gzipped_pickle(SESSIONS)
   imap = items['map']
   smap = sessions['map']

   # Train model
   trained_model = fit(smap,
                       imap,
                       number_of_recommendations=5,
                       weighting_func='log',
                       return_events_from_session=False)

   # Get sample session
   test_session_key = np.random.choice(list(smap.keys()))
   test_session = smap[test_session_key]
   print(test_session)  # [products], [timestamps]


.. code-block:: shell

   >>> [[214850771, 214677615, 214651777], [1407592501.048, 1407592529.941, 1407592552.98]]


.. code-block:: python

   recommendations = trained_model.recommend(test_session)
   for rec in recommendations:
       print('Item:', rec[0], '| weight:', rec[1])


.. code-block:: shell

   >>> Item: 214676306 | weight: 1.8718411072574241
   >>> Item: 214850758 | weight: 1.2478940715049494
   >>> Item: 214561775 | weight: 1.2478940715049494
   >>> Item: 214821020 | weight: 1.2478940715049494
   >>> Item: 214848322 | weight: 1.2478940715049494

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
