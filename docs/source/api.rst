API
===

Core functions
--------------

fit\_transform
..............

.. autofunction:: wsknn.fit_transform.fit
   :noindex:

predict
.......

.. autofunction:: wsknn.predict.predict
   :noindex:

Core WSKNN class
----------------

WSKNN
.....

.. automodule::  wsknn.model.wsknn
   :members:


Core WSKNN Data Structures
--------------------------

Sessions
........

.. automodule::  wsknn.preprocessing.structure.session.Sessions
   :members:

Items
.....

.. automodule::  wsknn.preprocessing.structure.item.Items
   :members:

Transform session-items map into item-sessions map
..................................................

.. autofunction:: wsknn.preprocessing.structure.session_to_item_map.map_sessions_to_items
   :noindex:

Save and Load model
-------------------

.. autofunction:: wsknn.import_export.ie.save
   :noindex:

.. autofunction:: wsknn.import_export.ie.load
   :noindex:

Evaluation metrics
------------------

.. autofunction:: wsknn.evaluate.metrics.score_model
   :noindex:

.. autofunction:: wsknn.evaluate.metrics.get_mean_reciprocal_rank
   :noindex:

.. autofunction:: wsknn.evaluate.metrics.get_precision
   :noindex:

.. autofunction:: wsknn.evaluate.metrics.get_recall
   :noindex:


Preprocessing
-------------

.. autofunction:: wsknn.preprocessing.parse_static.parse_files
   :noindex:

.. autofunction:: wsknn.preprocessing.parse_static.parse_flat_file
   :noindex:

.. autofunction:: wsknn.preprocessing.static_parsers.parse.parse_fn
   :noindex:

.. autofunction:: wsknn.preprocessing.static_parsers.pandas_parser.parse_pandas
   :noindex:

Utilities
---------

.. automodule::  wsknn.utils.transform
   :members:
   :undoc-members:
   :show-inheritance:
