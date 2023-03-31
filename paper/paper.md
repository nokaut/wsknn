---
title: WSKNN - Weighted Session-based K-NN recommender system
tags:
 - Python
 - machine learning
 - e-commerce
 - recommender system
 - recommender engine
authors:
 - name: Szymon Moliński
   orcid: 0000-0003-3525-2104
   affiliation: 1,2
affiliations:
 - name: Sales Intelligence
   index: 1
 - name: Digitree SA
   index: 2
date: 31 March 2023
bibliography: paper.bib

---

# Summary

The users of e-commerce systems generate huge amounts of unstructured, sequential data streams. Each sequence is a varying-length list of directional (timestamped) user-product interactions. There are hidden patterns within those sequences, users tend to interact with similar products, and based on this behavior, we can recommend the next n-products that the user may be interested in.

The `wsknn` package is a lightweight tool for modeling user-item interactions and making recommendations from sequential datasets. It is based on the k-Nearest Neighbors algorithm prepared to work with any categorical, sequential, and timestamped data, especially this generated within e-commerce systems. The package may be a stand-alone recommender, a reference against the more complex recommender systems, or a part of a bigger Machine Learning pipeline.

# Statement of need

`Wsknn` is an abbreviation from *Weighted Session-based K-NN recommender*, but algorithm itself is tuned and enhanced **Vector Multiplication Session-Based kNN (V-SKNN)** [@Ludewig2018]. The package utilizes the k-Nearest Neighbors algorithm that works with loosely structured data in the form of sequences with different lengths. This type of data is the most common representation of the timestamped events stream from a customer. The simple session example is shown below:

```json

{ "user xyz": [
   ["item a", "item b", "item h", "item n"],
   ["2022-01-01 09:00:00", "2022-01-01 09:03:12", "2022-01-01 09:03:30", "2022-01-01 10:43:56"]
 ]
}

```

It is the *session-based* part of the package’s name. What is weighted, and why? The user-actions sequence carries on more information than only products that the customer has interacted with. In the most common settings, a sequence per user has:

- Products (items),
- Timestamps,
- Event types,
- Transaction values.

The input may become more complex in this setting:

```json

{ "user xyz": [
   ["item a", "item b", "item h", "item n", "none"],
   ["2022-01-01 09:00:00", "2022-01-01 09:03:12", "2022-01-01 09:03:30", "2022-01-01 10:43:56", "2022-01-01 10:44:21"],
   ["view", "view", "add to cart", "add to cart", "transaction"],
   [0, 0, 0, 0, 230.87]
 ]
}

```

Clearly, there is a lot more information that can be used than only products' sets. The session-based k-NN algorithm builds a mapping between products that occur within the same session, but in some settings, it could not be enough. The recommender should include other factors too, and they might be derived from the session above:

-  The position of a product in a sequence,
-  The length of a sequence,
-  The recency of a sequence,
-  A specific action type in a sequence (for example: *transaction*),
-  Or custom weights applied to the sequence, for example, products’ prices.

`Wsknn` uses all this information to prepare a valid recommendation.

The package is related to past research projects within a company [@Twardowski2021] and current operations for large customers. In the closest future package will be enhanced with `tensorflow` version of the algorithm.

# Related work

The `wsknn` recommender was designed to evaluate complex deep-learning-based architectures [@Twardowski2021], but during the research, it became clear that the k-NN model's performance is close to or exceeds the performance of the neural nets algorithms. Moreover, the analysis of the literature about recommender systems shows that the k-NN-based solutions are performing well in different conditions [@Ludewig2018]. It makes `wsknn` a great benchmarking tool against novel algorithms and architectures, and the first-choice tool for the fresh start and design of the recommender system.

The package's algorithm can work in a cold-start scenario, and as a recommender for small and medium-sized datasets. During our own studies, the algorithm performed well for the small datasets (25k sessions; 3k items) and bigger datasets, but it has memory size limitations. As a memory-based model, it can grow up to the moment, when its usage is unfeasible. It could be an issue for production environments where the costs may exceed potential benefits. On the other hand, k-NN based approach may be placed in the bigger pipeline, where the large-space model is based on the neural network architecture, but the preliminary selection of recommendations can be done with the `wsknn` package. Especially powerful is using weights to control how the model chooses neighbors, how important items are in a session, and which actions bring the highest value to a recommendation.

A similar architecture can be found in a stand-alone repository [@recsystemsrepo] that seems to be not actively maintained and is linked to specific publications [@Latifi2020SessionawareRA]. The main difference between `wsknn` and the *V-SKNN* model from the presented repository is that the latter is a ready-to-use package. The analytical differences are related to the more ways of session-weighting within `wsknn` up to a point, where custom heuristics can be applied to the recommendations. The other example of a stand-alone repository is [@gru4recrepo] with Keras implementation of **Gru4Rec** session-based recommender [@Hidasi2015SessionbasedRW].

# Package structure

The package is lightweight, it depends only on the `numpy`, `more_itertools` and `pyyaml` packages. It works with every currently supported Python version. It has two main functions: `fit()` to build a memory representation of a model, and `predict()` to return recommendations. What is worth noticing is that **the recommendation strategy may be altered after fitting a model**; it allows testing different scenarios in parallel.

The users may control:

- the number of recommendations,
- the number of neighbors to choose items from (**the closest neighbors**),
- the sampling strategy of neighbors (common items, recent sessions, random subset, custom weights assigned to events' type),
- the sample size (an initial subset of neighbors to look for the closest neighbors),
- a session similarity weighting function,
- an item ranking strategy,
- additional parameters:
 - should algorithm return items that are in the recommended session?
 - is there any event (user action) that must be performed to build a similarity map (for example, the *transaction* event)?
 - additional sampling strategy weights,
 - should the algorithm recommend random items if the neighbors-items-set is smaller than a number of recommendations?

The sample flow and recommendations are presented in the repository [@wsknnrepo]. The package has built-in evaluation metrics:

- the **mean reciprocal rank** of top `k` recommendations,
- the **precision** score of top `k` recommendations,
- the **recall** score of top `k` recommendations.

The package can process static JSON-lines and gzipped JSON-lines files with e-commerce events with its `preprocessing` module. The module takes into account the fact that different event actions have their specific weights, and the session that ends with a purchase should be weighted as more reliable than session that ended without a transaction.
The basic data types, `Items` and `Sessions` that are storing item-sessions and session-items mappings may be updated sequentially, what allows to process a large amounts of data that comes from the e-commerce websites.

# Limitations

As with every Machine Learning system, WSKNN has its limitations:

- model *memorizes session-items and item-sessions maps*, and if the product base is large, and we use sessions for an extended period, then the model may be too big to fit an available memory; in this case, we can categorize products and train a different model for each category. Benchmarking shows that model memory size is directly related to the number of sessions.
- response time may be slower than from other models, especially if there are available many items to recommend, benchmarking shows that the mean response time rises with the number of items used for training,
- there’s additional overhead related to the preparation of the input. But this is related to the every other model, except simple Markov Models. That's why `wsknn` has a built-in `preprocessing` module that transforms common events structure into the model's format.

# Acknowledgements

Development of the package was partially based on the research project E-commerce Shopping Patterns Prediction System that was founded under Priority Axis 1.1 of Smart Growth Operational Programme 2014-2020 for Poland, co-funded by European Regional Development Fund. Project number: POIR.01.01.01-00-0632/18

# References
