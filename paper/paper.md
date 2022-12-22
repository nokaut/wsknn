---
title: 'WSKNN - Weighted Session-based K-NN recommender system'
tags:
  - Python
  - machine learning
  - e-commerce
  - recommender system
  - recommender engine
authors:
  - name: Szymon Moliński
    orcid: 0000-0003-3525-2104
    affiliation: 1
affiliations:
 - name: Digitree SA, Poland
   index: 1
date: 21 December 2022
bibliography: paper.bib

---

# Summary

The users of e-commerce systems generate huge amounts of unstructured, sequential data streams. Each sequence is a varying-length list of directional (timestamped) user-product interactions. There are hidden patterns within those sequences, users tend to interact with similar products, and based on this behavior, we can recommend the next n-products that the user may be interested in.

The `wsknn` package is a lightweight tool for modeling user-item interactions and making recommendations from sequential datasets. It is based on the k-Nearest Neighbors algorithm prepared to work with any categorical, sequential, and timestamped data, especially this generated within e-commerce systems. The package may be a stand-alone recommender, a reference against the more complex recommender systems, or a part of a bigger Machine Learning pipeline.

# Statement of need

`Wsknn` is an abbreviation from *Weighted Session-based K-NN recommender*. The package utilizes k-Nearest Neighbors algorithm that works with loosely structured data in the form of sequences with different lengths. This type of data is the most common representation of the timestamped events stream from a customer. The simple session example is shown below:

```json

{ "user xyz": [
    ["item a", "item b", "item h", "item n"],
    ["2022-01-01 09:00:00", "2022-01-01 09:03:12", "2022-01-01 09:03:30", "2022-01-01 10:43:56"]
  ]
}

```

It is the *session-based* part of the package’s name. What is weighted, and why? The user-actions sequence carries on more information than only products that customer has interacted with. In the most common settings, a sequence per user has:

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

Clearly, there is a lot more information that can be used than only products' sets. The session-based k-NN algorithm builds mapping between products that occur within the same session, but in some settings, it could be not enough. The recommender should include other factors too, and they might be derived from the session above:

-	The position of a product in a sequence,
-	The length of a sequence,
-	The recency of a sequence,
-	A specific action type in a sequence (for example: *transaction*),
-	Or custom weights applied to the sequence, for example products’ prices.

`Wsknn` uses all this information to prepare a valid recommendation.

# Related work

The `wsknn` recommender was not designed to replace all available session-based recommender systems and architectures. The basic design was to evaluate a complex deep-learning based model [BIB]. Models evaluation has shown that Mean Reciprocal Rank and Recall at n-recommendations are very close between the *simple* weighted session-based k-NN and thoroughly tuned deep learning models that can be consider as the state-of-the-art. Moreover, the analysis of the literature about recommender systems shows that the k-NN based solutions are performing well in a different conditions [BIB]. It makes `wsknn` a great benchmarking tool against the novel algorithms and architectures.

The systems can work in a cold-start scenario, and as a recommender for a small and medium sized datasets. The example is 

# Package structure

# Case study

# Acknowledgements

# References

# Funding
