# Settings Info (ENG)

## How to configure Settings.yml file?

In this document we will go through the `settings.yml` file and each variable that should be set before we start modeling. Check this file if you are not sure what each setting means and what values are available for each setting.

## I/O

Input and Output settings. Here we control paths to the input Item/Session Maps.

### `input_items`: (**str**)

String with the **absolute** path to pickled and pre-processed Items Dictionary. Structure of a single Item Map record is:

```python
items = {
        item_id: (
            [sequence_of_sessions],
            [sequence_of_the_first_session_timestamps]
        )
    }
```

File is prepared with the [vsknn-data-parser](https://git.sare25.com/s.molinski/vsknn-data-parser).

#### Example:

```commandline
input_items: "/Users/szymonmolinski/Documents/Gitlab/VSKNN-eksperymenty/lm/items_dict.pkl"
```

### `input_sessions`: (**str**)

String with the **absolute** path to the pickled and pre-processed Sessions Map. A structure of a single record within map is:

```python
sessions = {
        session_id: (
            [sequence_of_items],
            [sequence_of_timestamps],
            [sequence_of_event_type]
        )
    }
```

File is prepared with the [vsknn-data-parser](https://git.sare25.com/s.molinski/vsknn-data-parser).

#### Example:

```commandline
input_sessions: /Users/szymonmolinski/Documents/Gitlab/VSKNN-eksperymenty/lm/sessions_dict.pkl
```

## Model

Here we set model parameters and control the behavior of an algorithm. For all variables we can pass multiple arguments as a list to test different scenarios. It is possible because model that has been fit once may work differently with a different set of parameters.
The parent structure here is the `model` key:

```commandline
model:
  var1: ...
  var2: ...
  ...
```

We operate on variables within this `model` key. We can pass a single item for each key below.

### Number of recommendations

How many recommendations should model return? It is the most important setting. Recommendations are returned in a **decreasing** order so we may set this value large and take only a few of recommendations.

#### Example:

Return 10 most similar items:

```commandline
model:
  number_of_recommendations: 10
```

## Session neighbors parameters
### Number of closest neighbors

How many neighbors are sampled from the sessions space. Large value may slow down calculations but allows for more diverese results. Algorithm takes initially all known sessions and we limit this number by this parameter. How the neighboring sessions should be picked is set by the next parameter `session_sampling_strategy`. 

#### Example:

Get 1000 neighbors to rank items:

```commandline
model:
  number_of_closest_neighbors: 1000
```

### Session Sampling Strategy

Neighboring sessions are sampled based on the three strategies:
- **random**: select random subset of sessions,
- **recent**: select most recent sessions,
- **common_items**: select sessions that contains any item from the recommended session.
  
#### Example:

Select sessions with the same items as the user session:

```commandline
model:
  session_sampling_strategy: "common_items"
```

### Possible Neighbors Sample Size

How many neighbors are picked with the Session Sampling Strategy

### Example:

```commandline
model:
  possible_neighbors_sample_size: 1000
```

## Item Ranking Parameters

Now we have our pool of session. It is time for item ranking. It is a two-step process. In the first step we **may* weight each neighbor session items. **Weighting at this step uses information about the common item position within the neighbor session and the length of the neighbor session**. In practice, all weights are products of the equations simplified equation: `c * pos * length` where:
- `c` is some factor depended on the weighting style,
- `pos` is an item position within the neighboring session, the most recent items have the largest weights,
- `length` is a neighboring sequence length. Long sequences of events (item interactions) may penalize results. We assume that very long sequences are usually not-reliable (>100 items). **It is a hypothesis and it hasn't be tested!**

### Weight Session Items

Should items be weighted if we are interested in the similarity between the user session and neighboring sessions? 

The easiest thing is to not weight items at all, and we assume that **if the item occurs in the user session and in the neighbor session then it is enough: the similarity is set 1**.

If we set weighting strategy to any available value then our session-items will be weighted.

### Weighting Strategy

Weighting strategies:
- **linear**: basic linear weighting, distances between events in a sequence are equal and depended only on the position of element in a sequence, always gives larger weights than the other methods,
- **log**: oldest elements starts from the larger weights than in the other methods. Then function returns very similar weights, from approx. 20-90% of the sequence and the newest 90-100% of the events sharply rose. It is good to mimic a short-term memory.
- **quadratic**: the oldest elements are penalized much more than the middle-part and the newest elements in a sequence. Function rises quicker than the log10 function but values are smaller than returned from the linear function. The newest observation -> larger weight assigned to it.

#### Example:

```commandline
model:
  weight_session_items: True
  weighting_strategy: "log"
  
```

### Rank Strategy

**Items ranking:**

Each item from the user session is "weighted" to set the final item rank. First, we iterate from the newest items to the oldest in the user session. We check if any item has occured in the neighbor session. If it does, then based on its number from the end of a session (the newest items have the lowest numbers) we calculate decay for this item.

**Decay** is a factor by which final rank is multiplied and it should be between 0 and 1. We do it ONLY for the first occurence of the newest possible event! Then we update score in the items dict to get the final result.
  
#### Available Ranking strategies:
- linear:    simple linear function - up to 10 positions it is 1 - 0.1 * position normalized to [0:1], and 0 if the position is greater than 10.
- inv:       inverted position of the item: 1/position within (0:1] limits.
- log:       `1 / log10(position + 1.7)` normalized to the range [0:1].
- quadratic: `1 / (position**2)` - events closer to the end of sequence have larger weights than the past events, it is similar to linear function but weights are decreasing at larger and non-linear rate.

#### Example:  
```commandline
model
  rank_strategy: "quadratic"
```