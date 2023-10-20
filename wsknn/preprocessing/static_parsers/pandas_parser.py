from typing import Dict, List
import pandas as pd
from wsknn.preprocessing.structure.session_to_item_map import map_sessions_to_items


def _clip_multiple_transactions(session, items, times, actions, locs, min_session_length):
    loclen = len(items) - 1

    parsed = []
    pidx = None

    for idx, lc in enumerate(locs):
        if lc == 0:
            continue
        else:
            if idx == 0:
                previous = 0
            else:
                previous = locs[idx-1] + 1

            if lc + 1 < loclen:
                lc = lc + 1

            nitem = items[previous:lc]
            ntime = times[previous:lc]
            nact = actions[previous:lc]
            if len(nitem) >= min_session_length:
                parsed.append(
                    [session, nitem, ntime, nact]
                )
        pidx = idx

    if pidx < loclen:
        nitem = items[pidx:]
        ntime = times[pidx:]
        nact = actions[pidx:]
        if len(nitem) >= min_session_length:
            parsed.append(
                [session, nitem, ntime, nact]
            )

    return parsed


def _parse_full_ds(sess_idx, items, times, actions, is_transaction, weights, min_session_length):
    if is_transaction is None:
        # Skip dividing sessions and put everything in one sequence
        if weights is None:
            parsed = {
                x: [items[idx], times[idx], actions[idx]]
                for idx, x in enumerate(sess_idx) if len(items[idx]) >= min_session_length}
        else:
            parsed = {
                x: [items[idx], times[idx], actions[idx], weights[idx]]
                for idx, x in enumerate(sess_idx) if
                len(items[idx]) >= min_session_length}
    else:
        parsed = {}
        for idx, x in enumerate(is_transaction):
            if 1 in x:
                if len(items[idx]) > min_session_length:
                    if weights is None:
                        parsed[sess_idx[idx]] = [items[idx], times[idx], actions[idx]]
                    else:
                        parsed[sess_idx[idx]] = [items[idx], times[idx],
                                                 actions[idx], weights[idx]]
            # TODO: complex parsing schema, where single session may occur multiple times! FUTURE
            #     # We have a transaction here, time to divide sequence
            #     t_idx = list(locate(x))
            #     # Divide sequences based on the transaction indices
            #     parsed.extend(
            #         _clip_multiple_transactions(
            #             session=sess_idx[idx], items=items[idx], times=times[idx], actions=actions[idx], locs=t_idx,
            #             min_session_length=min_session_length
            #         )
            #     )
            # else:
            #     if len(items[idx]) >= min_session_length:
            #         parsed.append(
            #             {sess_idx[idx]: [items[idx], times[idx], actions[idx]]}
            #         )
    return parsed


def _prepare_values_session_map(df: pd.DataFrame,
                                session_id_key: str,
                                product_key: str,
                                time_key: str,
                                action_key: str = None,
                                purchase_action_name: str = None,
                                event_weights_key: str = None):
    df = df.sort_values([session_id_key, time_key])
    gdf = df.groupby(session_id_key)
    timestamps = gdf[time_key].apply(list).values
    products = gdf[product_key].apply(list)
    actions = None
    transactions = None
    event_weights = None
    if action_key is not None:
        actions = gdf[action_key].apply(list).values
    if purchase_action_name is not None:
        transactions = gdf['is_transaction'].apply(list).values
    if event_weights_key is not None:
        event_weights = gdf[event_weights_key].apply(list).values

    return products.index, products.values, timestamps, actions, transactions, event_weights


def _build_maps_from_df(df: pd.DataFrame,
                        session_id_key: str,
                        product_key: str,
                        time_key: str,
                        action_key: str = None,
                        purchase_action_name: str = None,
                        event_weights_key: str = None,
                        min_session_length: int = 3) -> Dict:
    # Prepare data for session map
    sess_idx, items, times, actions, is_transaction, weights = _prepare_values_session_map(
        df,
        session_id_key,
        product_key,
        time_key,
        action_key,
        purchase_action_name,
        event_weights_key
    )

    if action_key is None:
        if event_weights_key is None:
            level_1_parsed = {
                x: [items[idx], times[idx]]
                for idx, x in enumerate(sess_idx) if len(items[idx]) >= min_session_length
            }
        else:
            level_1_parsed = {
                x: [items[idx], times[idx], weights[idx]]
                for idx, x in enumerate(sess_idx) if
                len(items[idx]) >= min_session_length
            }
    else:
        level_1_parsed = _parse_full_ds(sess_idx, items, times, actions, is_transaction, weights, min_session_length)

    return level_1_parsed


def parse_pandas(df: pd.DataFrame,
                 session_id_key: str,
                 product_key: str,
                 time_key: str,
                 action_key: str = None,
                 allowed_actions: List = None,
                 purchase_action_name: str = None,
                 event_weights_key: str = None,
                 min_session_length: int = 3,
                 get_items_map: bool = True) -> Dict:
    """
    Function parses given dataset into Sessions and Items objects.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with events and sessions.

    session_id_key : str
        The name of the session key.

    product_key : str
        The name of the product key.

    time_key : str
        The name of the event timestamp key.

    action_key : str, default = None
        The name of the event action type key.

    allowed_actions : List, optional
        Allowed actions.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into
        the session vector).

    event_weights_key : str, optional
        The name of weights column.

    min_session_length : int, default = 3
        Minimum length of a single session.

    get_items_map : bool, default = True
        Should item-sessions map be created?

    Returns
    -------
    : Dict
        {"session-map": Dict, "item-map": Optional[Dict]}
    """

    # Clean dataframe
    if action_key is not None:
        # Check allowed actions
        if allowed_actions is not None:
            # Select rows with allowed actions
            df = df[df[action_key].isin(allowed_actions)]
        # Check purchase action
        if purchase_action_name is not None:
            df['is_transaction'] = (
                    df[action_key] == purchase_action_name
            ).astype(int)

    # Prepare maps
    sess_map = _build_maps_from_df(
        df,
        session_id_key,
        product_key,
        time_key,
        action_key,
        purchase_action_name,
        event_weights_key,
        min_session_length
    )

    output = {'session-map': sess_map}

    if get_items_map:
        items_map = map_sessions_to_items(sessions_map=sess_map)
        output['item-map'] = items_map

    return output
