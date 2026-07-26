"""
Graph-Based Detection Model  (Deliverable #3)
================================================

The spec asks for a sequence-aware detection approach and names three
options: "LSTM/GRU, Transformer, or graph-based for entity-resource
relationships". This module implements the **graph-based** option.

Rationale for choosing graph over a recurrent net:
  - Lateral movement -- the hardest class in this dataset -- is defined by
    *relationships*: an entity reaching resources that entities like it never
    touch. That is a property of the entity-resource graph, not of a single
    entity's time series.
  - It adds no heavy ML framework to the dependency chain, so the whole
    project stays reproducible with `pip install -r requirements.txt`.
  - Graph features are directly interpretable, which the explainability
    deliverable requires (a SHAP value on "peer affinity" means something to
    an analyst; a hidden LSTM activation does not).

Construction (strictly causal -- no leakage):
  A weighted bipartite graph  entity <--> resource  is built from the
  TRAINING window only. Test-window sessions are then scored against that
  graph, so nothing from the future informs the features.

Features produced
-----------------
  graph_edge_weight        share of the entity's training accesses that went
                           to this resource (0.0 = never touched it before)
  resource_popularity      fraction of all entities that use this resource
                           (rare resources are inherently more suspicious)
  peer_affinity            Jaccard overlap between the entity's own resource
                           set and the resource sets of the entities that use
                           this resource -- "do entities like me use this?"
  two_hop_reachable        1 if the resource is reachable from the entity via
                           entity -> resource -> peer entity -> resource,
                           i.e. inside its co-access community
  graph_anomaly_score      standalone graph-only anomaly score, usable as an
                           independent detector (evaluated below via AUC)
"""

import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURED = f"{_PROJECT_ROOT}/data/access_logs_featured.csv"


def build_bipartite_graph(train_df):
    """entity <--> resource graph from benign TRAINING traffic only."""
    g = nx.Graph()
    benign = train_df[train_df["label"] == "normal"]
    counts = benign.groupby(["entity_id", "resource_accessed"]).size()
    for (eid, res), w in counts.items():
        g.add_node(eid, bipartite=0)
        g.add_node(res, bipartite=1)
        g.add_edge(eid, res, weight=int(w))
    return g


def compute_graph_features(df, g, entity_resources, resource_entities,
                           entity_totals, n_entities):
    """Vectorised-per-row graph lookups (dict-based, so this stays fast)."""
    edge_w, res_pop, affinity, two_hop = [], [], [], []

    # precompute peer sets: entities sharing >=1 resource with each entity
    peer_cache = {}

    for eid, res in zip(df["entity_id"].values, df["resource_accessed"].values):
        own = entity_resources.get(eid, frozenset())
        users = resource_entities.get(res, frozenset())

        # 1. edge weight, normalised by the entity's total training activity
        w = g[eid][res]["weight"] if (g.has_node(eid) and g.has_node(res)
                                      and g.has_edge(eid, res)) else 0
        total = entity_totals.get(eid, 0)
        edge_w.append(w / total if total else 0.0)

        # 2. resource popularity across the entity population
        res_pop.append(len(users) / n_entities if n_entities else 0.0)

        # 3. peer affinity: Jaccard(own resource set, union of users' sets)
        if users:
            if eid not in peer_cache:
                peers = set()
                for r in own:
                    peers |= resource_entities.get(r, frozenset())
                peers.discard(eid)
                peer_cache[eid] = peers
            peers = peer_cache[eid]
            inter = len(users & peers)
            union = len(users | peers)
            affinity.append(inter / union if union else 0.0)
        else:
            affinity.append(0.0)

        # 4. two-hop reachability inside the co-access community
        if res in own:
            two_hop.append(1)
        else:
            reachable = 0
            peers = peer_cache.get(eid)
            if peers is None:
                peers = set()
                for r in own:
                    peers |= resource_entities.get(r, frozenset())
                peers.discard(eid)
                peer_cache[eid] = peers
            if users & peers:
                reachable = 1
            two_hop.append(reachable)

    return (np.array(edge_w), np.array(res_pop),
            np.array(affinity), np.array(two_hop))


def main():
    df = pd.read_csv(FEATURED, parse_dates=["timestamp"]).sort_values("timestamp")
    split_ts = df["timestamp"].quantile(0.75)
    train_df = df[df["timestamp"] <= split_ts]

    print("Building bipartite entity-resource graph from TRAINING window only...")
    g = build_bipartite_graph(train_df)
    n_ent = sum(1 for n, d in g.nodes(data=True) if d.get("bipartite") == 0)
    n_res = sum(1 for n, d in g.nodes(data=True) if d.get("bipartite") == 1)
    print(f"  nodes: {n_ent} entities + {n_res} resources | edges: {g.number_of_edges()}")

    benign = train_df[train_df["label"] == "normal"]
    entity_resources = {e: frozenset(s) for e, s in
                        benign.groupby("entity_id")["resource_accessed"].apply(set).items()}
    resource_entities = {r: frozenset(s) for r, s in
                         benign.groupby("resource_accessed")["entity_id"].apply(set).items()}
    entity_totals = benign.groupby("entity_id").size().to_dict()

    ew, rp, aff, th = compute_graph_features(
        df, g, entity_resources, resource_entities, entity_totals, n_ent)

    df["graph_edge_weight"] = ew
    df["resource_popularity"] = rp
    df["peer_affinity"] = aff
    df["two_hop_reachable"] = th

    # ------------------------------------------------------------------
    # COLD-START RECONCILIATION (important, and easy to get wrong)
    # ------------------------------------------------------------------
    # An entity that did not exist when the graph was built has NO edges, so
    # naive graph novelty rates it maximally anomalous -- a brand-new employee
    # would out-score real lateral movement. That directly contradicts the
    # cold-start requirement. Entities absent from the training graph are
    # therefore given population-median graph features (the same population
    # fallback policy the baseline profiler uses); the separate
    # is_cold_start_entity flag still tells the model they are new, so the
    # information is preserved without the false alarm.
    known = df["entity_id"].isin(entity_resources.keys())
    med = {c: float(df.loc[known, c].median())
           for c in ["graph_edge_weight", "peer_affinity"]}
    df.loc[~known, "graph_edge_weight"] = med["graph_edge_weight"]
    df.loc[~known, "peer_affinity"] = med["peer_affinity"]
    df.loc[~known, "two_hop_reachable"] = 1          # neutral, not suspicious
    print(f"  cold-start rows given population-median graph features: {(~known).sum()}")

    # Standalone graph-only anomaly score: unfamiliar edge + outside community
    # + peers don't use it + resource is rare.
    df["graph_anomaly_score"] = (
        2.5 * (df["graph_edge_weight"] == 0).astype(float)
        + 2.0 * (1 - df["two_hop_reachable"])
        + 1.5 * (1 - df["peer_affinity"])
        + 1.0 * (1 - df["resource_popularity"])
    )

    # Evaluate the graph detector ON ITS OWN over the held-out window
    test = df[df["timestamp"] > split_ts]
    y = (test["label"] != "normal").astype(int)
    auc_all = roc_auc_score(y, test["graph_anomaly_score"])

    lm = test[test["label"].isin(["normal", "anomaly_lateral_movement"])]
    auc_lm = roc_auc_score((lm["label"] != "normal").astype(int),
                           lm["graph_anomaly_score"])

    print(f"\nGraph-only detector AUC (all anomalies)      : {auc_all:.3f}")
    print(f"Graph-only detector AUC (lateral movement)   : {auc_lm:.3f}"
          "   <- the relationship-driven class")

    df.to_csv(FEATURED, index=False)
    with open(f"{_PROJECT_ROOT}/reports/graph_model_metrics.json", "w") as f:
        json.dump({"entities": n_ent, "resources": n_res,
                   "edges": g.number_of_edges(),
                   "graph_only_auc_all_anomalies": auc_all,
                   "graph_only_auc_lateral_movement": auc_lm}, f, indent=2)
    print("\nGraph features appended to access_logs_featured.csv")
    print("Saved reports/graph_model_metrics.json")


if __name__ == "__main__":
    main()
