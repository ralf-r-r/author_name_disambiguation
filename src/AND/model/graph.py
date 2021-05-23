import networkx as nx
import pandas as pd
from typing import List

__all__ = ['create_graph','get_disconnected_subgraphs']

def create_graph(df:pd.DataFrame) -> nx.Graph:
    """
    Creates a graph from the data set containing the predictions
    nodes are contribution_ids
    nodes are connected with edges if the pairwise prediction was positive (1)
    :param df:
    :return:
    """
    nodes1 = list(df["contribution_id"].unique())
    nodes2 = list(df["contribution_id_2nd"].unique())
    nodes1.extend(nodes2)
    nodes = list(set(nodes1))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    df_edges = df[df["prediction"]==1]
    edge_list = []
    for c1, c2 in df_edges[["contribution_id", "contribution_id_2nd"]].values:
        edge_list.append((c1, c2))
    G.add_edges_from(edge_list)

    return G

def get_disconnected_subgraphs(G:nx.Graph)->List[set]:
    """
    Finds the connected components and returns the contributions of the subgraphs
    :param G: nx.Graph
    :return: List[set]
    """
    subgraphs = []
    for connected_component in nx.connected_components(G):
        subgraphs.append(connected_component)
    return subgraphs