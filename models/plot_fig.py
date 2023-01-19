# Adapted from source code

import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from importlib import reload

import os
import  datetime

import data as dt

"""
Visualize graphs after certain epochs
Pictures saved in "visualization" folder
"""

def remove_edges(g, edges, attention, threshold):
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    index = (attention < threshold)+0

    delete_edges = edges[:, index]
    print('deleting {} edges'.format(delete_edges.shape[1]))
    edge_list = list(zip(delete_edges[0], delete_edges[1]))
    print(type(edge_list))
    g.remove_edges_from(edge_list[0])
    print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))
    return g

def construct_graph(edges, attention=None, threshold=0.01):
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    if attention is not None:
        edges = edges[:, attention > threshold]
    edge_list = zip(edges[0], edges[1])
    g = nx.Graph(edge_list)
    return g

def get_model_data(model,dataset,epoch):
    attention = model.odeblock.odefunc.attention_weights
    edges = model.odeblock.odefunc.edge_index
    print('edges shape: {}, attention shape: {}'.format(edges.shape, attention.shape))
    print(attention.min(), attention.mean(), attention.max())
    atts = attention.detach().cpu().numpy()[:, 0]
    print(atts.shape)
    plt.hist(atts, bins=np.linspace(0, 1, 11))
    plt.hist(atts, bins=np.linspace(0, 0.01, 11))
    print(attention.shape, edges.shape)
    labels = dataset.data.y.cpu().numpy()
    print(len(labels))
    edges = model.odeblock.odefunc.edge_index
    g = construct_graph(edges)
    print(g.edges([32]))
    print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))

    g.remove_edges_from([(32, 387), (32, 790), (32, 791), (32, 1063), (32, 32)])
    print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))

    delete_edges = edges[:, attention[:, 0].detach().cpu().numpy() < 0.1]
    print(delete_edges.shape)

    g = remove_edges(g, edges, attention, threshold=0.02)

    # for threshold in np.linspace(0, 0.01, 20):
    #     edges = model.odeblock.odefunc.edge_index
    #     g = construct_graph(edges)
    #     attention = model.odeblock.odefunc.edge_weight
    #     g = remove_edges(g, edges, attention, threshold)
    #     comps = nx.number_connected_components(g)
    #     print(
    #         '{} remaining edges. {} connected components at threshold {}'.format(g.number_of_edges(), comps, threshold))
    #
    # for threshold in np.linspace(0, 0.01, 20):
    #     edges = model.odeblock.odefunc.edge_index
    #     g = construct_graph(edges)
    #     attention = model.odeblock.odefunc.attention_weights[:, 0]  # just using one head for now.
    #     g = remove_edges(g, edges, attention, threshold)
    #     comps = nx.number_connected_components(g)
    #     print(
    #         '{} remaining edges. {} connected components at threshold {}'.format(g.number_of_edges(), comps, threshold))
    prefix_name = fileName = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    plt.close()
    nx.draw(g, with_labels=False, font_weight='bold', node_size=5, node_color=labels)
    plt.savefig("../picture/{}_path_epoch_{}.png".format(prefix_name,str(epoch)))
    plt.close()

    ccs = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    g0 = g.subgraph(ccs[0])

    cc_idx = list(ccs[0])

    nx.draw(g0, with_labels=False, font_weight='bold', node_size=5, node_color=labels[cc_idx])
    plt.savefig("../picture/{}_path_sub_epoch_{}.png".format(prefix_name,str(epoch)))
    plt.close()
