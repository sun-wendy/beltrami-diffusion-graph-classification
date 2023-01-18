import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import Data, InMemoryDataset
import numpy as np


cate = [ 'Airplane',
        'Bag',
        'Cap',
        'Car',
        'Chair',
        'Earphone',
        'Guitar',
        'Knife',
        'Lamp',
        'Laptop',
        'Motorbike',
        'Mug',
        'Pistol',
        'Rocket',
        'Skateboard',
        'Table']

for cate_ in cate:
    dataset = ShapeNet(root='/tmp/ShapeNet', categories=cate_,pre_transform=T.KNNGraph(k=6))
    print(cate_,':',len(list(set(dataset.data.y.numpy()))))


# def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
#   visited_nodes = set()
#   queued_nodes = set([start])
#   row, col = dataset.data.edge_index.numpy()
#   while queued_nodes:
#     current_node = queued_nodes.pop()
#     visited_nodes.update([current_node])
#     neighbors = col[np.where(row == current_node)[0]]
#     neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
#     queued_nodes.update(neighbors)
#   return visited_nodes
#
#
# def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
#   remaining_nodes = set(range(dataset.data.x.shape[0]))
#   comps = []
#   while remaining_nodes:
#     start = min(remaining_nodes)
#     comp = get_component(dataset, start)
#     comps.append(comp)
#     remaining_nodes = remaining_nodes.difference(comp)
#   return np.array(list(comps[np.argmax(list(map(len, comps)))]))
#
#
# print(get_component(dataset))
