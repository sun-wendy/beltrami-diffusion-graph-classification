# Adapted from source code

import os.path as osp
import argparse
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import time
import pickle
import numpy as np
from torch_geometric.data import Data, InMemoryDataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.min(),data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def main(opt,cate):
    dataset = ShapeNet(root='/tmp/ShapeNet', categories=cate, pre_transform=T.KNNGraph(k=6))

    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]
    y_min = y_new.min()
    y_new = y_new - y_min
    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
      x=x_new,
      edge_index=torch.LongTensor(edges),
      y=y_new,
      train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
    data = set_train_val_test_split(
      12345,
      dataset.data,
    1500)
    dataset.data = data

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(data.edge_index, embedding_dim=opt['embedding_dim'], walk_length=opt['walk_length'],
                     context_size=opt['context_size'], walks_per_node=opt['walks_per_node'],
                     num_negative_samples=opt['neg_pos_ratio'], p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc, z

    ### here be main code
    t = time.time()
    for epoch in range(1, opt['epochs'] + 1):

        loss = train()
        train_t = time.time() - t
        t = time.time()
        acc, _ = test()
        test_t = time.time() - t
        print(f'Epoch: {epoch:02d}, Train: {train_t:.2f}, Test: {test_t:.2f},  Loss: {loss:.4f}, Acc: {acc:.4f}')

    acc, z = test()
    print(f"[i] Final accuracy is {acc}")
    print(f"[i] Embedding shape is {z.data.shape}")

    fname = "%s_DW64_%s.pkl" % (
        'shapenet',opt['shapenet_data']
    )

    print(f"[i] Storing embeddings in {fname}")

    with open(osp.join(""
                       "data/pos_encodings", fname), 'wb') as f:
        # make sure the pickle is not bound to any gpu, and store test acc with data
        pickle.dump({"data": z.data.to(torch.device("cpu")), "acc": acc}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--walk_length', type=int, default=20,  # note this can grow much bigger (paper: 40~100)
                        help='Walk length')
    parser.add_argument('--context_size', type=int, default=16,  # paper shows increased perf until 16
                        help='Context size')
    parser.add_argument('--walks_per_node', type=int, default=16,  # best paper results with 18
                        help='Walks per node')
    parser.add_argument('--neg_pos_ratio', type=int, default=1,
                        help='Number of negatives for each positive')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (default 0)')
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument("--shapenet_data", default='Cap',type=str, help='Airplane,Bag,Cap,Car,Chair,Earphone,Guitar,Knife,Lamp,Laptop,Motorbike,Mug,Pistol,Rocket,Skateboard,Table')

    args = parser.parse_args()
    opt = vars(args)
    opt['rewiring'] = None
    cate = []
    for item in opt['shapenet_data'].split(','):
        cate.append(item)

    main(opt,cate)
