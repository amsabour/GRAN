from classifier.GraphSAGE import GraphSAGE
from classifier.DiffPool import DiffPool
from classifier.DGCNN import DGCNN
#
import torch
from torch.utils.data import DataLoader
from random import shuffle
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from classifier.losses import MulticlassClassificationLoss

import numpy as np


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def create_graph(n, p, label):
    edges = []
    for j in range(n):
        for k in range(j + 1, n):
            if np.random.uniform() < p:
                edges.append([j, k])

    edges = np.array(edges)
    edges = torch.from_numpy(edges).transpose(0, 1).cuda()
    edges_other_way = edges[[1, 0]]
    edges = torch.cat([edges, edges_other_way], dim=-1).to('cuda').long()

    x = torch.ones((n_nodes, 1)).cuda()
    batch = torch.zeros(n_nodes).long().cuda()
    y = torch.tensor([label]).long().cuda()

    data = Bunch(x=x, edge_index=edges, batch=batch, num_graphs=1, y=y, edge_weight=None)

    return data


model = GraphSAGE(1, 2, 3, 32, 'add').to('cuda')
# model = DiffPool(dim_features, dim_target, max_num_nodes=630).to('cuda')
# model = DGCNN(dim_features, dim_target, 'NCI1').to('cuda')
model.load_state_dict(torch.load('output/MODEL.pkl'))
model.train()

optimizer = Adam(model.parameters(), lr=0.005)
scheduler = ReduceLROnPlateau(optimizer, 'min')

loss_fun = MulticlassClassificationLoss(reduction='mean').cuda()
counter = 0

best_test_acc = 0

n_graphs = 10000

corrects = 0
total = 0

for i in range(n_graphs):
    optimizer.zero_grad()

    # Create the graph
    n_nodes = np.random.randint(10, 50)

    if i % 2 == 0:
        p = 0.25
        label = 0
    else:
        p = 0.75
        label = 1

    data = create_graph(n_nodes, p, label)

    output = model(data)
    loss, accuracy = loss_fun(data.y, output)
    loss = torch.mean(loss)

    if accuracy.item() == 100:
        corrects += 1
    total += 1

    if total % 55 == 0:
        print(output, data.y, loss)

        print("Accuracy so far: %.3f" % (corrects / total))








