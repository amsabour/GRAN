# from classifier.GraphSAGE import GraphSAGE
from dataset import GRANData
from utils.data_helper import create_graphs
import torch
from utils.arg_helper import get_config
from torch.utils.data import DataLoader
from random import shuffle
from torch.optim import Adam
from classifier.losses import MulticlassClassificationLoss


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def data_to_bunch(data):
    num_nodes = data[0]['num_nodes_gt']
    node_features = []
    node_labels = data[0]['node_label'][:, 0]
    for j in range(num_nodes.shape[0]):
        node_feature = torch.zeros(num_nodes[j], 3)
        node_feature[range(num_nodes[j]), node_labels[j][:num_nodes[j]]] = 1
        node_features.append(node_feature)

    x = torch.cat(node_features, 0).cuda()

    adj = data[0]['adj'][0]
    lower_part = torch.tril(adj, diagonal=-1)

    edge_mask = (lower_part != 0).to('cuda')
    edges = edge_mask.nonzero().transpose(0, 1).to('cuda').long()

    batch = torch.cat([torch.tensor([ii] * bb).view(1, -1) for ii, bb in enumerate(num_nodes)], dim=1).squeeze().long()

    y = data[0]['graph_label'].cuda()
    num_graphs = len(node_features)

    # print(torch.sum(num_nodes), batch.shape)
    # print(num_nodes)
    # print(batch)

    return Bunch(x=x, edge_index=edges, batch=batch, num_graphs=num_graphs, y=y)


def get_accuracy(loader, model):
    acc = 0
    graphs = 0
    for data in train_loader:
        data = data_to_bunch(data)
        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)
        loss, accuracy = loss_fun(data.y, *output)

        acc += accuracy * data.num_graphs
        graphs += data.num_graphs

    acc /= graphs
    return acc


graphs = create_graphs("PROTEINS", data_dir='data/')
shuffle(graphs)

num_graphs = len(graphs)
num_train = int(num_graphs * 0.8)

train_graphs = graphs[:num_train]
test_graphs = graphs[num_train:]

config = get_config('config/gran_PROTEINS.yaml', is_test='false')
config.use_gpu = config.use_gpu and torch.cuda.is_available()

train_dataset = GRANData(config, train_graphs, tag='train')
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=False,
                          collate_fn=train_dataset.collate_fn,
                          drop_last=False)

test_dataset = GRANData(config, test_graphs, tag='test')
test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=True,
                         collate_fn=test_dataset.collate_fn,
                         drop_last=False)

for i in range(1000):
    for data in train_loader:
        data = data_to_bunch(data)
        break
    # data = train_dataset.__getitem__(0)
    # print(data[0]['num_nodes'])

# model = GraphSAGE(3, 2, 3, 32, 'add').to('cuda')
# model.train()
# optimizer = Adam(model.parameters(), lr=0.001)
# loss_fun = MulticlassClassificationLoss(weight=torch.tensor([0.8086, 1.1914]).cuda())

# for i in range(1000):
#     # model.train()
#     for data in train_loader:
#         data = data_to_bunch(data)

# optimizer.zero_grad()
# output = model(data)
# if not isinstance(output, tuple):
#     output = (output,)
# loss, acc = loss_fun(data.y, *output)
# loss.backward()
# optimizer.step()
#
# model.eval()
# with torch.no_grad():
#     train_acc = get_accuracy(train_loader)
#     test_acc = get_accuracy(test_loader)
#     print("Epoch: %d ---- Train accuracy: %.3f, Test accuracy: %.3f" % (i + 1, train_acc, test_acc))
