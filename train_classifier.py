from classifier.GraphSAGE import GraphSAGE
from classifier.DiffPool import DiffPool
from classifier.DGCNN import DGCNN
#
from dataset import GRANData
from utils.data_helper import create_graphs
import torch
from utils.arg_helper import get_config
from torch.utils.data import DataLoader
from random import shuffle
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from classifier.losses import MulticlassClassificationLoss

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)



config = get_config('config/gran_PROTEINS.yaml', is_test='false')
config.use_gpu = config.use_gpu and torch.cuda.is_available()




def data_to_bunch(data):
    # for k in data[0].keys():
    #     l = data[0][k]

    #     if hasattr(l, 'shape'):
    #         print(k, " ", l.shape)
    #     else:
    #         print(k, " ", l)

    num_nodes = data[0]['num_nodes_gt']
    node_features = []
    node_labels = data[0]['node_label'][:, 0]
    for j in range(num_nodes.shape[0]):
        node_feature = torch.zeros(num_nodes[j], config.dataset.num_node_label)
        node_feature[range(num_nodes[j]), node_labels[j][:num_nodes[j]]] = 1
        node_features.append(node_feature)

    x = torch.cat(node_features, 0).cuda()

    adj = data[0]['adj'][:, 0]
    edges_list = []

    counter = 0
    for j in range(adj.shape[0]):
        A = adj[j]
        lower_part = torch.tril(A, diagonal=-1)
        edge_mask = (lower_part != 0).to('cuda')
        edges = edge_mask.nonzero().transpose(0, 1).to('cuda').long()

        edges_list.append(edges + counter)
        counter += num_nodes[j]

    edges = torch.cat(edges_list, dim=1).to('cuda').long()
    edges_other_way = edges[[1, 0]]
    edges = torch.cat([edges, edges_other_way], dim=1).to('cuda').long()

    batch = torch.cat([torch.tensor([ii] * bb).view(1, -1) for ii, bb in enumerate(num_nodes)], dim=1).to(
        'cuda').squeeze().long()

    y = data[0]['graph_label'].long().cuda()
    num_graphs = len(node_features)

    edges_truncated = data[0]['edges'].transpose(0, 1).to('cuda').long()
    batch_truncated = data[0]['batch'].to('cuda').long()

    truncated_node_features = []
    for j in range(num_nodes.shape[0]):
        truncated_size = (batch_truncated == j).sum()
        truncated_node_feature = node_features[j][:truncated_size]
        truncated_node_features.append(truncated_node_feature)

    x_truncated = torch.cat(truncated_node_features, dim=0).cuda()

    b1 = Bunch(x=x, edge_index=edges, batch=batch, num_graphs=num_graphs, y=y, edge_weight=None)
    b2 = Bunch(x=x_truncated, edge_index=edges_truncated, batch=batch_truncated, num_graphs=num_graphs, y=y,
               edge_weight=None)

    return [b1, b2]


def get_loss_accuracy(loader, model):
    corrects = 0
    total_loss = 0
    graphs = 0
    for data in loader:
        datas = data_to_bunch(data)
        for data in datas:
            output = model(data)

            loss, accuracy = loss_fun(data.y, output)
            loss = torch.mean(loss)

            total_loss += loss * data.num_graphs
            corrects += accuracy * data.num_graphs / 100
            graphs += data.num_graphs

    acc = corrects / graphs
    return total_loss, acc


graphs = create_graphs("PROTEINS", data_dir='data/')
shuffle(graphs)

num_graphs = len(graphs)
num_train = int(num_graphs * 0.8)

train_graphs = graphs[:num_train]
test_graphs = graphs[num_train:]

train_dataset = GRANData(config, train_graphs, tag='train')
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          collate_fn=train_dataset.collate_fn,
                          num_workers=4,
                          drop_last=False)

test_dataset = GRANData(config, test_graphs)
test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=True,
                         collate_fn=test_dataset.collate_fn,
                         num_workers=4,
                         drop_last=False)

dim_features = config.dataset.num_node_label
dim_target = 2

model = GraphSAGE(dim_features, dim_target, 3, 32, 'add').to('cuda')
# model = DiffPool(dim_features, dim_target, max_num_nodes=630).to('cuda')
# model = DGCNN(dim_features, dim_target, 'PROTEINS_full').to('cuda')
model.train()
optimizer = Adam(model.parameters(), lr=0.005)
scheduler = ReduceLROnPlateau(optimizer, 'min')

loss_weights = torch.tensor([0.8, 1.2]).cuda()
loss_fun = MulticlassClassificationLoss(weight=loss_weights, reduction='none').cuda()
counter = 0

best_test_acc = 0

for i in range(1000):
    model.train()

    for data in train_loader:
        counter += 1
        datas = data_to_bunch(data)

        for data in datas:
            optimizer.zero_grad()

            output = model(data)
            loss, acc = loss_fun(data.y, output)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            if counter % 10 == 1:
                print("Step %s: Loss is %.3f" % (counter, loss.item()))

    model.eval()
    with torch.no_grad():
        train_loss, train_acc = get_loss_accuracy(train_loader, model)
        test_loss, test_acc = get_loss_accuracy(test_loader, model)
        print("Epoch: %d ---- Train accuracy: %.3f, Train loss: %.3f, Test accuracy: %.3f, Test loss: %.3f" % (
            i + 1, train_acc, train_loss, test_acc, test_loss))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print("\033[92m" + "Best test accuracy updated: %s" % (test_acc.item()) + "\033[0m")
            torch.save(model.state_dict(), 'output/MODEL_PROTEINS_GRAPHSAGE.pkl')

        scheduler.step(test_loss)
