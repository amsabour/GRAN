from classifier.GraphSAGE import GraphSAGE
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


def data_to_bunch(data):
    num_nodes = data[0]['num_nodes_gt']
    node_features = []
    node_labels = data[0]['node_label'][:, 0]
    for j in range(num_nodes.shape[0]):
        node_feature = torch.zeros(num_nodes[j], 3)
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

    batch = torch.cat([torch.tensor([ii] * bb).view(1, -1) for ii, bb in enumerate(num_nodes)], dim=1).to(
        'cuda').squeeze().long()

    y = data[0]['graph_label'].long().cuda()
    num_graphs = len(node_features)

    return Bunch(x=x, edge_index=edges, batch=batch, num_graphs=num_graphs, y=y)


def get_loss_accuracy(loader, model):
    acc = 0
    loss = 0
    graphs = 0
    for data in loader:
        data = data_to_bunch(data)
        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)
        loss, accuracy = loss_fun(data.y, *output)

        number_of_ones = torch.sum(data.y).item()
        number_of_zeros = data.num_graphs - number_of_ones

        if number_of_ones == 0 or number_of_zeros == 0:
            loss = torch.mean(loss)
        else:
            weight_of_zero = 1 / (2 * number_of_zeros)
            weight_of_one = 1 / (2 * number_of_ones)
            weights = torch.ones_like(loss) * weight_of_zero + data.y * (weight_of_one - weight_of_zero)

            loss = torch.sum(loss * weights)

        loss += loss * data.num_graphs
        acc += accuracy * data.num_graphs
        graphs += data.num_graphs

    acc /= graphs
    return loss, acc


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

model = GraphSAGE(3, 2, 3, 32, 'add').to('cuda')
model.train()
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min')

loss_fun = MulticlassClassificationLoss(reduction='none').cuda()
counter = 0

best_test_acc = 0

for i in range(1000):
    model.train()
    for data in train_loader:
        counter += 1
        data = data_to_bunch(data)

        optimizer.zero_grad()
        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)
        loss, acc = loss_fun(data.y, *output)

        number_of_ones = torch.sum(data.y).item()
        number_of_zeros = data.num_graphs - number_of_ones

        if number_of_ones == 0 or number_of_zeros == 0:
            loss = torch.mean(loss)
        else:
            weight_of_zero = 1 / (2 * number_of_zeros)
            weight_of_one = 1 / (2 * number_of_ones)
            weights = torch.ones_like(loss) * weight_of_zero + data.y * (weight_of_one - weight_of_zero)

            loss = torch.sum(loss * weights)

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
            torch.save(model.state_dict(), 'output/PROTEINS.pkl')

        scheduler.step(test_loss)
