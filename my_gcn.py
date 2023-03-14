import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import Data,DataLoader
# from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
# import torch
from torch.nn import Linear
# import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# class SmallNet(torch.nn.Module):
#     def __init__(self):
#         super(SmallNet, self).__init__()
#         self.conv1 = GCNConv(2, 4)
#         self.linear1 = torch.nn.Linear(4,3)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         # print(x)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x) # [6,4]
#         x, _ = scatter_max(x, data.batch.long(), dim=0)  # [batch,4],,data.batch.shape [6]
#         x = self.linear1(x)  # [batch,3]
#         return x

# def init_data():
#     labels=np.array([0,1,2],dtype=int)
#     a=labels[0]
#     data_list = []
	
# 	#定义第一个节点的信息
#     x = np.array([
#         [0, 0],
#         [1, 1],
#         [2, 2]
#     ])
#     x = torch.tensor(x, dtype=torch.float)
#     edge = np.array([
#         [0, 0, 2],
#         [1, 2, 0]
#     ])
#     edge = torch.tensor(edge, dtype=torch.long)
#     data_list.append(Data(x=x, edge_index=edge.contiguous(), t=int(labels[0])))

# 	#定义第二个节点的信息
#     x = np.array([
#         [0, 0],
#         [1, 1],
#         [2, 2]
#     ])
#     x = torch.tensor(x, dtype=torch.float)
#     edge = np.array([
#         [0, 1],
#         [1, 2]
#     ])
#     edge = torch.tensor(edge, dtype=torch.long)
#     data_list.append(Data(x=x, edge_index=edge.contiguous(), t=int(labels[1])))

# 	#定义第三个节点的信息
#     x = np.array([
#         [0, 0],
# 	[1, 1],
#         [2, 2]
#     ])
#     x = torch.tensor(x, dtype=torch.float)
#     edge = np.array([
#         [0, 1, 2],
#         [2, 2, 0]
#     ])
#     edge = torch.tensor(edge, dtype=torch.long)
#     data_list.append(Data(x=x, edge_index=edge.contiguous(), t=int(labels[2])))
#     return data_list

# epoch_num=10000
# batch_size=2
# trainset=init_data()
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# device = torch.device('cpu')
# model = SmallNet().to(device)
# optimizer = torch.optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
# for epoch in range(epoch_num):
#     train_loss = 0.0
#     for i, batch in enumerate(trainloader):
#         batch = batch.to("cpu")
#         optimizer.zero_grad()
#         outputs = model(batch)
#         loss = criterion(outputs, batch.t)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.cpu().item()
#         # print('epoch: {:d} loss: {:.3f}'
#         #       .format(epoch + 1, loss.cpu().item()))
#     print('epoch: {:d} loss: {:.3f}'
#           .format(epoch + 1, train_loss / batch_size))

dataset = TUDataset(root='/data_local2/ljjdata/TCGA/test_gcn_data/TUDataset', name='MUTAG')

# 2、训练集、测试集准备
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]
# 训练集、测试集数量
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):# x = [1144,7],edge_index=[2,2510],batch=[1144]
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)  # 1144,hidden_channels
        x = x.relu()  # 1144,hidden_channels
        x = self.conv2(x, edge_index)  # 1144,hidden_channels
        x = x.relu()  # 1144,hidden_channels
        x = self.conv3(x, edge_index)  # 1144,hidden_channels

        # 2. Readout layer
        x = global_mean_pool(x, batch.long())  # [batch_size, hidden_channels][64,hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)  # [64,hidden_channels]
        x = self.lin(x) # [64,num_classes]
        
        return x

model = GCN(hidden_channels=64)
print(model)

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset. DataBatch(edge_index=[2, 2510], x=[1144, 7], edge_attr=[2510, 4], y=[64], batch=[1144], ptr=[65])
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')