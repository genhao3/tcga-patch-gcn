import os, time
import torch
import torch_geometric
from datasets.BatchWSI import BatchWSI
from models.model_graph_mil import *
device = torch.device('cuda:0')

dataroot = './data/TCGA/BRCA/'
large_graph_pt = 'TCGA-BH-A0DV-01Z-00-DX1.2F0B5FB3-40F0-4D27-BFAC-390FB9A42B39.pt' # example input

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# Graph Data Structure
# N: number of patches
# M: number of edges
# centroid: [N x 2] matrix containing centroids for each patch
# edge_index: [2 x M] matrix containing edges between patches (connected via adjacent spatial coordinates)
# edge_latent: [2 x M] matric containing edges between patches (connected via latent space)
# x: [N x 1024] matrix which uses 1024-dim extracted ResNet features for each iamge patch (features saved for simplicity)
data = torch.load(os.path.join(dataroot, large_graph_pt))
data

data = BatchWSI.from_data_list([torch.load(os.path.join(dataroot, large_graph_pt)), 
                                torch.load(os.path.join(dataroot, large_graph_pt))])
data

# Inference + Backprop using 23K patches
data = BatchWSI.from_data_list([torch.load(os.path.join(dataroot, large_graph_pt))])
data = data.to(device)
data

model_dict = {'num_layers': 4, 'edge_agg': 'spatial', 'resample': 0, 'n_classes': 1}
model = PatchGCN_Surv(**model_dict).to(device)
print("Number of Parameters:", count_parameters(model))

### Example Forward Paas + Gradient Backprop
start = time.time()
out = model(x_path=data)
out[0].backward()
print('Time Elapsed: %0.5f seconds' % (time.time() - start))

# Inference + Backprop using 92K patches
### Simulating a very large graph (containing 4 subgraphs of 23K patches each)
data = BatchWSI.from_data_list([torch.load(os.path.join(dataroot, large_graph_pt)), 
                                torch.load(os.path.join(dataroot, large_graph_pt)),
                                torch.load(os.path.join(dataroot, large_graph_pt)),
                                torch.load(os.path.join(dataroot, large_graph_pt))])
data = data.to(device)
data

model_dict = {'num_layers': 4, 'edge_agg': 'spatial', 'resample': 0, 'n_classes': 1}
model = PatchGCN_Surv(**model_dict).to(device)
print("Number of Parameters:", count_parameters(model))

### Example Forward Paas + Gradient Backprop
start = time.time()
out = model(x_path=data)
out[0].backward()
print('Time Elapsed: %0.5f seconds' % (time.time() - start))