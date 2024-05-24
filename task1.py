import os
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import abc_py
import re

# 确定设备
if torch.backends.mps.is_available():
    print("Metal is available!")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("Cuda is available!")
    device = torch.device("cuda")
else:
    print("Metal is not available.")
    device = torch.device("cpu")

# 定义数据集路径
dataset_path = './small'
yosys_path = 'oss-cad-suite/bin/'
lib_file = './lib/7nm/7nm.lib'
log_file = 'alu2.log'

# 加载.pkl文件内容
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 检查数据集中的.pkl文件
def check_pkl_files(dataset_path):
    pkl_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    return pkl_files

# 使用 yosys-abc 处理 AIG 文件
def run_yosys_abc(circuit_path, action_cmd, next_state):
    abc_run_cmd = f"{yosys_path}yosys-abc -c \"read {circuit_path}; {action_cmd} read_lib {lib_file}; write {next_state}; print_stats\" > {log_file}"
    os.system(abc_run_cmd)

# 处理AIG文件并生成图数据
def process_aig_file(circuit_path):
    _abc = abc_py.AbcInterface()
    _abc.start()

    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"circuitPath file '{circuit_path}' not found")

    _abc.read(circuit_path)
    data = {}
    numNodes = _abc.numNodes()

    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)

    edge_src_index = []
    edge_target_index = []

    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()

        data['num_inverted_predecessors'][nodeIdx] = 0

        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2

        if nodeType == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
        if nodeType == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2

        if aigNode.hasFanin0():
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

        if aigNode.hasFanin1():
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    if len(edge_src_index) != len(edge_target_index):
        raise ValueError("Mismatch in edge indices length")

    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes

    return data

# 创建图数据对象
def create_graph_data(data):
    edge_index = data['edge_index']
    x = torch.cat([data['node_type'].view(-1, 1), data['num_inverted_predecessors'].view(-1, 1)], dim=1).float()
    return Data(x=x, edge_index=edge_index)

# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# 训练函数
def train(model, data, optimizer, criterion, target):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out.view(-1), target)
    loss.backward()
    optimizer.step()
    return loss.item()

# 加载数据集
dataset = load_pkl_file(os.path.join(dataset_path, 'adder_901.pkl'))
inputs = dataset['input']
targets = torch.tensor(dataset['target'], device=device)

# 初始化模型、优化器和损失函数
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# 训练循环
for epoch in range(200):
    total_loss = 0
    for input_state, target in zip(inputs, targets):
        circuit_name, actions = input_state.split('_')
        circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
        next_state = f'{input_state}.aig'
        
        # 定义 Yosys ABC 操作命令
        synthesis_op_to_pos_dic = {
            0: "refactor",
            1: "refactor -z",
            2: "rewrite",
            3: "rewrite -z",
            4: "resub",
            5: "resub -z",
            6: "balance"
        }
        action_cmd = ''
        for action in actions:
            action_cmd += (synthesis_op_to_pos_dic[int(action)] + '; ')

        # 运行 Yosys ABC 生成新的 AIG 文件
        run_yosys_abc(circuit_path, action_cmd, next_state)

        # 处理新的 AIG 文件并生成图数据
        data = process_aig_file(next_state)
        graph_data = create_graph_data(data).to(device)
        loss = train(model, graph_data, optimizer, criterion, torch.tensor([target], device=device))
        total_loss += loss
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(inputs)}')

print("GCN training completed.")

# 可视化节点类型
plt.figure(figsize=(10, 6))
plt.hist(data['node_type'].cpu().numpy(), bins=np.arange(4) - 0.5, rwidth=0.8)
plt.xticks(ticks=[0, 1, 2], labels=['AND', 'PI', 'Other'])
plt.xlabel('Node Type')
plt.ylabel('Count')
plt.title('Distribution of Node Types')
plt.show()

# 可视化边
G = nx.DiGraph()
for i in range(data['nodes']):
    G.add_node(i, type=data['node_type'][i].item())

edges = list(zip(data['edge_index'][0].tolist(), data['edge_index'][1].tolist()))
G.add_edges_from(edges)

pos = nx.spring_layout(G)
node_colors = ['red' if G.nodes[i]['type'] == 0 else 'blue' if G.nodes[i]['type'] == 1 else 'green' for i in G.nodes]

plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, font_size=10, font_color='white', arrows=True)
plt.title('Graph Visualization of AIG Nodes and Edges')
plt.show()
