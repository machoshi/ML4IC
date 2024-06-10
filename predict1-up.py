import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm
from tqdm import tqdm
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

# 定义路径
dataset_path = './small/project_data'
yosys_path = 'oss-cad-suite/bin/'
lib_file = './lib/7nm/7nm.lib'
log_file = 'log/alu2.log'
model_path = 'save-up/model_epoch_450.pth'  # 替换为您的模型路径
output_path = './predictions'
aig_path = './tmp'
eval_file_path = './task1_targets.pkl'

# 创建保存目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 检查数据集中的.pkl文件
def check_pkl_files(dataset_path):
    pkl_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    return pkl_files

# 加载.pkl文件内容
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 使用 yosys-abc 处理 AIG 文件
def run_yosys_abc(circuit_path, action_cmd, next_state):
    abc_run_cmd = f"{yosys_path}yosys-abc -c \"read {circuit_path}; {action_cmd} read_lib {lib_file}; map; topo; stime\" > {log_file}"
    os.system(abc_run_cmd)
    with open(log_file) as f:
        area_information = re.findall(r'[a-zA-Z0-9.]+', f.readlines()[-1])
    eval_value = float(area_information[-9]) * float(area_information[-4])
    return eval_value

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

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(2, 32)
        self.bn1 = BatchNorm(32)
        self.conv2 = SAGEConv(32, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = SAGEConv(64, 128)
        self.bn3 = BatchNorm(128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)  # 增加Dropout防止过拟合

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)  # 使用全局平均池化层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载模型
model = GraphSAGE().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载所有 .pkl 文件并进行预测
pkl_files = check_pkl_files(dataset_path)
predictions = []
all_graphs = []

if os.path.exists(eval_file_path):
    with open(eval_file_path, 'rb') as f:
        eval_values = pickle.load(f)
else:
    eval_values = {}
    
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
criterion = F.mse_loss
model.eval()
total_loss = 0
correct_predictions = 0
total_predictions = 0
aig_files = [f for f in os.listdir('./InitialAIG/train/') if f.endswith('.aig')]

# 对所有aig进行一次run_yosys_abc
baseline_dict = {}
for aig_file in tqdm(aig_files, desc="baseline generate"):
    circuit_path = f'./InitialAIG/train/{aig_file}'
    baseline = run_yosys_abc(circuit_path, RESYN2_CMD, circuit_path)
    baseline_dict[aig_file] = baseline

# 在循环中直接调用run_yosys_abc的结果
for input_state in tqdm(eval_values, desc="Processing .pkl files"):
    circuit_name, actions = input_state.split('_')
    circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
    next_state = f'./tmp/{input_state}.aig'

    # 处理AIG文件并生成图数据
    data = process_aig_file(next_state)
    graph_data = create_graph_data(data)
    baseline = baseline_dict[f'{circuit_name}.aig']
    graph_data.y = torch.tensor([eval_values[input_state]], dtype=torch.float)
    target = 1 - eval_values[input_state] / baseline

    # 进行预测
    with torch.no_grad():
        graph_data = graph_data.to(device)
        prediction = model(graph_data)
        print(baseline, prediction)
        prediction = 1 - prediction / baseline
        predictions.append({'input': input_state, 'target': target, 'prediction': prediction})
        loss = criterion(prediction, graph_data.y)
        total_loss += loss.item()
        correct_predictions += ((prediction - graph_data.y).abs() < 0.01).sum().item()
        total_predictions += graph_data.y.size(0)

mean_loss = total_loss / len(eval_values)
accuracy = correct_predictions / total_predictions
print(mean_loss, accuracy)
print(f'Prediction Loss: {mean_loss}, Accuracy: {accuracy}')

# 保存预测结果
output_file = os.path.join(output_path, 'predictions-up.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(predictions, f)
print(f"Predictions saved to {output_file}")
