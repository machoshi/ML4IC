import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from tqdm import tqdm
import abc_py

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
model_path = './save/model_epoch_100.pth'  # 替换为您的模型路径
output_path = './predictions'
aig_path = './tmp'

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
        self.conv1 = GCNConv(2, 32)
        self.bn1 = BatchNorm(32)
        self.conv2 = GCNConv(32, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GCNConv(64, 128)
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
model = GCN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载所有 .pkl 文件并进行预测
pkl_files = check_pkl_files(dataset_path)
predictions = []

for pkl_file in tqdm(pkl_files, desc="Processing .pkl files for prediction"):
    dataset = load_pkl_file(os.path.join(dataset_path, pkl_file))
    inputs = dataset['input']
    targets = dataset['target']
    
    for input_state, target in zip(inputs, targets):
        circuit_name, actions = input_state.split('_')
        circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
        next_state = f'./tmp/{input_state}.aig'
        
        if not os.path.exists(next_state):
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

        # 处理AIG文件并生成图数据
        data = process_aig_file(next_state)
        graph_data = create_graph_data(data)
        graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)  # 添加batch属性
        
        # 进行预测
        with torch.no_grad():
            graph_data = graph_data.to(device)
            prediction = model(graph_data).cpu().numpy()[0]
            predictions.append({'input': input_state, 'target': target, 'prediction': prediction})

# 保存预测结果
output_file = os.path.join(output_path, 'predictions.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(predictions, f)
print(f"Predictions saved to {output_file}")
