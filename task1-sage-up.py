import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm
from sklearn.model_selection import train_test_split
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
dataset_path = './sample'
# dataset_path = './small/project_data'
save_path = './save-up'
aig_path = './tmp'
yosys_path = 'oss-cad-suite/bin/'
lib_file = './lib/7nm/7nm.lib'
log_file = 'log/alu2.log'
eval_file_path = './task1_targets.pkl'  # 用于保存评估值的文件路径
model_resume_path = 'save-up/model_epoch_450.pth'  # 要从哪个保存的模型继续训练
regenerate_aig = False
resume = True
target_saved = True

# 创建保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(aig_path):
    os.makedirs(aig_path)

# 加载.pkl文件内容
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 检查数据集中的.pkl文件
def check_pkl_files(dataset_path):
    pkl_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    return pkl_files

# 使用 yosys-abc 处理 AIG 文件并提取评估值
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

# 改进后的GraphSAGE模型定义
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

# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# 验证函数
def validate_with_accuracy(model, loader, criterion, threshold=0.01):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validating"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out.view(-1), data.y)
            total_loss += loss.item() * data.num_graphs
            
            # 计算准确率
            correct_predictions += ((out.view(-1) - data.y).abs() < threshold).sum().item()
            total_predictions += data.y.size(0)

    mean_loss = total_loss / len(loader.dataset)
    accuracy = correct_predictions / total_predictions
    return accuracy, mean_loss

# 加载或初始化评估值字典
if os.path.exists(eval_file_path):
    with open(eval_file_path, 'rb') as f:
        eval_values = pickle.load(f)
else:
    eval_values = {}

# 加载所有 .pkl 文件并准备数据集
pkl_files = check_pkl_files(dataset_path)
all_graphs = []

if target_saved:
    for input_state in tqdm(eval_values, desc="Processing .pkl files"):
        circuit_name, actions = input_state.split('_')
        circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
        next_state = f'./tmp/{input_state}.aig'

        # 处理AIG文件并生成图数据
        data = process_aig_file(next_state)
        graph_data = create_graph_data(data)
        graph_data.y = torch.tensor([eval_values[input_state]], dtype=torch.float)
        all_graphs.append(graph_data)

else:
    for pkl_file in tqdm(pkl_files, desc="Processing .pkl files"):
        dataset = load_pkl_file(os.path.join(dataset_path, pkl_file))
        inputs = dataset['input']
        
        for input_state in inputs:
            circuit_name, actions = input_state.split('_')
            circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
            next_state = f'./tmp/{input_state}.aig'

            if regenerate_aig or not os.path.exists(next_state):
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

                # 运行 Yosys ABC 生成新的 AIG 文件并获取评估值
                eval_value = run_yosys_abc(circuit_path, action_cmd, next_state)
            else:
                # 运行 Yosys ABC 获取评估值
                eval_value = run_yosys_abc(circuit_path, '', next_state)
            
            # 保存评估值
            eval_values[input_state] = eval_value
            # 将评估值字典保存到文件
            with open(eval_file_path, 'wb') as f:
                pickle.dump(eval_values, f)

            # 处理AIG文件并生成图数据
            data = process_aig_file(next_state)
            graph_data = create_graph_data(data)
            graph_data.y = torch.tensor([eval_value], dtype=torch.float)
            all_graphs.append(graph_data)
    
# 划分训练集和验证集
train_graphs, val_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)

# 初始化模型、优化器和损失函数
model = GraphSAGE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 调整学习率
criterion = torch.nn.MSELoss()

# 从保存的模型继续训练
start_epoch = 0
if os.path.exists(model_resume_path) and resume:
    model.load_state_dict(torch.load(model_resume_path))
    start_epoch = int(model_resume_path.split('_')[-1].split('.')[0])
    print(f"Resuming training from epoch {start_epoch}")

# 训练循环
num_epochs = 500
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, criterion)
    acc, val_loss = validate_with_accuracy(model, val_loader, criterion)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {acc}')
    # 每50个epoch保存一次模型
    if (epoch + 1) % 50 == 0:
        model_save_path = os.path.join(save_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
