import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.model_selection import train_test_split
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
save_path = './save'
aig_path = './tmp'
model_resume_path = './save/model_epoch_100.pth'  # 要从哪个保存的模型继续训练

# 创建保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(aig_path):
    os.makedirs(aig_path)

# 布尔值来决定是否重新生成AIG文件
regenerate_aig = False

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

# 改进后的GCN模型定义
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
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validating"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out.view(-1), data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# 加载所有 .pkl 文件并准备数据集
pkl_files = check_pkl_files(dataset_path)
all_graphs = []
all_targets = []

for pkl_file in tqdm(pkl_files, desc="Processing .pkl files"):
    dataset = load_pkl_file(os.path.join(dataset_path, pkl_file))
    inputs = dataset['input']
    targets = dataset['target']
    
    for input_state, target in zip(inputs, targets):
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

            # 运行 Yosys ABC 生成新的 AIG 文件
            run_yosys_abc(circuit_path, action_cmd, next_state)

        # 处理AIG文件并生成图数据
        data = process_aig_file(next_state)
        graph_data = create_graph_data(data)
        graph_data.y = torch.tensor([target], dtype=torch.float)
        all_graphs.append(graph_data)

# 划分训练集和验证集
train_graphs, val_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)

# 初始化模型、优化器和损失函数
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 调整学习率
criterion = torch.nn.MSELoss()

# 从保存的模型继续训练
start_epoch = 0
if os.path.exists(model_resume_path):
    model.load_state_dict(torch.load(model_resume_path))
    start_epoch = int(model_resume_path.split('_')[-1].split('.')[0])
    print(f"Resuming training from epoch {start_epoch}")

# 训练循环
num_epochs = 500
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
    
    # 每20个epoch保存一次模型
    if (epoch + 1) % 20 == 0:
        model_save_path = os.path.join(save_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

print("GCN training completed.")
