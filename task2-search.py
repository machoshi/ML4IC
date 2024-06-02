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
dataset_path = './small'
aig_path = './tmp'
yosys_path = '/home/zhangcy/ml/oss-cad-suite/bin/'
lib_file = './lib/7nm/7nm.lib'
log_file = 'log/alu2.log'
model0_path = './save/model_sage_epoch_1400.pth'
model1_path = './save/model_epoch_150.pth'

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

# 加载模型

model0 = GraphSAGE().to(device)
model0.load_state_dict(torch.load(model0_path, map_location=device))
model0.eval()

model1 = GraphSAGE().to(device)
model1.load_state_dict(torch.load(model1_path, map_location=device))
model1.eval()

# search part

def eval(circuit_name, actions, mode=0):
    circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
    next_state = f'./tmp/{circuit_name}_{actions}.aig'
    
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
        if mode == 0:
            prediction = model0(graph_data).cpu().numpy()[0]
        else:
            prediction = model1(graph_data).cpu().numpy()[0]
        return prediction

# --- dfs --- #

def dfs(circuit_name, cur_depth, cur_actions, best_score=float('-inf'), best_actions="", max_depth=10):
    if cur_depth == max_depth:
        score = eval(circuit_name, cur_actions, 0)
        if score > best_score:
            best_score = score
            best_actions = cur_actions
        return best_score, best_actions

    for action in "0123456":
        actions = cur_actions + action
        score = eval(circuit_name, actions, 0) + eval(circuit_name, actions, 1)
        if score > best_score:
            best_score, best_actions = dfs(circuit_name, cur_depth + 1, actions, best_score, best_actions, max_depth)

    return best_score, best_actions

# --- bfs --- #

from collections import deque

def bfs(circuit_name, best_score=float('-inf'), best_actions="", max_depth=10):
    queue = deque([(0, "")])  # (cur_depth, cur_actions)

    while queue:
        cur_depth, cur_actions = queue.popleft()

        if cur_depth == max_depth:
            score = eval(circuit_name, cur_actions, 0)
            if score > best_score:
                best_score = score
                best_actions = cur_actions
            continue

        for action in "0123456":
            actions = cur_actions + action
            score = eval(circuit_name, actions, 0) + eval(circuit_name, actions, 1)
            if score > best_score:
                queue.append((cur_depth + 1, actions))

    return best_score, best_actions

# --- beam --- #

from heapq import heappush, heappop

def beam_search(circuit_name, max_depth=10, beam_width=10):
    # Priority queue to store states with their evaluation scores
    beam = [(eval(circuit_name, "", 0), "")]  # (cur_score, cur_actions)
    
    for step in range(max_depth):
        new_beam = []
        for cur_score, cur_actions in beam:
            for action in "0123456":
                actions = cur_actions + action
                score = eval(circuit_name, actions, 0) + eval(circuit_name, actions, 1)
                heappush(new_beam, (score, actions))

        # Keep only the top beam_width states
        beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

    # Return the best path and its score
    best_score, best_actions = max(beam, key=lambda x: eval(circuit_name, x[1], 0))
    return eval(circuit_name, best_actions, 0), best_actions

# --- Astar --- #

from queue import PriorityQueue

def Astar(circuit_name, max_depth=10):
    open_set = PriorityQueue()
    open_set.put((-eval(circuit_name, "", 0), ""))  # (cur_score, cur_actions)
    closed_set = set()

    while not open_set.empty():
        cur_score, cur_actions = open_set.get()

        if len(cur_actions) == max_depth:
            return eval(circuit_name, cur_actions, 0), cur_actions 

        if cur_actions in closed_set:
            continue

        closed_set.add(cur_actions)

        for action in "0123456":
            actions = cur_actions + action

            if actions not in closed_set:
                g = eval(circuit_name, actions, 0)  # Current cost to reach this state
                h = eval(circuit_name, actions, 1)  # Heuristic future cost
                f = g + h  # Combined cost

                open_set.put((-f, actions))

    return None, None

# --- MCTS --- #

import math
import random

class TreeNode:
    def __init__(self, actions, parent=None):
        self.actions = actions
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 7

    def best_child(self, C):
        choices_weights = [
            (child.value / child.visits) + C * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def add_child(self, child_node):
        self.children.append(child_node)

def select_best_node(node, C=math.sqrt(2)):
    while node.is_fully_expanded():
        node = node.best_child(C)
    return node

def rollout(circuit_name, actions, remaining_steps):
    cur_actions = actions
    for _ in range(remaining_steps):
        action = random.choice(["0", "1", "2", "3", "4", "5", "6"])
        cur_actions = cur_actions + action
    return eval(circuit_name, cur_actions, 0) + eval(circuit_name, cur_actions, 1), cur_actions

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def MCTS(circuit_name, init_actions="", max_steps=10, simulations=100):
    root = TreeNode(init_actions)
    
    for _ in range(simulations):
        node = root
        
        # Selection
        node = select_best_node(node)
        
        # Expansion
        if not node.is_fully_expanded():
            for action in "0123456":
                actions = node.actions + action
                for child in node.children:
                    if actions == child.actions:
                        actions = "Fail"
                if actions == "Fail":
                    continue
                child_node = TreeNode(actions, parent=node)
                node.add_child(child_node)
                node = child_node
                break
        else:
            node = node.best_child(math.sqrt(2))

        # Simulation
        reward, rollout_actions = rollout(circuit_name, node.actions, max_steps - len(node.actions))

        # Backpropagation
        backpropagate(node, reward)

    # 返回访问次数最多的节点的路径
    best_child = max(root.children, key=lambda c: c.visits)
    node = best_child
    while node.parent is not None:
        node = node.parent

    return best_child.actions

def whole_MCTS(circuit_name, max_steps=10, simulations=100):
    actions = ""
    
    while len(actions) < max_steps:
        action = MCTS(circuit_name, actions, max_steps - len(actions), simulations)
        actions = actions + action
    
    return eval(circuit_name, actions, 0), actions

# ---

import re

def create(state):
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    circuit_name, actions = state.split('_')
    circuit_path = './InitialAIG/train/' + circuit_name + '.aig'
    next_state = './tmp/' + state + '.aig'
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ' ; ')
    abc_run_cmd = f"{yosys_path}yosys-abc -c \"read {circuit_path}; {action_cmd} read_lib {lib_file}; write {next_state}; print_stats\" > {log_file}"
    os.system(abc_run_cmd)

    return next_state

def evaluate(circuit_path, lib_file):
    abc_run_cmd = f"{yosys_path}yosys-abc -c \"read {circuit_path}; read_lib {lib_file}; map; topo; stime\" > {log_file}"
    os.system(abc_run_cmd)
    
    with open(log_file) as f:
        area_information = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        eval = float(area_information[-9]) * float(area_information[-4])
    return eval

def regularize(circuit_path, lib_file, eval):
    next_state = circuit_path.split('.aig')[0] + '_resyn.aig'
    next_bench = next_state.split('.aig')[0] + '.bench'
    resyn2_cmd = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abc_run_cmd = f"{yosys_path}yosys-abc -c \"read {circuit_path}; {resyn2_cmd} read_lib {lib_file}; write {next_state}; write_bench -l {next_bench}; map; topo; stime\" > {log_file}"
    os.system(abc_run_cmd)

    with open(log_file) as f:
        area_information = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(area_information[-9]) * float(area_information[-4])
        eval = 1 - eval / baseline
    return eval

def calc(circuit_name, actions):
    state = circuit_name + '_' + actions
    aig = create(state)
    eval = evaluate(aig, lib_file)
    eval = regularize(aig, lib_file, eval)
    print(circuit_name, '-', actions, ' ', eval)
    return eval

def getPath(circuit_name):
    print("beam_search:")
    beam_score, beam_actions = beam_search(circuit_name, 10, 3)
    best_score = beam_score
    print(beam_score, ' * ', beam_actions)
    print("A_star:")
    astar_score, astar_actions = Astar(circuit_name, 3)
    if best_score < astar_score:
        best_score = astar_score
    print(astar_score, ' * ', astar_actions)
    print("MCTS:")
    mcts_score, mcts_actions = whole_MCTS(circuit_name, 3, 50)
    if best_score < mcts_score:
        best_score = mcts_score
    print(mcts_score, ' * ', mcts_actions)
    
    print("dfs:")
    dfs_score, dfs_actions = dfs(circuit_name, 0, "", best_score * 0.5, "", 3)
    if best_score < dfs_score:
        best_score = dfs_score
    print(dfs_score, ' * ', dfs_actions)
    print("bfs:")
    bfs_score, bfs_actions = bfs(circuit_name, best_score * 0.5, "", 3)
    if best_score < bfs_score:
        best_score = bfs_score
    print(bfs_score, ' * ', bfs_actions)
    
    # final actions select
    beam_score = calc(circuit_name, beam_actions)
    astar_score = calc(circuit_name, astar_actions)
    mcts_score = calc(circuit_name, mcts_actions)
    dfs_score = calc(circuit_name, dfs_actions)
    bfs_score = calc(circuit_name, bfs_actions)
    
    best_score = beam_score
    best_actions = beam_actions
    if best_score < astar_score:
        best_score = astar_score
        best_actions = astar_actions
    if best_score < mcts_score:
        best_score = mcts_score
        best_actions = mcts_actions
    if best_score < dfs_score:
        best_score = dfs_score
        best_actions = dfs_actions
    if best_score < bfs_score:
        best_score = bfs_score
        best_actions = bfs_actions
    
    print(best_score, ' * ', best_actions)
    

getPath("alu2")
