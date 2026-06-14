import os
from model import MyGAT
from agents import AgentGraphWithDefense, AgentGraph
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from utils import get_sentence_embedding
from einops import rearrange
from torch_scatter import scatter_mean
import argparse 
from datetime import datetime
import asyncio
import copy
import time
from utils import get_adj_matrix

#test
from sklearn.cluster import DBSCAN
# from train_un import MyGAE
from train_un2 import ContrastiveGAE
from Dominant import GCNModelAE
import torch.nn.functional as F
from TAM import TAMModel, GATSCL
from Prem_gad import PREMModel


def response2embeddings(responses): 
    embeddings = [None for _ in range(len(responses))]
    for agent_idx, agent_response in responses: 
        embeddings[agent_idx] = get_sentence_embedding(agent_response)
    
    embeddings = np.array(embeddings)
    return embeddings


def embeddings2graph(embeddings, adj_matrix, use_emb="first"):
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()))
    edge_attr = torch.tensor(np.array(embeddings))[:, edge_index[1]]  # 仅仅针对统一回复的情况
    # import ipdb;ipdb.set_trace()
    #torch.tensor(np.array(response_embeddings)).shape  1*8*484
    #取出入读的特征 edge_attr 1*56*384
    # import ipdb;ipdb.set_trace()
    x = edge_attr[0, :] #第一轮 初始回复 边特征
    # x = edge_attr[-1, :] #第一轮 初始回复 边特征
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(embeddings[0])) #聚合操作
    
    # 扩展边特征维度以匹配模型要求
    edge_attr = edge_attr.transpose(0, 1)  # [num_edges, num_turns, hidden_dim] 交换维度
    # 使用重复来扩展维度
    edge_attr_expanded = edge_attr.reshape(edge_attr.size(0), -1)  # [num_edges, num_turns * hidden_dim]
    edge_attr_expanded = torch.nn.functional.pad(
        edge_attr_expanded,
        (0, 1536 - edge_attr_expanded.size(1)),
        mode='replicate'
    )  # 扩展到1536维度
     #edge_attr 边特征， 是多次对话的入度特征，多轮拼接
     #x 节点特征，其实就是初始第一轮的特征
    return x, edge_index, edge_attr_expanded


async def defense_communication(ag:AgentGraphWithDefense, gnn: MyGAT, qa_data, adj_m: np.ndarray,  num_dialogue_turns, defend_type, topk, attacker_idxes): 
    communication_data = []
    response_embeddings = []
    initial_responses = await ag.afirst_generate(qa_data)
    embeddings = response2embeddings(initial_responses)
    response_embeddings.append(embeddings)
    x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)
    
    # import ipdb;ipdb.set_trace()

    # TAM
    if defend_type == "TAM":
        z, feat1, feat2 = gnn(x, edge_index)
        # z = gnn.encode(x, edge_index)
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        message = gnn.inference(z, adj)
        _, predicts = torch.topk(-message, topk)

    # myself SCL
    elif defend_type == "SCL":
        #import ipdb;ipdb.set_trace()
        z = gnn.encode(x, edge_index)
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        message = gnn.inference_new(z, adj)
        _, predicts = torch.topk(-message, topk)
        # scores = 1 - gnn.normalize_score(message.detach().cpu().numpy())
        # predicts = (torch.from_numpy(scores)>0.5).int()
        # print(message)
        # print(message.max() - message.min())
        # if (message.max() - message.min()>1):
        #     import ipdb;ipdb.set_trace()
    
    # Dominant
    elif defend_type == "Dominant":
        x_recon, adj_recon, z = gnn(x, edge_index)
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        attr_errors = torch.mean((x - x_recon) ** 2, dim=1)
        struct_errors = torch.mean((adj - adj_recon) ** 2, dim=1)
        anomaly_scores = 0.8 * attr_errors + 0.2 * struct_errors
        _, predicts = torch.topk(anomaly_scores, topk)

    # PREM-GAD
    elif defend_type == "PREM":
        # 获取异常分数
        anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
        # 选择topk个异常分数最高的节点
        _, predicts = torch.topk(anomaly_scores.squeeze(), topk)

    # myself class
    # y = gnn.ana_infer(x, edge_index)
    # y = torch.sigmoid(y)
    # scores = 1 - gnn.normalize_score(message.detach().cpu().numpy())
    # predicts = (torch.from_numpy(scores)>0.5).int()

    # DGI
    # batch = torch.zeros(x.shape[0], dtype=torch.long)
    # # 获取节点表示和图表示
    # z_proj, graph_proj = gnn(x, edge_index, batch)
    # # 归一化特征
    # z_proj = F.normalize(z_proj, dim=1)
    # graph_proj = F.normalize(graph_proj, dim=1)
    # # 计算每个节点与图表示的余弦相似度
    # similarities = torch.mm(z_proj, graph_proj.t())  # [N, 1]
    # node_to_graph_sim = similarities.squeeze()  # [N]
    # # 计算异常分数 (1 - similarity，使得分数越高表示越异常)
    # anomaly_scores = 1 - node_to_graph_sim
    # # 根据阈值确定异常标签
    # anomaly_labels = (anomaly_scores > threshold).float()
    
    # 全 1
    # predicts = torch.ones(8).int()

    # DBSCAN
    # clustering = DBSCAN(eps=0.5, min_samples=3).fit(x)
    # predicts = torch.from_numpy(-1 * clustering.labels_) 


    # for idx, predict in enumerate(predicts): 
    #     if predict == 1: 
    #         ag.agents[idx].set_role("attacker")
    
    original_predicts = predicts.tolist()
    for predict in predicts:
        ag.agents[predict].set_role("attacker")
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        embeddings = response2embeddings(responses)
        response_embeddings.append(embeddings)
        x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)
        
        for idx in range(len(ag.agents)):
            ag.agents[idx].set_role("normal")

        # TAM + GCN_e
        if defend_type == "TAM":
            z, feat1, feat2 = gnn(x, edge_index)
            # z = gnn.encode(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = gnn.inference(z, adj)
            _, predicts = torch.topk(-message, topk)

        # myself SCL
        elif defend_type == "SCL":
            z = gnn.encode(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = gnn.inference_new(z, adj)
            _, predicts = torch.topk(-message, topk)
        
        # Dominant
        elif defend_type == "Dominant":
            x_recon, adj_recon, z = gnn(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            attr_errors = torch.mean((x - x_recon) ** 2, dim=1)
            struct_errors = torch.mean((adj - adj_recon) ** 2, dim=1)
            anomaly_scores = 0.5 * attr_errors + 0.5 * struct_errors
            _, predicts = torch.topk(anomaly_scores, topk)
        
        # PREM-GAD
        elif defend_type == "PREM":
            # 获取异常分数
            anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
            # 选择topk个异常分数最高的节点
            _, predicts = torch.topk(anomaly_scores.squeeze(), topk)

        # DBSCAN
        # clustering = DBSCAN(eps=0.5, min_samples=3).fit(x)
        # predicts = torch.from_numpy(-1 * clustering.labels_)

        # for idx, predict in enumerate(predicts): 
        #     if predict == 1: 
        #         ag.agents[idx].set_role("attacker")
        for a_i in range(len(responses)):
            ag.agents[a_i].set_role("normal")
        for predict in predicts:
            ag.agents[predict].set_role("attacker")
        communication_data.append(responses)

    return communication_data, original_predicts


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str, default="./agent_graph_dataset/tool_attack/test/dataset.json", help="Save path of the dataset")
    parser.add_argument("--graph_type", type=str, choices=["random", "chain", "tree", "star"], default="random")
    parser.add_argument("--gnn_checkpoint_path", type=str, default="./checkpoint/tool_attack/checkpoint.pth")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=60)
    
    parser.add_argument("--defend_type", type=str, default="SCL", choices=["SCL", "TAM", "Dominant", "PREM"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--rep_type", type=int, default=0)
    
    # PREM-GAD specific parameters
    parser.add_argument("--prem_k", type=int, default=2, help="PREM aggregation steps")
    args = parser.parse_args()

    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset, args.graph_type)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_defense = f"{current_time_str}-defense_type_{args.defend_type}-topk_{args.topk}-model_type_{args.model_type}-rep_type_{args.rep_type}.json"
    args.save_path_with_defense = os.path.join(args.save_dir, filename_defense)

    return args


async def main(): 
    args = parse_arguments()
    filepath = args.dataset_path
    graph_type = args.graph_type
    with open(filepath, "r") as f:
        dataset = json.load(f)
    dataset_len = len(dataset)
    dataset = dataset[-args.samples:]
    # import ipdb;ipdb.set_trace()
    num_dialogue_turns = len(dataset[0]["communication_data"])-1


    edge_dim = (4, 384)  # 保持原始维度结构

    # TAM
    if args.defend_type in ["TAM"]:
        gnn = TAMModel(
            in_channels=384,
            hidden_channels=1024,
            out_channels=512,
            dropout=0,
            readout='avg'
        )
    # Dominant
    elif args.defend_type == "Dominant":
        gnn = GCNModelAE(
            in_channels=384,
            hidden_channels=1024,
            latent_channels=512,
            dropout=0.
        )
    elif args.defend_type == "SCL":
        gnn = GATSCL(
            in_channels=384,
            hidden_channels=1024,
            out_channels=512,
            type=args.rep_type
        )
    elif args.defend_type == "PREM":
        gnn = PREMModel(
            n_in=384,
            n_hidden=1024,
            k=args.prem_k
        )
    
    checkpoint = torch.load(args.gnn_checkpoint_path, map_location=torch.device('cpu'))
    gnn.load_state_dict(checkpoint)

    final_dataset_nd = []
    final_dataset_wd = []
    for d in tqdm(dataset): 
        if graph_type == "random": 
            adj_m = np.array(d["adj_matrix"])
        elif graph_type in ["chain", "tree", "star"]: 
            adj_m = get_adj_matrix(graph_type, len(d["adj_matrix"]))
        else:
            raise Exception(f"Unknown graph type: {graph_type}! Can only be one of [random, chain, tree, star]")
        attacker_idxes = d["attacker_idxes"]
        system_prompts = d["system_prompts"]
        user_instruction = d["user_instruction"]
        available_tool_names = d["available_tool_names"]
        available_tool_descriptions = d["available_tool_descriptions"]
        agent_scratchpad = d["agent_scratchpad"]
        case = (user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)

        try:
            agwd = AgentGraphWithDefense(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph with defense
            communication_data_defense, original_predicts = await defense_communication(agwd, gnn, case, adj_m, num_dialogue_turns, args.defend_type, args.topk, attacker_idxes)
        except Exception as e: 
            print(e)
            continue
        
        d_wd = copy.deepcopy(d)
        d_wd["communication_data"] = communication_data_defense
        d_wd["original_predicts"] = original_predicts
        final_dataset_wd.append(d_wd)
    
    with open(args.save_path_with_defense, "w") as file:
        json.dump(final_dataset_wd, file, indent=None) 


if __name__ == "__main__": 
    asyncio.run(main())
