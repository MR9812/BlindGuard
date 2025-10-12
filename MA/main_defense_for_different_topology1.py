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
import os

from sklearn.cluster import DBSCAN
from train_un1 import MyGAE
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

def embeddings2graph(embeddings, adj_matrix):
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()))
    edge_attr = torch.tensor(np.array(embeddings))[:, edge_index[1]] 
    x = edge_attr[0, :]
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(embeddings[0]))
    
    edge_attr = edge_attr.transpose(0, 1)
    edge_attr_expanded = edge_attr.reshape(edge_attr.size(0), -1)
    edge_attr_expanded = torch.nn.functional.pad(
        edge_attr_expanded,
        (0, 1536 - edge_attr_expanded.size(1)),
        mode='replicate'
    )
    return x, edge_index, edge_attr_expanded


async def no_defense_communication(ag: AgentGraph, query, context, num_dialogue_turns): 
    communication_data = []
    initial_responses = await ag.afirst_generate(query, context)
    communication_data.append(initial_responses)
    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        communication_data.append(responses)
    return communication_data


async def defense_communication(ag:AgentGraphWithDefense, gnn: MyGAT, query, context, adj_m: np.ndarray,  num_dialogue_turns, defend_type, topk): 
    communication_data = []
    identified_attackers = []
    response_embeddings = []
    initial_responses = await ag.afirst_generate(query, context)
    embeddings = response2embeddings(initial_responses)
    response_embeddings.append(embeddings)
    x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)
    
    # TAM
    if defend_type == "TAM":
        z, feat1, feat2 = gnn(x, edge_index)
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
        anomaly_scores = 0.8 * attr_errors + 0.2 * struct_errors
        _, predicts = torch.topk(anomaly_scores, topk)

    # PREM-GAD
    elif defend_type == "PREM":
        anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
        _, predicts = torch.topk(anomaly_scores.squeeze(), topk)

    original_predicts = predicts.tolist()

    for predict in predicts:
        ag.agents[predict].set_role("attacker")
    communication_data.append(initial_responses)
    identified = []
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
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = gnn.inference(z, adj)
            _, predicts = torch.topk(-message, topk)

        # myself SCL
        elif defend_type == "SCL":
            predicts = original_predicts
        
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
            anomaly_scores = gnn.get_anomaly_scores(x, edge_index)
            _, predicts = torch.topk(anomaly_scores.squeeze(), topk)

        for predict in predicts:
            ag.agents[predict].set_role("attacker")
            if predict not in identified: 
                identified.append(predict)
        communication_data.append(responses)
        identified_attackers.append(copy.deepcopy(identified))

    return communication_data, original_predicts


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str, default="./agent_graph_dataset/memory_attack/test/dataset.json", help="Save path of the dataset")
    parser.add_argument("--graph_type", type=str, choices=["random", "chain", "tree", "star"], default="star")
    parser.add_argument("--gnn_checkpoint_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--defend_type", type=str, default="SCL", choices=["SCL", "TAM", "Dominant", "PREM"])
    parser.add_argument("--rep_type", type=int, default=0)
    
    parser.add_argument("--prem_k", type=int, default=2, help="PREM aggregation steps")

    args = parser.parse_args()

    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset, args.graph_type)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_no_defense = f"{current_time_str}-no_defense-model_type_{args.model_type}.json"
    filename_defense = f"{current_time_str}-defense_type_{args.defend_type}-topk_{args.topk}-model_type_{args.model_type}-rep_type_{args.rep_type}.json"
    args.save_path_no_defense = os.path.join(args.save_dir, filename_no_defense)
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
    num_dialogue_turns = len(dataset[0]["communication_data"])-1


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
        query = d["query"]
        context = d["adv_texts"]

        try:
            agwd = AgentGraphWithDefense(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph with defense
            communication_data_defense, original_predicts = await defense_communication(agwd, gnn, query, context, adj_m, num_dialogue_turns, args.defend_type, args.topk)
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