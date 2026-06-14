import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATConv
from einops import rearrange
from datetime import datetime
from Dominant import GCNModelAE
from TAM import TAMModel, GATSCL
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cal_AUROC(labels, probs, graph_size=8):
    pred_y = torch.zeros(graph_size, dtype=torch.long)
    true_y = torch.zeros(graph_size, dtype=torch.long)
    true_y[labels] = 1
    pred_y[probs] = 1
    labels = true_y.numpy()
    probs = pred_y.numpy()
    score_AUROC = roc_auc_score(labels, probs)
    # score['AUPRC'] = average_precision_score(labels, probs)
    return score_AUROC

class MyGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, out_channels, heads=8, num_layers=2, edge_dim=None):
        super(MyGAE, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        
        # 计算边特征的实际维度
        self.time_steps, self.edge_feat_dim = edge_dim
        self.total_edge_dim = self.time_steps * self.edge_feat_dim
        
        # 编码器层
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(GATConv(in_channels, hidden_channels, heads=heads, edge_dim=self.total_edge_dim))
        
        for _ in range(num_layers - 2):
            self.encoder_convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=self.total_edge_dim)
            )
        
        # 最后一层编码器，压缩到潜在空间
        self.encoder_convs.append(
            GATConv(hidden_channels * heads, latent_channels, heads=1, edge_dim=self.total_edge_dim)
        )
        
        # 解码器层
        self.decoder_convs = nn.ModuleList()
        self.decoder_convs.append(GATConv(latent_channels, hidden_channels, heads=heads, edge_dim=self.total_edge_dim))
        
        for _ in range(num_layers - 2):
            self.decoder_convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=self.total_edge_dim)
            )
        
        # 最后一层解码器，恢复到原始维度
        self.decoder_convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, edge_dim=self.total_edge_dim)
        )
        
        # 边属性重构器
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.time_steps * self.edge_feat_dim)
        )
        
    def encode(self, x, edge_index, edge_attr):
        # 处理边属性：将 [num_edges, time_steps, dim] 转换为 [num_edges, time_steps * dim]
        edge_attr = edge_attr.view(edge_attr.size(0), -1)
        
        # 编码过程
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index, edge_attr)
            if i != len(self.encoder_convs) - 1:  # 不是最后一层
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        return x
        
    def decode(self, z, edge_index, edge_attr):
        # 处理边属性：将 [num_edges, time_steps, dim] 转换为 [num_edges, time_steps * dim]
        edge_attr = edge_attr.view(edge_attr.size(0), -1)
        
        # 解码过程
        for i, conv in enumerate(self.decoder_convs):
            z = conv(z, edge_index, edge_attr)
            if i != len(self.decoder_convs) - 1:  # 不是最后一层
                z = F.relu(z)
                z = F.dropout(z, p=0.2, training=self.training)
        
        return z
        
    def reconstruct_edge_attr(self, z, edge_index):
        # 获取边两端的节点表示
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        
        # 连接源节点和目标节点的表示
        edge_input = torch.cat([z_src, z_dst], dim=1)
        
        # 预测边属性
        pred_edge_attr = self.edge_predictor(edge_input)
        # 重塑为原始维度 [num_edges, time_steps, edge_feat_dim]
        pred_edge_attr = pred_edge_attr.view(-1, self.time_steps, self.edge_feat_dim)
        
        return pred_edge_attr
        
    def forward(self, x, edge_index, edge_attr):
        # 编码
        z = self.encode(x, edge_index, edge_attr)
        
        # 解码节点特征
        reconstructed_x = self.decode(z, edge_index, edge_attr)
        
        # 重构边属性
        reconstructed_edge_attr = self.reconstruct_edge_attr(z, edge_index)
        
        return reconstructed_x, reconstructed_edge_attr

def reconstruction_loss(original, reconstructed, edge_index):
    """计算重构误差"""
    # 计算节点特征的重构误差
    node_loss = nn.MSELoss()(original, reconstructed)
    
    # 计算结构重构误差（可选）
    # 这里可以添加额外的结构相似性损失
    
    return node_loss

def train(model, train_loader, optimizer, defend_type, device):
    model.train()
    total_loss = 0
    num_batches = 0
    topk = 3
    num_auc = 0
    # 定义初始Tensor
    anamaly_ori_idx = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]).to(device)
    
    for data in train_loader:
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        # print(edge_index.shape)
        # import ipdb;ipdb.set_trace()
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        # import ipdb;ipdb.set_trace()
        optimizer.zero_grad()

        # import ipdb;ipdb.set_trace()
        
        # TAM
        if defend_type == "TAM":
            node_emb, feat1, feat2 = model(x, edge_index)
            # node_emb = model.encode(x, edge_index)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            loss_1, _ = model.max_message(node_emb, adj)
            reg_loss = model.reg_edge(node_emb, adj)
            loss = loss_1 + 1 * reg_loss

        # myself
        elif defend_type == "SCL":
            anamaly_idx = anamaly_ori_idx[torch.randperm(len(anamaly_ori_idx))]
            noise = torch.randn_like(x)
            noise = F.normalize(noise, dim=1)
            noise_magnitude = 0.8 * torch.norm(x, dim=1, keepdim=True)
            noised_embeddings = x + noise * noise_magnitude * anamaly_idx.unsqueeze(1)
            x = noised_embeddings
            node_emb = model.encode(x, edge_index)
            loss = model.neg_all(node_emb, anamaly_idx)

            # import ipdb;ipdb.set_trace()
            num_nodes = node_emb.size(0)
            adj = torch.eye(num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            message = model.inference_new(node_emb, adj)
            _, predicts = torch.topk(-message, topk)
            labels = torch.where(anamaly_idx == 1)[0].tolist()
            num_auc += cal_AUROC(labels, predicts.tolist())
        
        # Dominant
        elif defend_type == "Dominant":
            x_recon, adj_recon, z = model(x, edge_index=edge_index)
            attr_loss = F.mse_loss(x_recon, x)
            num_nodes = x.size(0)
            adj = torch.eye(num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            struct_loss = F.binary_cross_entropy(adj_recon, adj)
            loss = 0.8 * attr_loss + 0.2 * struct_loss

        # myself class
        # anamaly_idx = anamaly_ori_idx[torch.randperm(len(anamaly_ori_idx))]
        # noise = torch.randn_like(x)
        # noise = F.normalize(noise, dim=1)
        # noise_magnitude = 0.8 * torch.norm(x, dim=1, keepdim=True)
        # noised_embeddings = x + noise * noise_magnitude * anamaly_idx.unsqueeze(1)
        # x = noised_embeddings
        # loss = model.ana_class(x, edge_index, anamaly_idx)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1), num_auc / max(len(train_loader), 1)

def test(model, defend_type, test_data, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    topk = 3
    num_auc = 0
    idx = 0
    
    with torch.no_grad():
        for data in test_data:
            attack_idx = data['attacker_idxes']
            # import ipdb;ipdb.set_trace()
            x, edge_index, edge_attr = torch.from_numpy(data['features']).to(device), torch.from_numpy(data['edge_index']).to(device), torch.from_numpy(data['edge_attr']).to(device)
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(torch.from_numpy(data['features'])))
            
            # TAM
            if defend_type in ["TAM", "SCL"]:
                z = model.encode(x, edge_index)
                num_nodes = x.size(0)
                adj = torch.eye(num_nodes)
                adj[edge_index[0], edge_index[1]] = 1.0
                message = model.inference_new(z, adj)
                _, predicts = torch.topk(-message, topk)
                num_auc += cal_AUROC(attack_idx, predicts)
                # print(cal_AUROC(attack_idx, predicts))
            
            elif defend_type == "Dominant":
                x_recon, adj_recon, z = model(x, edge_index=edge_index)
                attr_loss = F.mse_loss(x_recon, x)
                num_nodes = x.size(0)
                adj = torch.zeros((num_nodes, num_nodes), device=device)
                adj[edge_index[0], edge_index[1]] = 1.0
                struct_loss = F.binary_cross_entropy(adj_recon, adj)
                loss = 0.8 * attr_loss + 0.2 * struct_loss

            # total_loss += loss.item()
            # num_batches += 1
    
    return num_auc / len(test_data)  # 避免除以0

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train Graph Autoencoder")
    
    parser.add_argument("--dataset_path", type=str, default="./ModelTrainingSet/memory_attack/dataset.pkl", help="Save path of the dataset")
    parser.add_argument("--test_path", type=str, default="./ModelTrainingSet/memory_attack/test_dataset.pkl", help="Save path of the dataset")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=512)  # 添加潜在空间维度参数
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--save_dir", type=str, default="./checkpoint_un2")
    parser.add_argument("--defend_type", type=str, default="SCL")
    parser.add_argument("--rep_type", type=int, default=0)
    
    args = parser.parse_args()
    
    normalized_path = os.path.normpath(args.dataset_path)
    #import ipdb;ipdb.set_trace()
    parts = normalized_path.split(os.sep)
    dataset = parts[-1]
    args.save_dir = os.path.join(args.save_dir, dataset)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time_str}-defend_type_{args.defend_type}-hiddim_{args.hidden_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}-rep_type_{args.rep_type}.pth"
    args.save_path = os.path.join(args.save_dir, filename)
    
    return args

def main():
    args = parse_arguments()
    set_seed(42)
    # 数据加载
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    import pickle
    with open(args.test_path, 'rb') as f:  # 注意模式是 'rb' (二进制读取)
        test_data = pickle.load(f)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(test_data)}")
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 修改验证集的batch_size，确保每个batch至少有两个样本
    test_batch_size = 1
    print(f"使用的测试batch大小: {test_batch_size}")
    
    # 获取输入维度
    example = train_dataset[0]
    edge_attr = example.edge_attr
    in_channels = edge_attr.size(-1)  # 使用边特征的维度作为输入维度
    
    # 设置设备
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
    # model = ContrastiveGAE(
    #     in_channels=in_channels,
    #     hidden_channels=args.hidden_dim,
    #     latent_channels=args.latent_dim,
    #     heads=args.num_heads,
    #     num_layers=args.num_layers
    # )

    # TAM + myself
    if args.defend_type in ["TAM"]:
        model = TAMModel(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=args.latent_dim,
            dropout=0,
            readout='avg'
        )
    elif args.defend_type == "Dominant":
        model = GCNModelAE(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            latent_channels=args.latent_dim,
            dropout=0.
        )
    elif args.defend_type == "SCL":
        model = GATSCL(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=args.latent_dim,
            type=args.rep_type
        )

    model.to(device)
    #detector.to(device)
    
    # 确保模型参数是float32
    for param in model.parameters():
        param.data = param.data.float()
    
    # checkpoint = torch.load("checkpoint_un2/mmlu/20250616_135909-hiddim_1024-latent_512-heads_8-layers_2-epochs_50-lr_0.001-temp_0.1.pth", map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint)
        
    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    best_loss = float('inf')
    best_auc = 0
    
    # 训练循环
    for epoch in range(args.epochs):
        train_loss, train_auc = train(model, trainloader, optimizer, args.defend_type, device=device)
        test_auc = test(model, args.defend_type, test_data, device=device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} || Training Loss: {train_loss:.4f} || Training AUC: {train_auc:.4f} || Test auc: {test_auc:.4f} || Samples in batch: 1")
        
        # if test_auc > best_auc:  # 只在测试损失大于0时保存
        #     best_auc = test_auc 
        #     torch.save(model.state_dict(), args.save_path)
        #     print("Model saved!")
        if train_auc > best_auc:  # 只在测试损失大于0时保存
            best_auc = train_auc
            torch.save(model.state_dict(), args.save_path)
            print("Model saved!")

if __name__ == "__main__":
    main()