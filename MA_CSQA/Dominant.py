import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_mean
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from datetime import datetime

class GCNModelAE(nn.Module):
    """
    图自编码器模型，包含属性重构和结构重构
    """
    def __init__(self, in_channels, hidden_channels, latent_channels, dropout=0.):
        super(GCNModelAE, self).__init__()
        self.dropout = dropout
        
        # 编码器
        self.encoder_conv1 = GCNConv(in_channels, hidden_channels)
        self.encoder_conv2 = GCNConv(hidden_channels, latent_channels)
        
        # 属性解码器
        self.attr_decoder_conv1 = GCNConv(latent_channels, hidden_channels)
        self.attr_decoder_conv2 = GCNConv(hidden_channels, in_channels)
        
        # 结构解码器（内积）
        # self.struct_decoder = nn.Sequential(
        #     nn.Linear(latent_channels, hidden_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels, latent_channels)
        # )
        self.struct_decoder_conv = GCNConv(latent_channels, hidden_channels)
        
    def encode(self, x, edge_index):
        # 编码过程
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode_attributes(self, x, edge_index):
        # 属性重构
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attr_decoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attr_decoder_conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode_structure(self, x, edge_index):
        # 结构重构（邻接矩阵重构）
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.struct_decoder_conv(x, edge_index)
        x = F.relu(x)
        adj_recon = torch.sigmoid(torch.mm(x, x.t()))
        return adj_recon
    
    def forward(self, x, edge_index):
        # 编码
        z = self.encode(x, edge_index)
        # 解码
        x_recon = self.decode_attributes(z, edge_index)
        adj_recon = self.decode_structure(z, edge_index)
        
        return x_recon, adj_recon, z

class AnomalyDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, data, optimizer, epochs=200):
        """
        训练模型
        """
        self.model.train()
        best_loss = float('inf')
        
        # 准备数据
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        adj = to_dense_adj(edge_index)[0]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, adj_recon, _ = self.model(x, edge_index)
            
            # 计算损失
            attr_loss = F.mse_loss(x_recon, x)
            struct_loss = F.binary_cross_entropy(adj_recon, adj)
            
            # 总损失
            loss = 0.8 * attr_loss + 0.2 * struct_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, '
                      f'Attr Loss: {attr_loss.item():.4f}, Struct Loss: {struct_loss.item():.4f}')
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def detect_anomalies(self, data, threshold=0.5):
        """
        检测异常节点
        """
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            adj = to_dense_adj(edge_index)[0]
            
            # 获取重构结果
            x_recon, adj_recon, z = self.model(x, edge_index)
            
            # 计算异常分数
            attr_errors = torch.mean((x - x_recon) ** 2, dim=1)
            struct_errors = torch.mean((adj - adj_recon) ** 2, dim=1)
            
            # 综合分数
            anomaly_scores = (attr_errors + struct_errors) / 2
            
            # 确定异常标签
            anomaly_labels = (anomaly_scores > threshold).float()
            
            # 转到CPU并转换为numpy
            scores = anomaly_scores.cpu().numpy()
            labels = anomaly_labels.cpu().numpy()
            
            # 为每个节点创建详细信息
            node_details = []
            for i in range(len(scores)):
                node_details.append({
                    'node_id': i,
                    'anomaly_score': float(scores[i]),
                    'is_anomaly': bool(labels[i]),
                    'attr_error': float(attr_errors[i].cpu()),
                    'struct_error': float(struct_errors[i].cpu())
                })
            
            # 按异常分数排序
            sorted_indices = np.argsort(-scores)
            
            return {
                'anomaly_scores': scores,
                'anomaly_labels': labels,
                'node_details': node_details,
                'sorted_indices': sorted_indices,
                'num_anomalies': int(labels.sum()),
                'threshold_used': threshold
            }
    
    def save_results(self, results, output_dir='output'):
        """
        保存检测结果
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        with open(os.path.join(output_dir, f'anomaly_detection_{timestamp}.txt'), 'w') as f:
            f.write("异常检测结果:\n")
            f.write(f"检测到的异常节点数量: {results['num_anomalies']}\n")
            f.write(f"使用的阈值: {results['threshold_used']}\n\n")
            
            f.write("节点详细信息 (按异常分数排序):\n")
            for idx in results['sorted_indices']:
                node = results['node_details'][idx]
                f.write(f"节点 {node['node_id']}:\n")
                f.write(f"  异常分数: {node['anomaly_score']:.4f}\n")
                f.write(f"  属性重构误差: {node['attr_error']:.4f}\n")
                f.write(f"  结构重构误差: {node['struct_error']:.4f}\n")
                f.write(f"  是否异常: {node['is_anomaly']}\n\n")
        
        # 保存分数
        np.save(os.path.join(output_dir, f'anomaly_scores_{timestamp}.npy'), results['anomaly_scores'])
        
        print(f"结果已保存到 {output_dir} 目录")

def parse_args():
    parser = argparse.ArgumentParser(description='GAE-based Anomaly Detection')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent layer dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Run mode')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    return parser.parse_args()

def main():
    args = parse_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    # 这里需要根据你的具体数据格式进行修改
    # data = YourDataLoader()
    
    # 初始化模型
    model = GCNModelAE(
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        latent_channels=args.latent_dim,
        dropout=args.dropout
    )
    
    detector = AnomalyDetector(model, device)
    
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        detector.train(data, optimizer, args.epochs)
        print("训练完成，模型已保存")
    
    else:  # test mode
        # 加载预训练模型
        model.load_state_dict(torch.load(args.model_path))
        print(f"加载模型从: {args.model_path}")
        
        # 进行异常检测
        results = detector.detect_anomalies(data, args.threshold)
        
        # 保存结果
        detector.save_results(results)
        
        # 如果有真实标签，计算评估指标
        if hasattr(data, 'y'):
            auc = roc_auc_score(data.y.cpu().numpy(), results['anomaly_scores'])
            ap = average_precision_score(data.y.cpu().numpy(), results['anomaly_scores'])
            print(f"AUC: {auc:.4f}")
            print(f"AP: {ap:.4f}")

if __name__ == '__main__':
    main()