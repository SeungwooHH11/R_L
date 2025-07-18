import numpy as np
import pandas as pd
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
TORCH_USE_CUDA_DSA=1
device = 'cuda'
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

class MultiHeadGATLayerMerged(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2):
        super().__init__()
        self.directions = ['U', 'D', 'R', 'L']
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.W = nn.ModuleDict()
        self.attn = nn.ParameterDict()

        for d in self.directions:
            self.W[d] = nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)
            ])
            self.attn[d] = nn.ParameterList([
                nn.Parameter(torch.empty(2 * out_dim)) for _ in range(num_heads)
            ])
            for a in self.attn[d]:
                nn.init.xavier_uniform_(a.data.view(2, -1), gain=1.414)

    def forward(self, x, A_dict):
        """
        x: (B, N, F)
        A_dict: {'U': (N, N), ...}
        """
        B, N, _ = x.size()
        device = x.device
        e_total = torch.full((B, N, N), float('-inf'), device=device)
        Wh_dict = {}

        for d in self.directions:
            Wh_dict[d] = []
            for h in range(self.num_heads):
                Wh = self.W[d][h](x)  # (B, N, F')
                Wh_dict[d].append(Wh)

                a = self.attn[d][h]  # (2F')
                a_src, a_dst = a[:self.out_dim], a[self.out_dim:]

                # Efficient attention score computation
                e_src = torch.einsum('bnd,d->bn', Wh, a_src)  # (B, N)
                e_dst = torch.einsum('bnd,d->bn', Wh, a_dst)  # (B, N)
                e = F.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B, N, N)

                A = A_dict[d].unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
                e = e.masked_fill(A == 0, float('-inf'))
                e_total = torch.where(A.bool(), e, e_total)

        # Shared softmax across all directions and heads
        alpha = F.softmax(e_total, dim=2)  # (B, N, N)

        # Message passing
        out = torch.zeros(B, N, self.out_dim, device=device)
        for d in self.directions:
            A = A_dict[d].unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
            alpha_d = alpha * A
            for h in range(self.num_heads):
                out += torch.bmm(alpha_d, Wh_dict[d][h])  # (B, N, F')

        out = out / self.num_heads
        return out
        
class GATModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, Up_A_hat, Down_A_hat, Right_A_hat, Left_A_hat, num_heads=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([MultiHeadGATLayerMerged(hidden_dim, hidden_dim, num_heads) for _ in range(3)])
        self.a = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.A_dict = {'U': Up_A_hat, 'D': Down_A_hat, 'R': Right_A_hat, 'L': Left_A_hat}
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, X):
        X = self.embedding(X)
        for layer in self.gat_layers:
            X = layer(X, self.A_dict) + self.a * X  # residual 연결
        return X


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.2):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim // num_heads
        
        # Linear transformation for input feature
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.zeros(size=(num_heads, self.out_dim)))
        self.a_dst = nn.Parameter(torch.zeros(size=(num_heads, self.out_dim)))

        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, A, X):
        """
        A: Adjacency matrix (b, n, n)
        X: Node features (b, n, f)
        """
        b, n, _ = X.shape  # Batch size, number of nodes

        # Apply linear transformation
        X_trans = self.W(X)  # (b, n, out_dim)

        # Multi-head attention
        X_split = X_trans.view(b, n, self.num_heads, self.out_dim)  # (b, n, heads, out_dim)

        # Compute attention coefficients
        attn_src = torch.einsum("bnhd,hd->bnh", X_split, self.a_src)  # (b, n, heads)
        attn_dst = torch.einsum("bnhd,hd->bnh", X_split, self.a_dst)  # (b, n, heads)

        attn_matrix = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (b, n, n, heads)
        attn_matrix = F.leaky_relu(attn_matrix, negative_slope=0.2)
        
        # Mask out non-existing edges (use adjacency matrix)
        attn_matrix = attn_matrix.masked_fill(A.unsqueeze(-1) == 0, float("-inf"))

        # Apply softmax normalization
        attn_matrix = F.softmax(attn_matrix, dim=2)
        #attn_matrix = self.dropout(attn_matrix)  # (b, n, n, heads)

        # Apply attention mechanism
        out = torch.einsum("bnnk,bnkd->bnkd", attn_matrix, X_split)  # (b, n, heads, out_dim)

        # Concatenate multi-head results
        out = out.reshape(b, n, -1)  # (b, n, out_dim * heads)
        return out

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, A_hat, num_heads=4):
        super(GATModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim, hidden_dim, num_heads)
        self.init_weights()
        self.A = A_hat  # (b, n, n)

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.modules():
            if isinstance(m, GATLayer):
                m.apply(self._init_gat_weights)

    def _init_gat_weights(self, m):
        if isinstance(m, GATLayer):
            nn.init.xavier_uniform_(m.W.weight)
            nn.init.xavier_uniform_(m.a_src)
            nn.init.xavier_uniform_(m.a_dst)

    def forward(self, X):
        """
        X: (b, n, f) - Node feature matrix
        """
        X = self.embedding(X)  # (b, n, hidden_dim)
        X = self.gat1(self.A, X) 
        X = self.gat2(self.A, X) 
        X = self.gat3(self.A, X) 
        return X


# GCN Layer 정의
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A_hat, X):
        return F.relu(torch.matmul(torch.matmul(A_hat, X), self.weight))

# GCNModel1 정의
class GCNModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim,A_hat):
        super(GCNModel1, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.init_weights()
        self.A=A_hat

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.modules():
            if isinstance(m, GCNLayer):
                m.init_weights()

    def forward(self,  X):
        X=self.embedding(X)
        X = self.gcn1(self.A, X)/4.0
        X = self.gcn2(self.A, X)/4.0
        X = self.gcn3(self.A, X)/4.0
        
        return X

# GCNModel2 정의
class GCNModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim,Up_A_hat, Down_A_hat, Right_A_hat, Left_A_hat):
        super(GCNModel2, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(12)])
        self.a = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.init_weights()
        self.U=Up_A_hat
        self.D=Down_A_hat
        self.R=Right_A_hat
        self.L=Left_A_hat

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.gcn_layers:
            layer.init_weights()

    def forward(self, X):
        X = self.embedding(X)
        for i in range(0, 12, 4):
            X = (
                self.gcn_layers[i](self.U, X) +
                self.gcn_layers[i+1](self.D, X) +
                self.gcn_layers[i+2](self.R, X) +
                self.gcn_layers[i+3](self.L, X) +
                self.a * X
            )/4.0
        return X


# Actor 네트워크 (어디에 배치할지 결정)
class Actor_net(nn.Module):
    def __init__(self, hidden_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc4 = nn.Linear(int(hidden_dim/4), 1)
        self.initialize_weights()


    def forward(self, X, mask):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        
        X = X - mask * 1e8  #mask b, r*c, 1
        
        return F.softmax((X), dim=1)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)

# Critic 네트워크 (State Value 계산)
class Critic_net(nn.Module):
    def __init__(self, input_dim):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim/4))
        self.fc2 = nn.Linear(int(input_dim/4), int(input_dim/8))
        self.fc3 = nn.Linear(int(input_dim/8), int(input_dim/16))
        self.fc4 = nn.Linear(int(input_dim/16), 1)
        self.initialize_weights()


    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        
        return X
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PPO(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lookahead_block_num ,grid_size, learning_rate, lmbda, gamma, alpha, beta, epsilon, mod):
        super(PPO, self).__init__()
        self.r=grid_size[0]
        self.c=grid_size[1]
        adj_matrix,Up_A_hat,Down_A_hat,Left_A_hat,Right_A_hat=self.grid_to_adjacency_matrix(self.r, self.c)
        self.A=torch.tensor(adj_matrix,dtype=torch.float32).to(device)
        self.U=torch.tensor(Up_A_hat,dtype=torch.float32).to(device)
        self.D=torch.tensor(Down_A_hat,dtype=torch.float32).to(device)
        self.R=torch.tensor(Right_A_hat,dtype=torch.float32).to(device)
        self.L=torch.tensor(Left_A_hat,dtype=torch.float32).to(device)
        self.hidden_dim = hidden_dim
        self.input_dim=feature_dim
        self.mod=mod
        #self.embedding=nn.Linear()
        self.lookahead_block_num=lookahead_block_num
        if mod=='GCN1':
            self.gnn = GCNModel1(self.input_dim,self.hidden_dim,self.A).to(device)
        if mod=='GCN2':
            self.gnn = GCNModel2(self.input_dim,self.hidden_dim,self.U,self.D,self.R,self.L).to(device)
        if mod=='GAT':
            self.gnn=GATModel(self.input_dim,self.hidden_dim,self.A,num_heads=2)
        if mod=='GAT2':
            self.gnn = GATModel2(self.input_dim,self.hidden_dim,self.U,self.D,self.R,self.L,num_heads=2).to(device)
        self.Actor_net = Actor_net(self.hidden_dim+self.lookahead_block_num*feature_dim).to(device)
        self.Critic_net = Critic_net(self.r*self.c*self.hidden_dim+self.lookahead_block_num*feature_dim+1).to(device)
        if mod=='MLP':
            self.Actor_net = MLP((self.r*self.c+self.lookahead_block_num)*feature_dim,self.r*self.c).to(device)
            self.Critic_net =MLP((self.r*self.c+self.lookahead_block_num)*feature_dim+1,1).to(device)
        
        self.temperature = 1.5
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    
    def grid_to_adjacency_matrix(self,n, m):
        size = n * m
        adj_matrix = np.zeros((size, size), dtype=np.float32)
        Up_A_hat = np.zeros((size, size), dtype=np.float32)
        Down_A_hat = np.zeros((size, size), dtype=np.float32)
        Right_A_hat = np.zeros((size, size), dtype=np.float32)
        Left_A_hat = np.zeros((size, size), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                
                # 상 (i-1, j)
                if i > 0:
                    adj_matrix[idx][(i-1) * m + j] = 1
                    adj_matrix[(i-1) * m + j][idx] = 1
                    Up_A_hat[idx][(i-1) * m + j] = 1
                    Up_A_hat[(i-1) * m + j][idx] = 1
                    
    
                # 하 (i+1, j)
                if i < n - 1:
                    adj_matrix[idx][(i+1) * m + j] = 1
                    adj_matrix[(i+1) * m + j][idx] = 1
                    Down_A_hat[idx][(i+1) * m + j] = 1
                    Down_A_hat[(i+1) * m + j][idx] = 1
                    
                # 좌 (i, j-1)
                if j > 0:
                    adj_matrix[idx][i * m + (j-1)] = 1
                    adj_matrix[i * m + (j-1)][idx] = 1
                    Left_A_hat[idx][i * m + (j-1)] = 1
                    Left_A_hat[i * m + (j-1)][idx] = 1
                    
                # 우 (i, j+1)
                if j < m - 1:
                    adj_matrix[idx][i * m + (j+1)] = 1
                    adj_matrix[i * m + (j+1)][idx] = 1
                    Right_A_hat[idx][i * m + (j+1)] = 1
                    Right_A_hat[i * m + (j+1)][idx] = 1
    
    
        # Self-loop 추가 (GCN 학습 안정화)
        np.fill_diagonal(adj_matrix, 1)
        
    
        return adj_matrix,Up_A_hat,Down_A_hat,Left_A_hat,Right_A_hat

    
    def Locate(self,grids,blocks,masks,ans):
        b,r,c,f=grids.shape #blocks b, lookahead*featuredim
        blocks=blocks.reshape(b,-1)
        
        if self.mod=='GCN1' or self.mod=='GCN2' or self.mod=='GAT' or self.mod=='GAT2':
            new_grids=grids.reshape(b,-1,f)
            blocks_expanded = blocks.unsqueeze(1).repeat(1, r*c, 1)  # (b, r*c, lookahead*featuredim)
            if ans!= None:
                output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
                merged_tensor = torch.concat([output_vector, blocks_expanded], dim=2)
                output=self.Actor_net(merged_tensor,masks) # b, r*c, 1
                pi = output.squeeze(-1).gather(1, ans)
                return pi,ans
            else:
                with torch.no_grad():
                    output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
                    merged_tensor = torch.concat([output_vector, blocks_expanded], dim=2)
                    output=self.Actor_net(merged_tensor,masks).squeeze(-1)
                    samples = torch.multinomial(output, 1) 
                    pi = output.gather(1, samples)
                    return pi,samples
        else:
            new_grids=grids.reshape(b,-1)
            if ans!=None:
                merged_tensor=torch.concat([new_grids, blocks], dim=1)
                output=self.Actor_net(merged_tensor) #b,r*c
                output=output-masks.squeeze(-1)* 1e8 #b, r*C
                output=F.softmax(output,dim=-1)
                pi = output.gather(1, ans)
                return pi, ans
            else:
                merged_tensor=torch.concat([new_grids, blocks], dim=1)
                output=self.Actor_net(merged_tensor)
                output=output-masks.squeeze(-1)* 1e8 #b*r*C
                output=F.softmax(output,dim=-1)
                samples = torch.multinomial(output, 1) 
                pi = output.gather(1, samples)
                return pi, samples
    
    def calculate_v(self, grids,blocks,block_lefts):
        b,r,c,f=grids.shape #blocks b, lookahead*featuredim
        blocks=blocks.reshape(b,-1)

        if self.mod=='GCN1' or self.mod=='GCN2' or self.mod=='GAT' or self.mod=='GAT2':
            new_grids=grids.reshape(b,-1,f)
            output_vector=self.gnn(new_grids) #b,r*c,hidden_dim
            output_vector=output_vector.reshape(b,-1)
            merged_tensor = torch.cat([output_vector, blocks], dim=1)
            merged_tensor=torch.cat([merged_tensor,block_lefts],dim=1)
            state_values=self.Critic_net(merged_tensor) #b,1
        else:
            new_grids = grids.reshape(b,-1) #b r*c*  
            merged_tensor = torch.cat([new_grids, blocks], dim=1)
            merged_tensor = torch.cat([merged_tensor, block_lefts], dim=1)
            state_values=self.Critic_net(merged_tensor) #b 1
        return state_values

    def update(self, gridss, blockss,block_leftss, actionss, rewardss,doness, maskss, probss, ep_len, step1, model_dir):
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        b,r,c,fea=gridss.shape
        #ep_len
        #gridss (b,r,c,fea)
        #blockss (b,num_overhead,fea)
        #actions (b)
        #rewards (b)
        #dones (b)
        #masks (b,r*c,1)
        #probs (b)
        gridss[:,:,:,0]=gridss[:,:,:,0]/500.0
        blockss[:,:,0]=blockss[:,:,0]/500.0
        grids=torch.tensor(gridss,dtype=torch.float32).to(device)
        block_leftss=torch.tensor(block_leftss.reshape(-1,1),dtype=torch.float32).to(device) #b,1
        blocks=torch.tensor(blockss.reshape(b,-1,fea),dtype=torch.float32).to(device)
        actions=torch.tensor(actionss.reshape(-1,1),dtype=torch.int64).to(device)
        probs = torch.tensor(probss.reshape(-1,1), dtype=torch.float32).to(device)
        log_p=torch.log(probs) #(b,1)
        rewards = torch.tensor(rewardss.reshape(-1,1), dtype=torch.float32).to(device)
        reward_sum=torch.zeros(size=(len(ep_len),1), dtype=torch.float32).to(device)
        prob_sum = torch.zeros(size=(len(ep_len), 1), dtype=torch.float32).to(device)
        pi_cal_sum = torch.zeros(size=(len(ep_len), 1), dtype=torch.float32).to(device)
        state_value_estimation=torch.zeros(size=(len(ep_len),1),dtype=torch.float32).to(device)
        masks= torch.tensor(maskss.reshape(b,r*c,1),dtype=torch.float32).to(device)
        dones = torch.tensor(doness.reshape(-1,1), dtype=torch.float32).to(device)
        block_leftss=block_leftss/100.0
        # 0 0 0 1 0 0 0 0 1 len=4
        # 0 0 1 0 0 0 0 1 
        # 0 0 0 0 1 len=5

        pi_cal,_=self.Locate(grids,blocks,masks,actions) #b,1
        pi_cal_log=torch.log(pi_cal)
        i = 0
        for e,ep in enumerate(ep_len):
            state_value_estimation[e,0]=self.calculate_v(grids[i,:,:,:].unsqueeze(0),blocks[i,:,:].unsqueeze(0),block_leftss[i,:].unsqueeze(0))
            reward_sum[e,0]=torch.sum(rewards[i:i+ep])
            prob_sum[e,0]=torch.sum(log_p[i:i+ep])
            pi_cal_sum[e,0]=torch.sum(pi_cal_log[i:i+ep])
            i+=ep
        eps = 1e-8 

        ratio = torch.exp(pi_cal_sum -prob_sum)  # a/b == exp(log(a)-log(b))
        adv=reward_sum-state_value_estimation
        surr1 = ratio * adv.detach()
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv.detach()
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_value_estimation, reward_sum.detach()) * self.alpha

        ave_loss = loss.mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1 % 20 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss






