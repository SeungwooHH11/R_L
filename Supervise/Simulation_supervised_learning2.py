import numpy as np
from Modified_Retrieval2 import *
from Location_supervised import *
import vessl

import random

def generate_random_integers(n, m):
    return [random.randint(0, n) for _ in range(m)]
class Stockyard_simulation:
    def __init__(self,yard_size,initial_block,lam,weight,TP_type,Block_per_Day,mod):
        self.yard_size=yard_size
        self.initial_block=initial_block
        self.lam=lam
        self.weight_distribution=weight
        self.TP_type=TP_type
        self.mod=mod
        self.Block_per_Day=Block_per_Day


    def one_hot_encode(self,weight, thresholds):
        return [1 if weight < t else 0 for t in thresholds]
    
    def Generate_grid(self,seed):
        if seed is not None:
            np.random.seed(seed)  # 재현성을 위해 시드 설정
        
        grid = np.zeros((self.yard_size[0], self.yard_size[1],len(self.TP_type)+2), dtype=int)  # n*n*3 반출시간, 무게, step(블록이 배정된 스텝) 
        
        grid_save = np.zeros((self.yard_size[0], self.yard_size[1],2), dtype=int)  # n*n*3 반출시간, 무게, step(블록이 배정된 스텝)
        # 블록의 무작위 위치 선택 (중복 없이 m개 선택)
        positions = np.random.choice(self.yard_size[0] * self.yard_size[1], self.initial_block, replace=False)
        blocks=np.zeros((self.initial_block,2))
        # 반출일 및 무게 할당
        for e,pos in enumerate(positions):
            x, y = divmod(pos, self.yard_size[0])
            time=np.random.exponential(scale=1/self.lam)
            weight=np.random.randint(100, 501)
            embedded_weight=self.one_hot_encode(weight,self.TP_type)
            grid_save[x, y, 0] = time  # 반출일 하루를 100으로 lam=1/250
            grid_save[x, y, 1] = weight  # 무게 (100~500)
            grid[x, y, 0] = time  # 반출일 하루를 100으로 lam=1/250
            grid[x, y, 1:-1] = embedded_weight  # 무게 (100~500)
            blocks[e,0]=time
            blocks[e,1]=weight
        
        return grid,grid_save,blocks
        
    def Create_blocks(self):
        
        block_num=np.random.randint(self.Block_per_Day[0],self.Block_per_Day[1])
        blocks=np.zeros((block_num,2))
        
        for i in range(block_num):
            blocks[i, 0] = np.random.exponential(scale=1/self.lam)  # 반출일 하루를 100으로 lam=1/250
            blocks[i, 1] = np.random.randint(100, 501)  # 무게 (100~500)
        return blocks

    
    def Create_mask(self,grid,TP_capa):
        r,c,f=grid.shape
        mask = (grid[:, :, 1:1+len(self.TP_type)].sum(axis=2) > 0).astype(np.uint8)  # 첫 번째 열이 0 초과인 위치를 1로 설정
        mask = mask[:, :, np.newaxis].copy() 
        mask=mask.reshape(r,c)
        rows, cols = len(mask), len(mask[0])
        visited = [[False] * cols for _ in range(rows)]  # 방문 여부 기록
        new_grid = [[1] * cols for _ in range(rows)]  # 모든 값을 1로 초기화
    
        # BFS를 위한 큐
        queue = deque()
    
        # Step 1: 첫 번째 행에서 0을 찾고 BFS 시작
        for x in range(cols):
            if mask[0][x] == 0:
                queue.append((0, x))
                visited[0][x] = True  # 방문 체크
                new_grid[0][x] = 0  # 그대로 유지
    
        # BFS 탐색 (상, 하, 좌, 우)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            y, x = queue.popleft()
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < rows and 0 <= nx < cols and not visited[ny][nx] and mask[ny][nx] == 0:
                    queue.append((ny, nx))
                    visited[ny][nx] = True
                    new_grid[ny][nx] = 0  # 유지
        new_grid=np.array(new_grid)
        need_retrieve=False
        if new_grid[0].sum()==cols:
            need_retrieve=True
            check_list=np.argwhere(mask == 0)
            check_num=np.zeros(len(check_list))

            for e,space in enumerate(check_list):
                count,ispossible=Count_retrieval(grid,TP_capa,space)
                check_num[e]=count
            min_value=check_num.min()
            index=np.argwhere(check_num==min_value).flatten()
            for i in index:
                new_grid[check_list[i,0],check_list[i,1]]=0
        return new_grid,need_retrieve
      
    
    def block_encoding(self,arr, thresholds):
        column1 = arr[:, [0]]  # 1열 (index 0) 유지
        column2 = arr[:, 1]    # 2열 (index 1)의 값만 가져옴
        one_hot_matrix = (column2[:, np.newaxis] <= thresholds).astype(int)  # 원-핫 인코딩

        return np.hstack((column1, one_hot_matrix))  # 기존 1열과 원-핫 인코딩 결과를 결합

        
    def find_indices(self,grid):
        n=len(self.TP_type)
        condition_1 = grid[:, :, 0] <= 100  # 1열이 100 미만
        condition_2 = np.any(grid[:, :, 1:n+1] == 1, axis=2)  # 2열부터 n열까지 중 하나라도 1이면 True
        indices = np.where(condition_1 & condition_2)
        
        return indices
        
    def Run_simulation(self,simulation_day,lookahead_num,ppo,grid,total_block,total_block_encoded):
        current_time=0
        total_rearrangement=0
        #grid,grid_save,init_blocks=self.Generate_grid(10)
        
        grids=[]
        blocks=[]
        actions=[]
        
        dones=[]
        masks=[]
        probs=[]
        block_lefts=[]
        #gird에 step을 저장
        #gridss (n,e,r,c,fea+1) fea=2+TP_type
        #blockss (n,e,num_overhead,fea)
        #actions (n,e,1)
        #rewards (n,e,1)
        #dones (n,e,1)
        #masks (n,e,r*c,1)
        #probs (n,e,1)
        '''
        grid,grid_save,init_blocks=self.Generate_grid(10)
        total_block=[]
        for i in range(1,simulation_day+1):
            Created_blocks=self.Create_blocks() # num,2
            total_block.append(Created_blocks)
        for e,block_by_day in enumerate(total_block):
            if e==0:
                block_concat=block_by_day
            else:
                block_concat=np.concatenate((block_concat, block_by_day), axis=0)
        total_block_encoded=self.block_encoding(block_concat,self.TP_type)
        '''
        
        max_length=len(total_block_encoded)
        rewards=[0]
        step=0
        block_num=0
        #print('Simulation start')
        block_left_num=max_length
        for i in range(simulation_day):
            block_located = total_block_encoded[block_num:block_num+len(total_block[i])]

            #print(len(block_located), 'located')
            #total_block = total_block[total_block[:, 0] > 100*(i+1)]
            for e,row in enumerate(block_located):

                cc=np.where(grid[:,:,1:1+len(self.TP_type)].sum(axis=2)>0)
                if np.array(cc).shape[1]==grid.shape[0]*grid.shape[1]:
                    continue
                grids.append(grid.copy())
                block_lefts.append(block_left_num)
                block_left_num -= 1
                blocks_vec=total_block_encoded[block_num+e:int(min(block_num+e+lookahead_num,max_length)),:].copy()
                if len(blocks_vec)<lookahead_num:
                    blocks_vec_temp=np.zeros((lookahead_num,1+len(self.TP_type)))
                    blocks_vec_temp[:,0]=250
                    blocks_vec_temp[:,1:int(1+(1+len(self.TP_type))/2.0)]=1
                    blocks_vec_temp[:len(blocks_vec)]=blocks_vec
                    blocks_vec=blocks_vec_temp
                
                blocks.append(blocks_vec.copy())
                mask,need_retrieval=self.Create_mask(grid.copy(),TP_capa=len(self.TP_type)-1)
                masks.append(mask.reshape(-1,1).copy())
                
                grid_tensor=torch.tensor(grid[:,:,:-1].reshape(1,grid.shape[0],grid.shape[1],-1),dtype=torch.float32).to(device)
                grid_tensor[:,:,0]=grid_tensor[:,:,0]/(500.0)
                block_tensor=torch.tensor(blocks_vec.reshape(1,lookahead_num,-1),dtype=torch.float32).to(device)
                block_tensor[:,:,0]=block_tensor[:,:,0]/(500.0)
                mask_tensor=torch.tensor(mask.reshape(1,-1,1),dtype=torch.float32).to(device)
                
                prob,coord=ppo.Locate(grid_tensor,block_tensor,mask_tensor,ans=None)
                
                probs.append(prob.item())
                actions.append(coord.item())
                dones.append(0)
                rewards.append(0)
                r=coord.item()//grid.shape[0]
                c=coord.item()%grid.shape[0]
                
                target_block=[r,c]
                
                step+=1
                #적치
                grid[r,c,0]=total_block_encoded[block_num+e,0]
                grid[r,c,1:-1]=total_block_encoded[block_num+e,1:]
                grid[r,c,-1]=step
                if need_retrieval:
                    ispossible,rearrange_num,end_grid,step,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=Retrieval(grid.copy(),len(self.TP_type)-1,target_block.copy(),ppo,step,grids,blocks,block_lefts,block_left_num,actions,rewards,dones,masks,probs,lookahead_num,len(self.TP_type),'Not OR')
                    total_rearrangement+=rearrange_num
                    grid=end_grid.copy()
            
            indices = self.find_indices(grid)
            #print(len(indices[0]), 'retrieved')
            while True:
                indices = self.find_indices(grid)
                if len(indices[0]) == 0:
                    break
                values = grid[indices[0], indices[1], 0]
                target_index = np.argmin(values)
                target_r=indices[0][target_index]
                target_c=indices[1][target_index]
                
                target_block=[target_r,target_c]
                TP_type_len=len(self.TP_type)
                
                TP_capacity=np.random.randint(TP_type_len-grid[target_r,target_c,1:-1].sum(),TP_type_len)
                #TP_capacity=len(self.TP_type)-1
                ispossible,rearrange_num,end_grid,step,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=Retrieval(grid,TP_capacity,target_block,ppo,step,grids,blocks,block_lefts,block_left_num,actions,rewards,dones,masks,probs,lookahead_num,TP_type_len,'OR')
                while ispossible==False:
                    TP_capacity=np.random.randint(TP_type_len-grid[target_r,target_c,1:-1].sum(),TP_type_len)
                    ispossible,rearrange_num,end_grid,step,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=Retrieval(grid,TP_capacity,target_block,ppo,step,grids,blocks,block_lefts,block_left_num,actions,rewards,dones,masks,probs,lookahead_num,TP_type_len,'OR')

                total_rearrangement+=rearrange_num
                grid=end_grid.copy()

            block_num += len(total_block[i])
            grid[:, :, 0] -= 100
            grid[:, :, 0] = np.maximum(grid[:, :, 0], 0)
            
        dones[-1]=1
        return total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts



    def Train(self,train_data_num,update_num,train_step_num,eval_step,pr_num,batch_num,K,simulation_day,lookahead_num,ppo,model_dir,history_dir):
        eval_set=[]
        
        for _ in range(pr_num):
            grid,grid_save,init_blocks=self.Generate_grid(None)
            total_block=[]
            for i in range(1,simulation_day+1):
                Created_blocks=self.Create_blocks() # num,2
                total_block.append(Created_blocks)
            for e,block_by_day in enumerate(total_block):
                if e==0:
                    block_concat=block_by_day
                else:
                    block_concat=np.concatenate((block_concat, block_by_day), axis=0)
            total_block_encoded=self.block_encoding(block_concat,self.TP_type)
            eval_set.append([grid.copy(),total_block.copy(),total_block_encoded.copy()])

        ave_rearrangement=0
        '''
        for ev_set in eval_set:
            for _____ in range(5):
                total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=self.Run_simulation(simulation_day,lookahead_num,ASR_1,ev_set[0].copy(),ev_set[1].copy(),ev_set[2].copy())
                ave_rearrangement+=total_rearrangement
        print('ASR ',ave_rearrangement/pr_num/5)
        
        ave_rearrangement=0
        for ev_set in eval_set:
            for _____ in range(batch_num):
                total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=self.Run_simulation(simulation_day,lookahead_num,Random_1,ev_set[0].copy(),ev_set[1].copy(),ev_set[2].copy())
                ave_rearrangement+=total_rearrangement
        print('Random ',ave_rearrangement/pr_num/batch_num)
        ave_rearrangement=0
        for ev_set in eval_set:
            for _____ in range(batch_num):
                total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=self.Run_simulation(simulation_day,lookahead_num,BLF_1,ev_set[0].copy(),ev_set[1].copy(),ev_set[2].copy())
                ave_rearrangement+=total_rearrangement
        print('BLF',ave_rearrangement/pr_num/batch_num)
        
        gridss = []
        blockss = []
        actionss = []
        maskss = []

        for tr_step in range(train_data_num):
            if tr_step%10==0:
                print(tr_step)

            grid, grid_save, init_blocks = self.Generate_grid(None)
            total_block = []

            for i in range(1, simulation_day + 1):
                Created_blocks = self.Create_blocks()  # num,2
                total_block.append(Created_blocks)
            for e, block_by_day in enumerate(total_block):
                if e == 0:
                    block_concat = block_by_day
                else:
                    block_concat = np.concatenate((block_concat, block_by_day), axis=0)
            total_block_encoded = self.block_encoding(block_concat, self.TP_type)
            total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=self.Run_simulation(simulation_day,lookahead_num,ASR_1,grid.copy(),total_block.copy(),total_block_encoded.copy())
            gridss.append(grids.copy())
            blockss.append(blocks.copy())
            actionss.append(actions.copy())
            maskss.append(masks.copy())
        print('train_data_constructed')

        gridss = np.concatenate(gridss, axis=0)

        blockss = np.concatenate(blockss, axis=0)
        actionss = np.concatenate(actionss, axis=0)

        maskss = np.concatenate(maskss, axis=0)
        np.save('gridss7.npy',gridss)
        np.save('blockss7.npy', blockss)
        np.save('actionss7.npy', actionss)
        np.save('maskss7.npy', maskss)
        '''
        gridss=np.load('/input/gridss7.npy')
        blockss = np.load('/input/blockss7.npy')
        actionss = np.load('/input/actionss7.npy')
        maskss = np.load('/input/maskss7.npy')

        total_len=len(gridss)

        for step in range(train_step_num):
            ave_loss=0
            indices = [random.randint(0, total_len-1) for _ in range(update_num)]
            train_grids=gridss[indices].copy()
            train_blocks = blockss[indices].copy()
            train_actions = actionss[indices].copy()
            train_masks = maskss[indices].copy()

            for ____ in range(K):
                loss=ppo.Actor_net_update(train_grids[:,:,:,:-1],train_blocks,train_actions,train_masks)
                ave_loss+=loss

            if step%eval_step==0:
                ave_rearrangement=0
                for ev_set in eval_set:
                    for _____ in range(batch_num):
                        total_rearrangement,grids,blocks,actions,rewards,dones,masks,probs,block_lefts=self.Run_simulation(simulation_day,lookahead_num,ppo,ev_set[0].copy(),ev_set[1].copy(),ev_set[2].copy())
                        ave_rearrangement+=total_rearrangement
                vessl.log(step=step, payload={'eval_rearrangement': ave_rearrangement/pr_num/batch_num})

                torch.save({
                    'model_state_dict': ppo.state_dict(),
                    'optimizer_state_dict': ppo.optimizer.state_dict(),

                }, model_dir+'trained_model' + str(step) + '.pth')

if __name__=="__main__":
    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='/output/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
      
    device='cuda'
    pr_size=(7,7)
    bpd=(10,14)
    init_block=10
    ST_sim=Stockyard_simulation(yard_size=pr_size,initial_block=init_block,lam=1/250,weight=(1,501),TP_type=[300,400,550],Block_per_Day=bpd,mod=0)

    ppo=PPO(feature_dim=4, hidden_dim=32, lookahead_block_num=1,grid_size=pr_size, learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, epsilon=0.2, mod='GAT').to(device)
    #ST_sim.Train(train_step=1000,eval_step=1,K=500,pr_num=500,batch_num=1,simulation_day=10,lookahead_num=1,ppo=ppo,model_dir='',ASR_1=ASR_1,Random_1=Random_1,BLF_1=BLF_1)
    ST_sim.Train(train_data_num=1000, update_num=10000, train_step_num=500, eval_step=10, pr_num=20, batch_num=50, K=5, simulation_day=10, lookahead_num=1,ppo=ppo,model_dir=model_dir,history_dir=history_dir)
