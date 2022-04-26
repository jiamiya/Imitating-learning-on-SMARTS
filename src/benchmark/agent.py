import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import sys
import math
import numpy as np
import random
import os

states_size =33
eps = np.finfo(np.float32).eps.item()
max_timesteps = 1000
gamma = 0.99
a1_ratio =  3.0
a2_ratio = 0.3
var_high = 0.5
ratio = 1.0

class pool():
    def __init__(self) -> None:
        # self.state_mean = np.array([ 1.54376446e+02,  1.27779893e+01,  4.93921946e+00,  2.03807427e+00,
        #     2.19235545e+00,  1.46564455e+00,  1.81278382e-02,  7.51103999e+00,
        #      -2.34892890e-02 , 7.50764744e+00 , 2.59485551e-02,  1.30680339e+01,
        #       1.70145002e+01  ,1.60516051e+01 , 1.70560655e+01 , 1.31306857e+01,
        #      1.69884199e+01  ,1.60225656e+01,  1.69546131e+01 ,-5.33223712e-01,
        #      7.15538883e-01  ,8.06288259e-01 , 7.58342956e-01 ,-4.68566651e-01,
        #     -6.67499014e-01 ,-7.27817593e-01 ,-7.10314489e-01,  1.62912344e+01,
        #     1.94965635e+01  ,1.55385175e+01,  1.63964414e+01  ,1.95069278e+01,
        #      1.55207420e+01])
        params = np.load("benchmark/dim33/experience_parameters.npy",allow_pickle=True)
        self.action_mean,self.action_std,self.state_mean,self.state_std = params
        print(self.action_mean,self.action_std,self.state_mean,self.state_std)
        

class Road():
    def __init__(self) -> None:
        # self.straight_road_center = [3.88, 8.25, 11.62, 15.38, 19.08, 22.77] 
        self.straight_road_center = [3.5, 8.42, 12.11, 15.80, 19.49, 23.18] 
        self.straight_road_width = 3.658
        self.curve = [(0,-1.98),(180.00,2.57)]

    def get_lane_id(self,x,y):
        # print(x,y)
        id = round((y-1.0)/self.straight_road_width)-1
        id = max(0,id)
        id = min(id,len(self.straight_road_center)-1)
        return int(id)

    def get_info(self,x,y):
        # info : id, left_offset, right_offset
        id = self.get_lane_id(x,y)
        left_offset = y-(self.straight_road_center[id]-self.straight_road_width/2)
        right_offset = (self.straight_road_center[id]+self.straight_road_width/2)-y
        return id,left_offset,right_offset

    def get_road_heading(self,x,y):
        if x<180 and y<2.5:
            heading = math.atan2(4.55,180)
        else:
            heading = 0
        return heading

    def get_self_heading(self,x,y):
        road_heading = self.get_road_heading(x,y)
        if x<180 and y<2.5:
            center_offset = y - (4.55/180*x-1.98)
        elif x>=180 and y<5:
            center_offset = y - 3.2
        else:
            lane_id = self.get_lane_id(x,y)
            center_offset = y - self.straight_road_center[lane_id]
        heading = road_heading - 0.05*(center_offset)
        return heading

road = Road()

def get_nearest_neibor(neighbors_list,x,y):
    if len(neighbors_list)==0:
        return None
    nearest_id = 0
    nearest_dist = 99999999
    for i in range(len(neighbors_list)):
        neibor = neighbors_list[i]
        tmp_dist = (neibor.position[0]-x)**2+(neibor.position[1]-y)**2
        if tmp_dist<nearest_dist:
            nearest_dist = tmp_dist
            nearest_id = i
    return neighbors_list.pop(nearest_id)


def obs2vec(obs):
    obs_vec = []
    normalize_count = 1.0

    # ego state
    # print(obs.ego_vehicle_state.position[0],obs.ego_vehicle_state.position[1])
    x,y = obs.ego_vehicle_state.position[0],obs.ego_vehicle_state.position[1]
    lane_id,left_space,right_space = road.get_info(x,y)
    obs_vec.append(obs.ego_vehicle_state.position[0]/normalize_count)
    obs_vec.append(obs.ego_vehicle_state.position[1]/normalize_count)
    obs_vec.append(obs.ego_vehicle_state.bounding_box.length/normalize_count)
    obs_vec.append(obs.ego_vehicle_state.bounding_box.width/normalize_count)
    obs_vec.append(left_space)
    obs_vec.append(right_space)
    obs_vec.append( obs.ego_vehicle_state.heading+math.pi/2)
    obs_vec.append(obs.ego_vehicle_state.speed/normalize_count)
    lane_oritation = road.get_self_heading(x,y)
    obs_vec.append(lane_oritation)
    # if not(obs.ego_vehicle_state.steering is None):
    #     obs_vec.append(obs.ego_vehicle_state.steering)
       
    # else:
    #     obs_vec.append(math.nan)
    # 
    # if obs.ego_vehicle_state.yaw_rate is not None:
    #     obs_vec.append(obs.ego_vehicle_state.yaw_rate)
    # else:
    #     obs_vec.append(0)
    obs_vec.append(obs.ego_vehicle_state.linear_velocity[0]/normalize_count)
    obs_vec.append(obs.ego_vehicle_state.linear_velocity[1]/normalize_count)

    # neighbor state
    neighbor_list = obs.neighborhood_vehicle_states.copy()
    lidar_view = np.ones(8)*30.0
    lidar_view_speed = np.zeros(8)
    lane_front_space = np.ones(3)*30
    lane_back_space = np.ones(3)*30
    # print("-"*5)
    for n in range(10):
        if len(neighbor_list)==0:
            break
        else:
            neibor = get_nearest_neibor(neighbor_list,obs.ego_vehicle_state.position[0],obs.ego_vehicle_state.position[1])
            neibor_lane,n_l,n_r = road.get_info(neibor.position[0],neibor.position[1])
            neibor_dx = neibor.position[0]-obs.ego_vehicle_state.position[0]
            neibor_dy = neibor.position[1]-obs.ego_vehicle_state.position[1]
            # print(neibor_dx,neibor_dy,neibor_lane,lane_id)
            neibor_l = neibor.bounding_box.length
            neibor_w = neibor.bounding_box.width

            if neibor_lane == lane_id:
                if neibor_dx>0 and lane_front_space[1]>neibor_dx:
                    lane_front_space[1] = neibor_dx
                if neibor_dx<0 and lane_back_space[1]>-neibor_dx:
                    lane_back_space[1] = -neibor_dx
            
            if neibor_lane == lane_id-1:
                if neibor_dx>0 and lane_front_space[0]>neibor_dx:
                    lane_front_space[0] = neibor_dx
                if neibor_dx<0 and lane_back_space[0]>-neibor_dx:
                    lane_back_space[0] = -neibor_dx

            if neibor_lane == lane_id+1:
                if neibor_dx>0 and lane_front_space[2]>neibor_dx:
                    lane_front_space[2] = neibor_dx
                if neibor_dx<0 and lane_back_space[2]>-neibor_dx:
                    lane_back_space[2] = -neibor_dx

            if neibor_dx<0:
                neibor_dx += 0.5*neibor_l
            else:
                neibor_dx -= 0.5*neibor_l

            if neibor_dy<0:
                neibor_dy += 0.5*neibor_w
            else:
                neibor_dy -= 0.5*neibor_w

            neibor_distance = math.sqrt(neibor_dx**2+neibor_dy**2)
            area_id = round(math.atan2(neibor_dy,neibor_dx)/math.pi*4)%8
            if neibor_distance<lidar_view[area_id]:
                lidar_view[area_id] = neibor_distance 
                lidar_view_speed[area_id] = neibor.speed-obs.ego_vehicle_state.speed

    # print("-")
    return_vec = np.array(obs_vec,dtype = np.float64)
    return_vec = np.concatenate((obs_vec,lidar_view,lidar_view_speed,lane_front_space,lane_back_space))
    return_vec[np.isnan(return_vec)] = 0
    # p
    return return_vec

class Net(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=4,using_batch_norm = False):
        super(Net, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num-1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            # if using_batch_norm:
            #     layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._net(inputs)

class RLAlgo:
    def __init__(self,discrete_action=True, tau=0.01):
        self.discrete_action = discrete_action
        self._tau = tau
        # self.policy = None
    
    # update target network
    # def soft_update(self, source, target, tau=None):
    #     if tau is None:
    #         tau = self._tau
    #     with torch.no_grad():
    #         for target_param, param in zip(target.parameters(), source.parameters()):
    #             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    # sample actions
    def action(self, state):
        with torch.no_grad():
            p_out = self.policy(state) ##
        log_prob, action = self.dist(p_out)
        return action, log_prob
    
    # calculate probability of action
    def dist(self, p_out, action=None):
        mean, var = torch.chunk(p_out, 2, dim=-1)
        # print(mean.shape,var.shape)
        mean = torch.tanh(mean)*ratio
        # mean1 = mean_[:,:1]*a1_ratio
        # mean2 = mean_[:,1:]*a2_ratio
        # mean = torch.cat((mean1,mean2),dim=1)
        # if torch.isnan(torch.sum(mean)):
        #     print(p_out)
        #     p
        var = torch.nn.functional.softplus(var)
        # var1 = var[:,:1]*a1_ratio
        # var1 = torch.clamp(var1,1e-4,var_high)
        # var2 = var[:,1:]*a2_ratio
        # var2 = torch.clamp(var2,1e-4,var_high*a2_ratio/a1_ratio)
        # var = torch.cat((var1,var2),dim=1)

        var = torch.clamp(var,1e-4,var_high)
        # if torch.isnan(torch.sum(var)):
        #     print("*")
        #     p
        m = Normal(mean, var)
        if action is None:
            action = m.sample()
            # print("action shape",action.shape)
            # print(action)
            # p
            action = torch.clamp(action,-ratio,ratio)
            # action[:,1] = torch.clamp(action[:,1],-a2_ratio,a2_ratio)
        log_prob = m.log_prob(action)
            
        return log_prob, torch.Tensor(action)
        
        return log_prob.reshape(-1, 1), torch.Tensor(action).reshape(-1, 1)

    # def compute_adv(self, batch, gamma):
    #     s = batch["state"]
    #     a = batch["action"]
    #     r = batch["reward"].reshape(-1, 1)
    #     s1 = batch["next_state"]
    #     done = batch["done"].reshape(-1, 1)
    #     old_log_prob = batch["log_prob"].reshape(-1, 1)
    #     with torch.no_grad():
    #         adv = r + gamma * (1-done) * self.value(s1) - self.value(s)

    #     return adv

class PPO(RLAlgo):

    def __init__(
        self, 
        state_space, 
        action_space, 
        learning_rate,
        hidden_size,
        clip_range=0.2,
        load_path = "",
        **kwargs,
    ):
        super(PPO, self).__init__( **kwargs)
        self._clip_range = clip_range

        input_size = state_space
        output_size = action_space*2

        print(output_size,input_size)
        self.policy = Net(hidden_size,input_size,output_size)
        policy_path = "./benchmark/dim33/policy.pth"
        if os.path.exists(policy_path):
            policy_weight = torch.load(policy_path)
            self.policy.load_state_dict(policy_weight)
            print("load policy model")

        self.target_policy = Net(hidden_size,input_size,output_size)
        self.value = Net(hidden_size,input_size,1,using_batch_norm=True)
        value_path = load_path+"value.pth"
        if os.path.exists(value_path):
            value_weight = torch.load(value_path)
            self.value.load_state_dict(value_weight)
            print("load value model")

        self.target_value = Net(hidden_size,input_size,1,using_batch_norm=True)
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(target_param.data * 0.0 + param.data * 1.0)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=learning_rate*2)
        self.logsigmoid = nn.LogSigmoid()

    def save_model(self,path = 'model/'):
        policy_weight = self.policy.state_dict()
        policy_path = path+'policy.pth'
        torch.save(policy_weight, policy_path)

        value_weight = self.target_value.state_dict()
        value_path = path+'value.pth'
        torch.save(value_weight, value_path)
        print("save_model!")
    
    def update(self, batch, gamma):
        return
        batchsize = batch["state"].shape[0]
        mini_batch = 100
        # print("*")
        # print(batch['action'])

        for i in range(batchsize//mini_batch):
            s = batch["state"][i*mini_batch:(i+1)*mini_batch]
            a = batch['action'][i*mini_batch:(i+1)*mini_batch]#.view(-1, 1)
            r = batch["reward"].reshape(-1, 1)[i*mini_batch:(i+1)*mini_batch]
            s1 = batch["next_state"][i*mini_batch:(i+1)*mini_batch]
            adv = batch["adv"][i*mini_batch:(i+1)*mini_batch].detach()
            done = batch["done"].reshape(-1, 1)[i*mini_batch:(i+1)*mini_batch]
            old_log_prob = batch["log_prob"][i*mini_batch:(i+1)*mini_batch].detach()#.reshape(-1, 1)
            
            td_target = r + gamma * self.target_value(s1) * (1 - done) # 时序差分目标
            td_delta = adv#td_target - self.value(s) # 时序差分误差

            
            # print(probs_ratio)
            if self.discrete_action:
                policy_out = torch.softmax(self.policy(s),dim=1)
                log_probs = torch.log( policy_out.gather(1, a))##
            else:
                p_out = self.policy(s)
                mean_, var = torch.chunk(p_out, 2, dim=-1)
                # print(mean,var,a)
                
                # print("in forward")
                # print(mean.shape,var.shape,a.shape)
                mean = torch.tanh(mean_)*ratio
                # mean = torch.zeros_like(mean_)
                # mean1 = mean_[:,:1]*a1_ratio
                # mean2 = mean_[:,1:]*a2_ratio
                # mean = torch.cat((mean1,mean2),dim=1)
                
                var = torch.nn.functional.softplus(var)
                # var1 = var[:,:1]*a1_ratio
                # var1 = torch.clamp(var1,1e-4,var_high)
                # var2 = var[:,1:]*a2_ratio
                # var2 = torch.clamp(var2,1e-4,var_high*a2_ratio/a1_ratio)
                # var = torch.cat((var1,var2),dim=1)
                var = torch.clamp(var,1e-4,var_high)
                m = Normal(mean, var)
                log_probs = m.log_prob(a)#.reshape(-1, 1)
                # log_probs = 1.0/np.sqrt(2*np.pi)/var * torch.exp(-(a-mean)*(a-mean)/2/(var*var))
            # print(log_probs.shape,old_log_prob.shape)
            log_probs =torch.sum(log_probs,dim=1)
            old_log_prob =torch.sum(old_log_prob,dim=1)
            probs_ratio = log_probs.exp()/(old_log_prob.exp()+1e-8)
            # probs_ratio = torch.sum(probs_ratio,dim=1)
            # print(log_probs.exp())
            # print(old_log_prob.exp())
            # print(probs_ratio)
            # p
            

            loss1 = probs_ratio*adv
            loss2 = torch.clamp(probs_ratio,1-self._clip_range,1+self._clip_range)*adv
            actor_loss = -torch.mean(torch.min(loss1,loss2))
            # print(actor_loss)
            # actor_loss = torch.mean( -log_probs * td_delta.detach())
            critic_loss = torch.mean(F.mse_loss( td_target.detach(),self.value(s))) # 均方误差损失函数
            # print(td_target.shape,log_probs.shape,(-log_probs * td_delta.detach()).shape,(F.mse_loss(self.value(states), td_target.detach())),self.value(states).shape,self.policy(states).shape)
            # p
            self.opt_policy.zero_grad()
            self.opt_value.zero_grad()
            actor_loss.backward() # 计算策略网络的梯度
            critic_loss.backward() # 计算价值网络的梯度
            self.opt_policy.step() # 更新策略网络参数
            self.opt_value.step() # 更新价值网络参数

            if torch.isnan(torch.sum(self.policy(s))):
                print(log_probs)
                print(probs_ratio)
                print(mean)
                print(var)
                print(actor_loss)
                p

        
        self.soft_update(self.value, self.target_value, self._tau)

device = 'cpu'
experience_pool = pool()
hidden_size= 128
learning_rate= 1e-5
agent = PPO(
        hidden_size=hidden_size, 
        state_space=states_size, 
        action_space=2, 
        learning_rate=learning_rate,
        discrete_action=False,
    )


def get_action(obs):
    state = obs["obs"]
    ego_x,ego_y = state.ego_vehicle_state.position[0],state.ego_vehicle_state.position[1]
    # print(state)
    state = obs2vec(state)
    # print(state)
    state = torch.tensor([state],dtype=torch.float32)
    state_ = ((state-experience_pool.state_mean) / experience_pool.state_std ).type(torch.float32)
    
    action, log_prob = agent.action(state_)
    
    action = (action * experience_pool.action_std + experience_pool.action_mean).type(torch.float32)
    front_dis = state[0,-5]
    back_dis = state[0,-2]
    if front_dis<15 and front_dis<=back_dis:
        action[0,0] = (front_dis-15)*0.5
    elif back_dis<15 and front_dis>=back_dis:
        action[0,0] = -(back_dis-15)*0.5
    # else:
    #     action[0,0] = 0
    if state[0,1]<1 and action[0,0]<0:
        action[0,0] = 1
    # desired_heading = road.get_self_heading(ego_x,ego_y)
    # heading = desired_heading-state[0,0]
    # action[0,1] = state[0,8]-state[0,6]
    action = action.cpu().numpy().reshape(-1)
    return action