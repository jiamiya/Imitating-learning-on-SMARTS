import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import sys
import math
import numpy as np
import random
import os

states_size =11
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
        params = np.load("dim11_experience_parameters.npy",allow_pickle=True)
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
    # global distance_count
    # global total_count
    # if obs.distance_travelled>5:
    #     print(obs.distance_travelled)
    #     print(distance_count, total_count)
    #     distance_count+=1
    # total_count+=1
    # time.sleep(0.2)
    obs_vec = []
    normalize_count = 1.0
    # events
    # collide = len(obs.events.collisions)
    # obs_vec.append(collide)
    # obs_vec.append(obs.events.off_road)
    # obs_vec.append(obs.events.off_route)
    # obs_vec.append(obs.events.on_shoulder)
    # obs_vec.append(obs.events.wrong_way)
    # obs_vec.append(obs.events.not_moving)
    # obs_vec.append(obs.events.reached_goal)
    # obs_vec.append(obs.events.reached_max_episode_steps)
    # obs_vec.append(obs.events.agents_alive_done)

    # ego state
    # print(obs.ego_vehicle_state.position[0],obs.ego_vehicle_state.position[1])
    x,y = obs.ego_vehicle_state.position[0],obs.ego_vehicle_state.position[1]
    lane_id,left_space,right_space = road.get_info(x,y)
    # obs_vec.append(obs.ego_vehicle_state.position[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.position[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.bounding_box.length/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.bounding_box.width/normalize_count)
    # obs_vec.append(left_space)
    # obs_vec.append(right_space)
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
    # obs_vec.append(obs.ego_vehicle_state.linear_velocity[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.linear_velocity[1]/normalize_count)

    # obs_vec.append(obs.ego_vehicle_state.angular_velocity[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_velocity[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_velocity[2]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.linear_acceleration[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.linear_acceleration[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_acceleration[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_acceleration[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_acceleration[2]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.linear_jerk[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.linear_jerk[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_jerk[0]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_jerk[1]/normalize_count)
    # obs_vec.append(obs.ego_vehicle_state.angular_jerk[2]/normalize_count)

    # neighbor state
    neighbor_list = obs.neighborhood_vehicle_states.copy()
    # lidar_view = np.ones(8)*30.0
    # lidar_view_speed = np.zeros(8)
    front_speed = np.zeros(1)
    back_speed = np.zeros(1)
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

            if neibor_dx<0:
                neibor_dx += 0.5*neibor_l
            else:
                neibor_dx -= 0.5*neibor_l

            if neibor_dy<0:
                neibor_dy += 0.5*neibor_w
            else:
                neibor_dy -= 0.5*neibor_w

            if neibor_lane == lane_id:
                if neibor_dx>0 and lane_front_space[1]>neibor_dx:
                    lane_front_space[1] = neibor_dx
                    front_speed[0] = neibor.speed-obs.ego_vehicle_state.speed
                if neibor_dx<0 and lane_back_space[1]>-neibor_dx:
                    lane_back_space[1] = -neibor_dx
                    back_speed[0] = neibor.speed-obs.ego_vehicle_state.speed
            
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

            

            neibor_distance = math.sqrt(neibor_dx**2+neibor_dy**2)
            area_id = round(math.atan2(neibor_dy,neibor_dx)/math.pi*4)%8
            # if neibor_distance<lidar_view[area_id]:
            #     lidar_view[area_id] = neibor_distance 
            #     lidar_view_speed[area_id] = neibor.speed-obs.ego_vehicle_state.speed
            # obs_vec.append((neibor.position[0]-obs.ego_vehicle_state.position[0]))
            # obs_vec.append((neibor.position[1]-obs.ego_vehicle_state.position[1]))
            # obs_vec.append(neibor.bounding_box.length)
            # obs_vec.append(neibor.bounding_box.width)
            # obs_vec.append((neibor.heading))
            # if abs(obs_vec[-5]) - 0.5*obs_vec[-3]<0 and abs(obs_vec[-4]) - 0.5*obs_vec[-2]<0:
            #     print(obs_vec[-5:])

            # obs_vec.append(neibor.speed/normalize_count)
    #         print(obs_vec[-6:])
    # print("-")
    return_vec = np.array(obs_vec,dtype = np.float64)
    return_vec = np.concatenate((obs_vec,front_speed,back_speed , lane_front_space,lane_back_space))
    # print(lidar_view)
    # print("*")
    # print(return_vec[-6:])
    # print(return_vec.shape)
    return_vec[np.isnan(return_vec)] = 0
    # is_30 = return_vec==30.0
    # print(return_vec)
    # return_vec[is_30] = 0
    # print(return_vec)
    # print("*")
    # print(return_vec)
    # print(return_vec.shape)
    # p
    return return_vec

class Net(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=3,using_batch_norm = False):
        super(Net, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num-1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            if using_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
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
    
    # update target network
    def soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self._tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    # sample actions
    def action(self, state):
        # with torch.no_grad():
        p_out = self.policy(state) ##
        # print(p_out)
        log_prob, action = self.dist(p_out)
        return action, log_prob
    
    # calculate probability of action
    def dist(self, p_out, action=None):
       
        mean, var = torch.chunk(p_out, 2, dim=-1)
        # print(mean.shape,var.shape)
        # mean = torch.tanh(mean)
        # print("mean",mean)
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
        # print(var)
        # if torch.isnan(torch.sum(var)):
        #     print("*")
        #     p
        m = Normal(mean, var)
        if action is None:
            action = mean#m.sample()
            # print("action",action)
            # print("action shape",action.shape)
            # print(action)
            # p
            action = torch.clamp(action,-ratio,ratio)
            # action[:,1] = torch.clamp(action[:,1],-a2_ratio,a2_ratio)
        log_prob = m.log_prob(action)
            
        return log_prob, action
        
        return log_prob.reshape(-1, 1), torch.Tensor(action).reshape(-1, 1)

    def compute_adv(self, batch, gamma):
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"].reshape(-1, 1)
        s1 = batch["next_state"]
        done = batch["done"].reshape(-1, 1)
        old_log_prob = batch["log_prob"].reshape(-1, 1)
        with torch.no_grad():
            adv = r + gamma * (1-done) * self.value(s1) - self.value(s)

        return adv

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
        policy_path = "./benchmark/policy.pth"
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

device = 'cpu'
experience_pool = pool()
hidden_size= 64
learning_rate= 1e-4
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
    action = action.detach().numpy()
    # action denormalized
    action = (action * experience_pool.action_std + experience_pool.action_mean)#.type(torch.float32)
    if  state[0,-5]<8 and  state[0,-5]<=state[0,-2]:
        action[0,0] = ( state[0,-5]-16)*2
    elif state[0,-2]<8 and  state[0,-5]>=state[0,-2]:
        action[0,0] = -(state[0,-2]-16)*2

    # action clip: avoid driving reversely
    velocity = state[0,1]
    if velocity<1 and action[0,0]<0:
        action[0,0] = 10
    action = action.reshape(-1)
    return action