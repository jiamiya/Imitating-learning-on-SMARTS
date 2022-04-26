import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import sys
import math
from traffic_simulator import TrafficSim
import numpy as np
from pkl_2_npy import obs2vec,Road
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
# import gym
torch.manual_seed(1)
road = Road()

if torch.cuda.is_available():
    device = "cpu"
else:
    device = "cpu"

states_size =11
eps = np.finfo(np.float32).eps.item()
max_timesteps = 1000
gamma = 0.98
a1_ratio =  3.0
a2_ratio = 0.3
ratio = 15.0
var_high = 0.1
reward_normalized = True
replay_buffer_size = 2000
batch_size = 2000

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

def trans2tensor(batch):
    for k in batch:
        # print(batch[k])
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device)
        elif isinstance(batch[k][0], torch.Tensor):
            batch[k] = torch.cat(batch[k]).to(device=device)
        else:
            batch[k] = torch.tensor(batch[k], device=device, dtype=torch.float32)

    return batch

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
        with torch.no_grad():
            p_out = self.policy(state) ##
        log_prob, action = self.dist(p_out)
        return action, log_prob
    
    # calculate probability of action
    def dist(self, p_out, action=None):
        
        mean, var = torch.chunk(p_out, 2, dim=-1)
        # print(mean.shape,var.shape)
        mean =  torch.clamp(mean,-ratio,ratio)#torch.tanh(mean)
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
        load_path = "model/",
        **kwargs,
    ):
        super(PPO, self).__init__( **kwargs)
        self._clip_range = clip_range

        input_size = state_space
        output_size = action_space*2

        print(output_size,input_size)
        self.policy = Net(hidden_size,input_size,output_size)
        policy_path = load_path+"policy.pth"
        if os.path.exists(policy_path):
            policy_weight = torch.load(policy_path)
            self.policy.load_state_dict(policy_weight)
            print("load policy model")

        # self.target_policy = Net(hidden_size,input_size,output_size)
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
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=learning_rate*10)
        self.logsigmoid = nn.LogSigmoid()

    def save_model(self,path = 'model/'):
        policy_weight = self.policy.state_dict()
        policy_path = path+'policy.pth'
        torch.save(policy_weight, policy_path)

        value_weight = self.target_value.state_dict()
        value_path = path+'value.pth'
        torch.save(value_weight, value_path)
        print("save_model!")
    
    def update(self, batch, gamma,update_policy = True):
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
                # mean = torch.tanh(mean_)
                mean =  torch.clamp(mean_,-ratio,ratio)
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
            if update_policy:
                actor_loss.backward() 
                self.opt_policy.step() 
            critic_loss.backward() 
            self.opt_value.step() 

            if torch.isnan(torch.sum(self.policy(s))):
                print(log_probs)
                print(probs_ratio)
                print(mean)
                print(var)
                print(actor_loss)
                p

        
        self.soft_update(self.value, self.target_value, self._tau)


class Discriminator(nn.Module):
    def __init__(self, input_size=int(states_size+2), h_size=32):
        super(Discriminator, self).__init__()
        self.input_n = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_n, h_size),
            # nn.BatchNorm1d(h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            # # nn.BatchNorm1d(h_size),
            nn.ReLU(),
            # nn.Linear(h_size, h_size),
            # # nn.BatchNorm1d(h_size),
            # nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(h_size, 1),
            # nn.Tanh()
            nn.Sigmoid()
        )
        model_path = 'model/discriminator.pth'
        if os.path.exists(model_path):
            weight = torch.load(model_path)
            self.net.load_state_dict(weight)
            print("load discriminator model")

    def forward(self, x):
        x = self.net(x)
        return x
    
    def save_model(self,path = 'model/discriminator.pth'):
        weight = self.net.state_dict()
        torch.save(weight, path)
    
# data_expert = np.load('exp_data.npy',allow_pickle=True) # np.array(data_expert)
import time
class replay_buffer():
    def __init__(self):
        self.agent_replay_buffer = []
        # self.expert_replay_buffer = []
        self.buffer_max_size = replay_buffer_size
        # self.traj_len = 200
        self.batch_size = batch_size
        # self.expert_data = np.load('exp_data0.npy',allow_pickle=True)
        # self.expert_index = [] # (i,start,end)
        self.expert_buffer = []
        self.tmp_file = 0
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.mean = None
        self.std = None
        self.buffer_counter = 0
    
    def normalize_agent_batch(self,batch):
        # batch = trans2tensor({"state": states, "action":acts, 
        #                   "log_prob": log_probs,
        #                   "next_state": next_states, "done":dones,  
        #                   "reward": rewards, })
        s = (batch["state"]-self.state_mean)/self.state_std
        s1 = (batch["next_state"]-self.state_mean)/self.state_std
        a = (batch["action"]-self.action_mean)/self.action_std
        batch["state"] = s.type(torch.float32)
        batch["next_state"] = s1.type(torch.float32)
        batch["action"] = a.type(torch.float32)
        index = [int(i) for i in range(len(s))]
        random.shuffle(index)
        # print(index)
        for k in batch.keys():
            batch[k] = batch[k][index]
            # print(batch[k].shape)
        # time.sleep(1)
        return batch
    
    def init_expert_buffer(self,):
        num = self.tmp_file
        self.expert_buffer = []
        outliner_count = 0
        for i in range(5):
            self.tmp_file = i
            self.expert_data = np.load('exp_data'+str(self.tmp_file)+'.npy',allow_pickle=True)
            for i in range(len(self.expert_data)):
                action_origin = []
                for  j in range(len(self.expert_data[i])):
                    # the environment bug
                    if abs(self.expert_data[i][j][2][1])>5:
                        self.expert_data[i][j][2][1] = 0
                        outliner_count +=1
                    action_origin.append( self.expert_data[i][j][2])
                for  j in range(len(self.expert_data[i])):
                    action = np.mean(action_origin[max(0,j-2):min(len(self.expert_data[i]),j+3)],axis = 0) #filtered
                    # print(action.shape,self.expert_data[i][j][0].shape)
                    exp_vec = np.concatenate( (self.expert_data[i][j][0],action,))
                    self.expert_buffer.append(exp_vec)
                    # print(exp_vec.shape)
                    # time.sleep(1)
                    # self.expert_buffer.append((self.expert_data[i][j][0],self.expert_data[i][j][1],action,self.expert_data[i][j][3]))

        random.shuffle(self.expert_buffer)
        expert_origin = np.array(self.expert_buffer)
        # print(expert_origin.shape)
        self.mean = expert_origin.mean(axis = 0)
        self.action_mean = self.mean[-2:]
        self.state_mean = self.mean[:-2]
        # print(self.mean)
        self.std = expert_origin.std(axis = 0)
        self.action_std = self.std[-2:]
        self.state_std = self.std[:-2]
        # print(self.std)
        # print(expert_origin[0])
        expert_normalized = (expert_origin-self.mean)/self.std
        self.expert_buffer = expert_normalized#.tolist()

        # print(len(self.expert_buffer))
        # print(self.expert_buffer[0])
        # exit()
        print("init file ",len(self.expert_buffer))
        print(self.action_mean,self.action_std)
        print(self.state_mean,outliner_count)

    def get_expert_batch(self):
        if self.buffer_counter > 400:
            self.init_expert_buffer()
            self.buffer_counter = 0
        self.buffer_counter+=1
        rand_choice = random.randint(0,len(self.expert_buffer)-self.batch_size-1)
        return_batch = self.expert_buffer[rand_choice:rand_choice+self.batch_size,:]
        # print(return_batch.shape)
        return return_batch
    
    def get_agent_batch(self):
        batchSize = self.batch_size
        states = []
        acts = []
        rewards = []
        next_states = []
        log_probs = []
        advs = []
        dones = []
        if(len(self.agent_replay_buffer)<self.batch_size):
            batchSize = len(self.agent_replay_buffer)
        if batchSize==0:
            p
        agent_data_batch = []
        # random.shuffle(self.agent_replay_buffer)
        for j in range(batchSize):
            rand_num = j#random.randint(0,len(self.agent_replay_buffer)-1)
            tmp_data = self.agent_replay_buffer[rand_num]
            # agent_data_batch.append()
            states.append(tmp_data[0])
            acts.append(tmp_data[1])
            next_states.append(tmp_data[2])
            log_probs.append(tmp_data[3])
            rewards.append(tmp_data[4])
            advs.append(tmp_data[5])
            dones.append(tmp_data[6])
            if len(self.agent_replay_buffer)<=0:
                break
        agent_data_batch = trans2tensor({"state": states, "action":acts, 
                          "log_prob": log_probs,"adv":advs,
                          "next_state": next_states, "done":dones,  
                          "reward": rewards, })
        return agent_data_batch
    
    def append_agent_data(self,agent_data):
        # agent_data: batchsize*(s+a+s1+logprob+score+adv+done)
        if agent_data.shape[1]!= 2*(states_size+3)+1:
            print(agent_data.shape)
            p
        states = agent_data[:,:states_size]
        actions = agent_data[:,states_size:states_size+2]
        next_states = agent_data[:,states_size+2:2+states_size*2]
        log_probs = agent_data[:,-5:-3]
        rewards = agent_data[:,-3]
        advs = agent_data[:,-2]
        dones = agent_data[:,-1:]
        for i in range(agent_data.shape[0]):
             self.agent_replay_buffer.append((states[i],actions[i],next_states[i],log_probs[i],rewards[i],advs[i],dones[i]))
        if len(self.agent_replay_buffer)>self.buffer_max_size:
            self.agent_replay_buffer.pop(0)





env = TrafficSim(["./ngsim"])
env.seed(1)
random.seed(1)
print("set env success")


def train(agent ,discriminator , num_episode=1000, mini_epoch=10, discrete_action=False, print_every=50):
    rewards_log = []
    episodes_log = []
    experience_pool = replay_buffer()
    experience_pool.init_expert_buffer()
    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    critireon = torch.nn.MSELoss()
    discriminator_update = True


    agent_scores = []
    exp_scores = []
    bar = tqdm(range(num_episode))
    score_info = []
    reach_goal = []
    for i_episode in bar:
        states = []
        acts = []
        rewards = []
        next_states = []
        log_probs = []
        dones = []
        
        state = env.reset()
        ego_x,ego_y = state.ego_vehicle_state.position[0],state.ego_vehicle_state.position[1]
        velocity =  state.ego_vehicle_state.speed
        state = obs2vec(state)
        state = torch.tensor([state], device=device, dtype=torch.float32)
        # print(state.shape)
        # p
        last_reward = 0
        for time_step in (range(max(2000,batch_size))):
            # print("*")
            # print(state.dtype)
            # velocity = int(state[0,5])
            state = ((state-experience_pool.state_mean) / experience_pool.state_std ).type(torch.float32)
            # state = torch.tensor(state.clone().detach(), dtype=torch.float32)
            # print(state.dtype,state)
            action, log_prob = agent.action(state)
            state = (state * experience_pool.state_std + experience_pool.state_mean).type(torch.float32)
            # print(action,"&")
            # print(experience_pool.action_std,experience_pool.action_mean)
            action = (action * experience_pool.action_std + experience_pool.action_mean).type(torch.float32)

            #action clip
            # front_dis = state[0,-5]
            # back_dis = state[0,-2]
            # if front_dis<15 and front_dis<=back_dis:
            #      action[0,0] = (front_dis-15)*1# - state[0,-14] # relative spped
            # elif back_dis<15 and front_dis>=back_dis:
            #      action[0,0] = -(back_dis-15)*1
            # # elif back_dis<front_dis and front_dis<15 and back_dis<15:
            # #     action[0,0] = 0.1*(front_dis-back_dis)
            # else:
            #     action[0,0] = 0
            # if velocity<1 and action[0,0]<0:
            #     action[0,0] = 0
            # action[0,1] = state[0,2]-state[0,0]

            # desired_heading = road.get_self_heading(ego_x,ego_y)
            # heading = desired_heading-state[0,0]
            # print(action[0,1],heading)
            action = action.cpu()
                
            next_state_, reward, done, info = env.step(action.numpy().reshape(-1))
            velocity =  next_state_.ego_vehicle_state.speed
            ego_x,ego_y = next_state_.ego_vehicle_state.position[0],next_state_.ego_vehicle_state.position[1]
            next_state = torch.tensor([obs2vec(next_state_)], device=device, dtype=torch.float32)
            # experience_pool.append_agent_data((state,next_state,action,))
            
            # collect samples
            states.append(state) # not normalized
            acts.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # state = torch.tensor([next_state], device=device, dtype=torch.float32)
            state = next_state #torch.tensor(obs2vec(next_state), device=device, dtype=torch.float32)

            if len(states)>=experience_pool.batch_size:
                break

            if done:
                travelled = info
                # print(info)
                # bar.set_postfix({"score":info})
                score_info.append(info)
                reach_goal.append(next_state_.events.reached_goal)
                state = env.reset()
                ego_x,ego_y = state.ego_vehicle_state.position[0],state.ego_vehicle_state.position[1]
                state = obs2vec(state)
                state = torch.tensor([state], device=device, dtype=torch.float32)
                # print(reach_goal[-1])
                # print(,time_step)
                continue
        
        rewards_log.append(np.sum(rewards))
        episodes_log.append(i_episode)
        # print(len(states),len(acts))
        # print(states[0].shape,acts[0].shape)
        batch = trans2tensor({"state": states, "action":acts, 
                          "log_prob": log_probs,
                          "next_state": next_states, "done":dones,  
                          "reward": rewards, })

        # print(batch["action"][0])
        batch = experience_pool.normalize_agent_batch(batch)
        # print(batch["action"][0])

        # discriminator
        expert_batch = experience_pool.get_expert_batch() 
        expert_batch =  torch.tensor(expert_batch,dtype = torch.float32).to(device) # batchsize * (s_a)
        # print(batch["state"].shape,batch["action"].shape)
        agent_batch = torch.cat((batch["state"],batch["action"]),dim=1).detach().to(device) 
        # print("agent size",agent_batch.shape)
        # print("expert_batch size",expert_batch.shape)
        mini_batch_size = 50
        agent_acc = 0
        expert_acc = 0
        for i in range(agent_batch.shape[0]//mini_batch_size):
            expert_mini_batch = expert_batch[i*mini_batch_size:(i+1)*mini_batch_size,:]
            agent_mini_batch = agent_batch[i*mini_batch_size:(i+1)*mini_batch_size,:]
            # mask for position
            # expert_mini_batch[:,:2] = 0
            # agent_mini_batch[:,:2] = 0
            expt_score = discriminator(expert_mini_batch)
            agt_score = discriminator(agent_mini_batch)
            agent_acc += torch.sum(agt_score<0.5).detach().numpy()
            expert_acc += torch.sum(expt_score>0.5).detach().numpy()
            loss_discrime = -(torch.mean(torch.log(1.0-agt_score)) + torch.mean(torch.log(expt_score)))
            if ((i%10==0 and discriminator_update) or i_episode<20):# or i_episode<50
                opt_dis.zero_grad()
                loss_discrime.backward()
                opt_dis.step()
        agent_acc /= agent_batch.shape[0]
        expert_acc /= agent_batch.shape[0]
        if agent_acc>0.9:
            discriminator_update = False
        if agent_acc<0.8:
            discriminator_update = True
       
        expert_score = discriminator(expert_batch)
        agent_score = discriminator(agent_batch)
        a_score = torch.mean(agent_score).detach().numpy()
        e_score = torch.mean(expert_score).detach().numpy()
        agent_scores.append(a_score)
        exp_scores.append(e_score)
        # print(torch.mean(expert_score),torch.mean(agent_score))
        bar.set_postfix({"agent":a_score,"exp":e_score,"dis":np.mean(score_info[max(0,len(score_info)-30):]),"a1":action,"agt_acc":agent_acc,"exp_acc":expert_acc })


        # print(torch.mean(expert_score),torch.mean(agent_score))
        # loss_discrime = torch.mean(critireon(agent_score,torch.zeros_like(agent_score)))+torch.mean(critireon(expert_score,torch.ones_like(expert_score)))
        # critireon

        if i_episode>=500  or True:
            batch["reward"] = -torch.log(1.0-agent_score).detach() 


        if reward_normalized:
            r = batch["reward"] 
            batch["reward"] = (r - r.mean()) / (r.std() + 1e-8)
        # print(torch.mean(batch["reward"]))

        
        
        # print(batch["reward"].shape)

        batch["adv"] = agent.compute_adv(batch, gamma)
        # print(batch["adv"].shape)
        # print(batch["log_prob"].shape)
        # print(batch["done"].shape)
        # print(batch["next_state"].shape)
        # print(agent_score.shape)
        # agent_batch = torch.cat((agent_batch,batch["next_state"],batch["log_prob"],batch["reward"].reshape(-1,1),batch["adv"],batch["done"].reshape(-1,1)),dim=1).detach().numpy()
        # experience_pool.append_agent_data(agent_batch)
        # large_batch = experience_pool.get_agent_batch()
        large_batch = batch
        # print(large_batch["reward"].shape,expert_batch.shape)

        # if reward_normalized:
        #     r = large_batch["reward"] 
        #     large_batch["reward"] = (r - r.mean()) / (r.std() + 1e-8)

        
        # action = large_batch['action']
        # action1 = action[:,:1] / a1_ratio
        # action2 = action[:,1:] /a2_ratio
        # action =  torch.cat((action1,action2),dim=1)
        # large_batch['action'] = action
        update_policy = i_episode>25
        for i in range(5):
            agent.update(large_batch, gamma,update_policy=update_policy)
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            print("Episode: {}, Reward: {}".format(i_episode+1, np.mean(rewards_log[-print_every:])))
            print(np.mean(np.array(agent_scores[-print_every:])),np.mean(np.array(exp_scores[-print_every:])))
            print(np.mean(score_info[-print_every:]))
            agent.save_model(path="model/prebc_epoch"+str(i_episode))
            discriminator.save_model(path = 'model/prebc_epoch'+str(i_episode)+"discriminator.pth")
            # agent_scores.clear()
            # exp_scores.clear()
    # env.close()
    infos = {
        "rewards": rewards_log,
        "episodes": episodes_log,
        "agent_scores":agent_scores,
        "exp_scores":exp_scores,
        "distance":score_info
    }
    return infos



if __name__ == "__main__":
    hidden_size= 64
    learning_rate= 1e-5
    agent = PPO(
            hidden_size=hidden_size, 
            state_space=states_size, 
            action_space=2, 
            learning_rate=learning_rate,
            discrete_action=False,
        )
    discrim_net = Discriminator()
    info = train(agent,discrim_net, 200, 10, discrete_action=False)
    # print(len(agt))
    plt.plot(info["agent_scores"],label ='agent')
    plt.plot(info["exp_scores"],label = 'expert')
    plt.legend()
    plt.show()

    plt.plot(info["distance"])
    plt.legend()
    plt.show()
    np.save("gail_distance.npy",info["distance"])