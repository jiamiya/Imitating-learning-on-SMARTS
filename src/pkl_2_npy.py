import pickle
import numpy as np
# from traffic_simulator import TrafficSim
# from net import Discriminator
from tqdm import tqdm
import time
import torch
import math

file_num = 4
# if file_num==0:
#     file_num="_last"

f = open('expert'+str(file_num)+'.pkl',"rb+")
data= pickle.load(f)
f.close()

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

# global distance_count
# distance_count = 0
# global total_count
# total_count = 0

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

def get_expert_data():
    '''
    data -> list1[list2[]]
    list2: a trajectory of expert
    list1: all experts
    '''
    # print((data['actions'][20][10]))
    # print((data['terminals'][20][10]))
    # return

    all_expert_trajectory = []
    
    for i in tqdm(range(len(data['actions']))):#5):#11)):#
        total_dis = 0
        # print(data['actions'][i].shape)
        # print(len(data['observations'][i]))
        # print(len(data['next_observations'][i]))
        # print(data['terminals'][i].shape)
        tmp_expert_trajectoy = [] # (obs, next_obs, actions, terminates)
        for j in range(len(data['observations'][i])):#5):#
            # j = 10
            # print("_"*10)
            tmp_obs = data['observations'][i][j]
            next_obs = data['next_observations'][i][j]
            action = data['actions'][i][j] # eg. [ 1.3, 0.9]
            terminal = data['terminals'][i][j] # eg. False
            total_dis+=tmp_obs.distance_travelled
            # print(terminal,total_dis)
            # if(terminal):
            #     print(total_dis)
            #     total_dis = 0

            tmp_vec = obs2vec(tmp_obs)
            next_vec = obs2vec(next_obs)
            tmp_expert_trajectoy.append((tmp_vec , next_vec , action , terminal))
            # print(tmp_vec)
            # print(tmp_vec.shape)
            # print(data['terminals'][i][j])
        all_expert_trajectory.append(tmp_expert_trajectoy)
    return all_expert_trajectory

if __name__ == "__main__":
    all_expert = get_expert_data()
    
    np.save('exp_data'+str(file_num)+'.npy',all_expert)
    exit()
    all_expert = np.load('exp_data.npy',allow_pickle=True)
    # print(all_expert[10][10])
    print(len(all_expert))
    print(len(all_expert[10]))
    
    # exit()
    env = TrafficSim(["./ngsim"])
    obs = env.reset()
    done = False
    # while(not done):

    #     act = np.random.normal(0, 1, size=(2,))
    #     obs, rew, done, info = env.step(act)
    #     tmp_vec = obs2vec(obs)
    #     print(tmp_vec[:10])
    #     print(rew)
    #     print(done)
    #     print(info)

    device = 'cpu'
    discriminator_network = Discriminator(input_size = 94).to(device)
    optimizer_discriminator = torch.optim.Adam(discriminator_network.parameters(), lr=1e-3)
    last_obs = obs

    
    for e in range(20):
        for i in range(len(all_expert)):
            tmp_expert = all_expert[i]
            for j in range(len(tmp_expert)):
                (expert_s, expert_next_s, expert_a, expert_terminal) = tmp_expert[j]
                expert_s = torch.FloatTensor(expert_s).to(device)
                expert_a = torch.FloatTensor(expert_a).to(device)
                expert_s_a = torch.cat((expert_s,expert_a)).reshape(1,-1)

                agent_act = np.random.normal(0, 1, size=(2,))
                agent_obs, agent_rew, agent_done, agent_info = env.step(agent_act)
                agent_s = torch.FloatTensor(obs2vec(last_obs)).to(device)
                agent_a = torch.FloatTensor(agent_act).to(device)
                agent_s_a = torch.cat((agent_s,agent_a)).reshape(1,-1)

                expert_score = discriminator_network(expert_s_a)
                agent_score = discriminator_network(agent_s_a)
                loss_discrime = -(torch.log(1-agent_score) + torch.log(expert_score))
                optimizer_discriminator.zero_grad()
                loss_discrime.backward(retain_graph = True)
                optimizer_discriminator.step()
                print(agent_score,expert_score)
                time.sleep(0.1)


                last_obs = agent_obs
                if expert_terminal or agent_done:
                    if agent_done:
                        last_obs = env.reset()
                    break
            

    print("finished")