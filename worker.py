import time
import random
import os
from copy import deepcopy
from typing import Tuple
import threading
import ray
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
import numpy as np
from model import Network
from environment import Environment
from buffer import SumTree, LocalBuffer
import configs

@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, episode_capacity=configs.episode_capacity, local_buffer_capacity=configs.max_episode_length,
                init_env_settings=configs.init_env_settings, max_comm_agents=configs.max_comm_agents,
                alpha=configs.prioritized_replay_alpha, beta=configs.prioritized_replay_beta):

        self.capacity = episode_capacity
        self.local_buffer_capacity = local_buffer_capacity
        self.size = 0
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(episode_capacity*local_buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings:[]}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = np.zeros(((local_buffer_capacity+1)*episode_capacity, configs.max_num_agents, *configs.obs_shape), dtype=np.bool)
        self.act_buf = np.zeros((local_buffer_capacity*episode_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((local_buffer_capacity*episode_capacity), dtype=np.float16)
        self.hid_buf = np.zeros((local_buffer_capacity*episode_capacity, configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
        self.done_buf = np.zeros(episode_capacity, dtype=np.bool)
        self.size_buf = np.zeros(episode_capacity, dtype=np.uint)
        self.comm_mask_buf = np.zeros(((local_buffer_capacity+1)*episode_capacity, configs.max_num_agents, configs.max_num_agents), dtype=np.bool)


    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(configs.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)
        
    def get_data(self):

        if len(self.batched_data) == 0:
            print('no prepared data')
            data = self.sample_batch(configs.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: Tuple):
        '''
        data: actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5, hid_buf 6, td_errors 7, done 8, size 9, comm_mask 10
        '''
        if data[0] >= 12:
            stat_key = (data[1], data[2])

            if stat_key in self.stat_dict:

                self.stat_dict[stat_key].append(data[8])
                if len(self.stat_dict[stat_key]) == 201:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:

            idxes = np.arange(self.ptr*self.local_buffer_capacity, (self.ptr+1)*self.local_buffer_capacity)
            start_idx = self.ptr*self.local_buffer_capacity
            # update buffer size
            self.size -= self.size_buf[self.ptr].item()
            self.size += data[9]
            self.counter += data[9]

            self.priority_tree.batch_update(idxes, data[7]**self.alpha)

            self.obs_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1, :data[1]] = data[3]
            self.act_buf[start_idx:start_idx+data[9]] = data[4]
            self.rew_buf[start_idx:start_idx+data[9]] = data[5]
            self.hid_buf[start_idx:start_idx+data[9], :data[1]] = data[6]
            self.done_buf[self.ptr] = data[8]
            self.size_buf[self.ptr] = data[9]
            self.comm_mask_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1] = 0
            self.comm_mask_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1, :data[1], :data[1]] = data[10]

            self.ptr = (self.ptr+1) % self.capacity

    def sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_comm_mask = [], [], [], [], [], [], []
        idxes, priorities = [], []
        b_hidden = []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.local_buffer_capacity
            local_idxes = idxes % self.local_buffer_capacity

            for idx, global_idx, local_idx in zip(idxes.tolist(), global_idxes.tolist(), local_idxes.tolist()):
                
                assert local_idx < self.size_buf[global_idx], 'index is {} but size is {}'.format(local_idx, self.size_buf[global_idx])

                steps = min(configs.forward_steps, (self.size_buf[global_idx].item()-local_idx))
                seq_len = min(local_idx+1, configs.seq_len)

                if local_idx < configs.seq_len-1:
                    obs = self.obs_buf[global_idx*(self.local_buffer_capacity+1):idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[global_idx*(self.local_buffer_capacity+1):idx+global_idx+1+steps]
                    hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                elif local_idx == configs.seq_len-1:
                    obs = self.obs_buf[idx+global_idx+1-configs.seq_len:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[global_idx*(self.local_buffer_capacity+1):idx+global_idx+1+steps]
                    hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                else:
                    obs = self.obs_buf[idx+global_idx+1-configs.seq_len:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[idx+global_idx+1-configs.seq_len:idx+global_idx+1+steps]
                    hidden = self.hid_buf[idx-configs.seq_len]

                if obs.shape[0] < configs.seq_len+configs.forward_steps:
                    pad_len = configs.seq_len+configs.forward_steps-obs.shape[0]
                    obs = np.pad(obs, ((0,pad_len),(0,0),(0,0),(0,0),(0,0)))
                    comm_mask = np.pad(comm_mask, ((0,pad_len),(0,0),(0,0)))

                action = self.act_buf[idx]
                reward = 0
                for i in range(steps):
                    reward += self.rew_buf[idx+i]*0.99**i

                if self.done_buf[global_idx] and local_idx >= self.size_buf[global_idx]-configs.forward_steps:
                    done = True
                else:
                    done = False
                
                b_obs.append(obs)
                b_action.append(action)
                b_reward.append(reward)
                b_done.append(done)
                b_steps.append(steps)
                b_seq_len.append(seq_len)
                b_hidden.append(hidden)
                b_comm_mask.append(comm_mask)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float16)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.HalfTensor(b_reward).unsqueeze(1),
                torch.HalfTensor(b_done).unsqueeze(1),
                torch.HalfTensor(b_steps).unsqueeze(1),
                torch.LongTensor(b_seq_len),
                torch.from_numpy(np.concatenate(b_hidden)),
                torch.from_numpy(np.stack(b_comm_mask)),

                idxes,
                torch.from_numpy(weights).unsqueeze(1),
                self.ptr
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr*self.local_buffer_capacity) | (idxes >= self.ptr*self.local_buffer_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.local_buffer_capacity) & (idxes >= self.ptr*self.local_buffer_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)

    def stats(self, interval: int):
        print('buffer update speed: {}/s'.format(self.counter/interval))
        print('buffer size: {}'.format(self.size))

        print('  ', end='')
        for i in range(configs.init_env_settings[1], configs.max_map_lenght+1, 5):
            print('   {:2d}   '.format(i), end='')
        print()

        for num_agents in range(configs.init_env_settings[0], configs.max_num_agents+1):
            print('{:2d}'.format(num_agents), end='')
            for map_len in range(configs.init_env_settings[1], configs.max_map_lenght+1, 5):
                if (num_agents, map_len) in self.stat_dict:
                    print('{:4d}/{:<3d}'.format(sum(self.stat_dict[(num_agents, map_len)]), len(self.stat_dict[(num_agents, map_len)])), end='')
                else:
                    print('   N/A  ', end='')
            print()

        for key, val in self.stat_dict.copy().items():
            # print('{}: {}/{}'.format(key, sum(val), len(val)))
            if len(val) == 200 and sum(val) >= 200*configs.pass_rate:
                # add number of agents
                add_agent_key = (key[0]+1, key[1]) 
                if add_agent_key[0] <= configs.max_num_agents and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []
                
                if key[1] < configs.max_map_lenght:
                    add_map_key = (key[0], key[1]+5) 
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []
                
        self.env_settings_set = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

    def ready(self):
        if len(self) >= configs.learning_starts:
            return True
        else:
            return False
    
    def get_env_settings(self):
        return self.env_settings_set

    def check_done(self):

        for i in range(configs.max_num_agents):
            if (i+1, configs.max_map_lenght) not in self.stat_dict:
                return False
        
            l = self.stat_dict[(i+1, configs.max_map_lenght)]
            
            if len(l) < 200:
                return False
            elif sum(l) < 200*configs.pass_rate:
                return False
            
        return True

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[200000, 400000], gamma=0.5)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler()

        while not ray.get(self.buffer.check_done.remote()) and self.counter < configs.training_times:

            for i in range(1, 10001):

                data_id = ray.get(self.buffer.get_data.remote())
                data = ray.get(data_id)
    
                b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask, idxes, weights, old_ptr = data
                b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), b_reward.to(self.device)
                b_done, b_steps, weights = b_done.to(self.device), b_steps.to(self.device), weights.to(self.device)
                b_hidden = b_hidden.to(self.device)
                b_comm_mask = b_comm_mask.to(self.device)

                b_next_seq_len = [ (seq_len+forward_steps).item() for seq_len, forward_steps in zip(b_seq_len, b_steps) ]
                b_next_seq_len = torch.LongTensor(b_next_seq_len)

                with torch.no_grad():
                    b_q_ = (1 - b_done) * self.tar_model(b_obs, b_next_seq_len, b_hidden, b_comm_mask).max(1, keepdim=True)[0]

                b_q = self.model(b_obs[:, :-configs.forward_steps], b_seq_len, b_hidden, b_comm_mask[:, :-configs.forward_steps]).gather(1, b_action)

                td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))

                priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

                loss = (weights * self.huber_loss(td_error)).mean()
                self.loss += loss.item()

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                scaler.step(self.optimizer)
                scaler.update()
 
                self.scheduler.step()

                # store new weights in shared memory
                if i % 5  == 0:
                    self.store_weights()

                self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

                self.counter += 1

                # update target net, save model
                if i % configs.target_network_update_freq == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())
                
                if i % configs.save_interval == 0:
                    torch.save(self.model.state_dict(), os.path.join(configs.save_path, '{}.pth'.format(self.counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss / (self.counter-self.last_counter)))
        
        self.last_counter = self.counter
        self.loss = 0
        return self.done


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment(curriculum=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = configs.max_episode_length
        self.counter = 0

    def run(self):
        done = False
        obs, pos, local_buffer = self.reset()
        
        while True:

            # sample action
            actions, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(pos.astype(np.float32)))

            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, 5)
            # take action in env
            (next_obs, next_pos), rewards, done, _ = self.env.step(actions)
            # return data and update observation
            local_buffer.add(q_val[0], actions[0], rewards[0], next_obs, hidden, comm_mask)

            if done == False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(next_obs.astype(np.float32)), torch.from_numpy(next_pos.astype(np.float32)))
                    data = local_buffer.finish(q_val[0], comm_mask)

                self.global_buffer.add.remote(data)
                done = False
                obs, pos, local_buffer = self.reset()

            self.counter += 1
            if self.counter == configs.actor_update_steps:
                self.update_weights()
                self.counter = 0

    def update_weights(self):
        '''load weights from learner'''
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))
    
    def reset(self):
        self.model.reset()
        obs, pos = self.env.reset()
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], obs)
        return obs, pos, local_buffer

