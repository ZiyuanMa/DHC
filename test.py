import random
import time
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
from search import find_path
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1) 

# def create_test(agent_range:Union[int,list,tuple], map_range:Union[int,list,tuple], density=None):

#     name = './test{}_{}.pkl'.format(agent_range, map_range)

#     tests = {'maps': [], 'agents': [], 'goals': [], 'opt_steps': []}

#     if type(agent_range) is int:
#         num_agents = agent_range
#     elif type(agent_range) is list:
#         num_agents = random.choice(agent_range)
#     else:
#         num_agents = random.randint(agent_range[0], agent_range[1])

#     if type(map_range) is int:
#         map_length = map_range
#     elif type(map_range) is list:
#         map_length = random.choice(map_range)
#     else:
#         map_length = random.randint(map_range[0]//5, map_range[1]//5) * 5

#     env = Environment(fix_density=None, num_agents=num_agents, map_length=map_length)

#     for _ in tqdm(range(test_num)):
#         tests['maps'].append(np.copy(env.map))
#         tests['agents'].append(np.copy(env.agents_pos))
#         tests['goals'].append(np.copy(env.goals_pos))

#         actions = find_path(env)
#         while actions is None:
#             env.reset()
#             tests['maps'][-1] = np.copy(env.map)
#             tests['agents'][-1] = np.copy(env.agents_pos)
#             tests['goals'][-1] = np.copy(env.goals_pos)
#             actions = find_path(env)

#         tests['opt_steps'].append(len(actions))

#         if type(agent_range) is int:
#             num_agents = agent_range
#         elif type(agent_range) is list:
#             num_agents = random.choice(agent_range)
#         else:
#             num_agents = random.randint(agent_range[0], agent_range[1])

#         if type(map_range) is int:
#             map_length = map_range
#         elif type(map_range) is list:
#             map_length = random.choice(map_range)
#         else:
#             map_length = random.randint(map_range[0]//5, map_range[1]//5) * 5

#         env.reset(num_agents=num_agents, map_length=map_length)

#     tests['opt_mean_steps'] = sum(tests['opt_steps']) / len(tests['opt_steps'])

#     with open(name, 'wb') as f:
#         pickle.dump(tests, f)

def create_test(test_env_settings, num_test_cases):

    for map_length, num_agents, density in test_env_settings:

        name = './test_set/{}length_{}agents_{}density.pth'.format(map_length, num_agents, density)
        print('-----{}length {}agents {}density-----'.format(map_length, num_agents, density))

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)

# def test_model(test_case='test32_40_0.3.pkl'):

#     network = Network()
#     network.eval()
#     network.to(device)

#     with open(test_case, 'rb') as f:
#         tests = pickle.load(f)

#     model_name = 462500
#     while os.path.exists('./models/{}.pth'.format(model_name)):
#         state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
#         network.load_state_dict(state_dict)
#         env = Environment()

#         case = 2
#         show = False
#         show_steps = 100

#         fail = 0
#         steps = []

#         start = time.time()
#         for i in range(test_num):
#             env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
#             done = False
#             network.reset()

#             while not done and env.steps < configs.max_episode_length:
#                 if i == case and show and env.steps < show_steps:
#                     env.render()

#                 obs_pos = env.observe()

#                 actions, q_vals, _ = network.step(torch.FloatTensor(obs_pos).to(device))

#                 if i == case and show and env.steps < show_steps:
#                     print(obs_pos[0, 3:7, 4, 4])
#                     print(q_vals)
#                     print(actions)


#                 _, _, done, _ = env.step(actions)
#                 # print(done)

#             steps.append(env.steps)

#             if not np.array_equal(env.agents_pos, env.goals_pos):
#                 fail += 1
#                 if show:
#                     print(i)


#             if i == case and show:
#                 env.close(True)
        
#         f_rate = (test_num-fail)/test_num
#         mean_steps = sum(steps)/test_num
#         duration = time.time()-start

#         print('--------------{}---------------'.format(model_name))
#         print('finish: %.4f' %f_rate)
#         print('mean steps: %.2f' %mean_steps)
#         print('time spend: %.2f' %duration)


#         model_name -= configs.save_interval


def render_test_case(model, test_case, number):

    network = Network()
    network.eval()
    network.to(device)

    with open(test_case, 'rb') as f:
        tests = pickle.load(f)

    model_name = model
    while os.path.exists('./models/{}.pth'.format(model_name)):
        state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
        network.load_state_dict(state_dict)
        env = Environment()

        case = 2
        show = False
        show_steps = 100

        fail = 0
        steps = []

        start = time.time()
        for i in range(test_num):
            env.load(tests[i][0], tests[i][1], tests[i][2])
            
            done = False
            network.reset()

            while not done and env.steps < configs.max_episode_length:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs, pos = env.observe()

                actions, q_vals, _, _ = network.step(torch.FloatTensor(obs).to(device), torch.FloatTensor(pos).to(device))

                _, _, done, _ = env.step(actions)
                # print(done)

            steps.append(env.steps)

            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                if show:
                    print(i)


            if i == case and show:
                env.close(True)
        
        f_rate = (test_num-fail)/test_num
        mean_steps = sum(steps)/test_num
        duration = time.time()-start

        print('--------------{}---------------'.format(model_name))
        print('finish: %.4f' %f_rate)
        print('mean steps: %.2f' %mean_steps)
        print('time spend: %.2f' %duration)

        model_name -= configs.save_interval


def test_model(model_range):
    network = Network()
    network.eval()
    network.to(device)

    test_set = configs.test_env_settings

    pool = mp.Pool(mp.cpu_count())

    if isinstance(model_range, int):
        state_dict = torch.load('./models/{}.pth'.format(model_range), map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print('-----test model {}-----'.format(model_range))

        for case in test_set:
            # test_model_with_one(case, network)
            print("test case {} {} {}".format(case[0], case[1], case[2]))
            with open('./test_set/test{}_{}_{}.pkl'.format(case[0], case[1], case[2]), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_model_with_one, tests)

            print("result: {} out of {}".format(sum(ret), len(ret)))
        print()

    elif isinstance(model_range, tuple):
        for model_name in range(model_range[0], model_range[1]+1, configs.save_interval):
            state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()

            print('-----test model {}-----'.format(model_name))

            for case in test_set:
                # test_model_with_one(case, network)
                print("test case {} {} {}".format(case[0], case[1], case[2]))
                with open('./test_set/test{}_{}_{}.pkl'.format(case[0], case[1], case[2]), 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_model_with_one, tests)

                print("result: {} out of {}".format(sum(ret), len(ret)))
            print()


def test_model_with_one(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos = env.observe()
    
    done = False
    network.reset()

    while not done and env.steps < configs.max_episode_length:

        actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)), torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), _, done, _ = env.step(actions)

    return np.array_equal(env.agents_pos, env.goals_pos)

def make_animation():
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    test_name = 'test_set/40_length_16_agents_0.3_density.pkl'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)
    test_case = 16
    
    steps = 25
    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('models/200000.pth', map_location=device)
    network.load_state_dict(state_dict)

    env = Environment()
    env.load(tests[test_case][0], tests[test_case][1], tests[test_case][2])

    fig = plt.figure()
            
    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)


        actions, _, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device), torch.from_numpy(pos.astype(np.float32)).to(device))
        (obs, pos), _, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps-env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)


    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    ani.save('dynamic_images.mp4')

    

if __name__ == '__main__':

    # make_animation()
    create_test(test_env_settings=configs.test_env_settings, num_test_cases=configs.num_test_cases)
    # test_model((200000, 210000))
    # make_animation()
    # create_test(1, 20)
