import Agent.agent as dqn_agent
from Agent.memory_utils import StateBuilder
from config import config
from collections import deque
import Agent.plot_utils as plotter
from unityagents import UnityEnvironment
import network
import numpy as np
import pickle

agent = dqn_agent.DQNAgent(network.DQNetwork, config)
state_builder = StateBuilder(config['history_len'])

env = UnityEnvironment('Banana_Windows_x86_64/Banana.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

epsilon = 1.
steps = 0
episodes = 0
total_scores = deque(maxlen=10)
scores = []
train_flag = True
loss = 0.
while train_flag:
    try:
        env_info = env.reset(train_mode=True)[brain_name]
        observation = env_info.vector_observations[0]
        state_builder.reset(observation)
        state = state_builder.get_state()

        total_score = 0
        done = False
        dead = False
        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.choice(4)
            else:
                action = agent.get_action(state)
                action = action.item()
            env_info = env.step(action)[brain_name]
            next_observation = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            state_builder.append(next_observation)
            next_state = state_builder.get_state()

            agent.append_memory(state, action, reward, next_state, done)
            dead = False
            state = next_state

            total_score += reward
            steps += 1

            if steps > config['train_start']:
                loss = agent.train()
                if epsilon > config['epsilon_lower_bound']:
                    epsilon *= config['epsilon_decay']

            if steps % config['sync_target_every'] == 0:
                agent.sync_network()

            if done:
                episodes += 1
                total_scores.append(total_score)
                scores.append(total_score)
                if np.mean(total_scores) > config['target_score']:
                    train_flag = False
                    env.close()

        if episodes % config['print_every'] == 0:
            print("Episode {}".format(episodes),
                  "  Step: {}".format(steps),
                  "  Score: {:.4}".format(total_score),
                  "  epsilon: {:.4}".format(epsilon),
                  "  loss: {:.5}".format(loss))

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Saving model...")
        train_flag = False
        env.close()

agent.save_weights(config['weights_save_path'])
with open(config['rewards_save_path'], 'wb') as f:
    pickle.dump(scores, f)

plotter.save_line_plot(scores, "Total Reward / Episode", config['plot_save_path'])
