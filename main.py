import warnings
warnings.filterwarnings('ignore')

import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.ppo2 import PPO2

from env import Yamb
from ppo import PPO2_custom

def play_round(yamb, state, agent=None, n=3, render=False):    
    forced_play_next = 0
    
    for i in range(n):
        if render:
            yamb.render()
            
        if agent is None:
            action = yamb.action_space.sample()
        else:
            raise NotImplementedError

        state, reward, done, info = yamb.step(action)
        
        if info["forced_play_next"] > 0:
            forced_play_next = info["forced_play_next"]
        
        if done or info["end_round"]:
            # print("ROUND END")
            # print("INFO:", info["ending"], "\n")
            break
        
    return state, done, forced_play_next, reward


if __name__ == "__main__":
    yamb = Yamb()
    
    num_episodes = 10000
    report_every = 20
    
    # multiprocess environment
    yamb = make_vec_env(Yamb)

    model = PPO2_custom(MlpPolicy, yamb, verbose=1, tensorboard_log="./ppo2_yamb/")
    model.learn(total_timesteps=500000, log_interval=4000)
    model.save("ppo1_yamb")
    
    exit()
    
    # BELOW IS OLD TESTING CODE
    """
        for episode_n in range(num_episodes):
            state = yamb.reset()
            done = False
            forced_play_next = 0
            
            num_steps = 0
            reward_sum = 0
            # print("----------------------------------EPISODE START----------------------------------")
            # both players play rounds until the episode is over
            while not done:
                # one round consists of 3 steps; 3 throws of the dice
                # print("----------------------------------FIRST PLAYER-------------------------------------\n")
                state, done, forced_play_next, reward = play_round(yamb, state)
                yamb.switch(forced_play_next)
                
                reward_sum += reward
                
                if done:
                    break
                
                # print("----------------------------------SECOND PLAYER------------------------------------\n")
                # same goes for the opponent
                state, done, forced_play_next, _ = play_round(yamb, state)
                yamb.switch(forced_play_next)
                
                num_steps += 1
                
            if episode_n % report_every == 0:
                print(str(episode_n) + ": NUM_STEPS - " + str(num_steps) + ", REWARD_SUM - " + str(reward_sum))
    """