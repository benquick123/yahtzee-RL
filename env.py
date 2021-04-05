import os
import time
from collections import Counter

import gym
import numpy as np

gym.register("Yamb-v0")

class Yamb(gym.Env):
    metadata = {'render.modes': ["ansi"]}
    reward_range = (-1, 1500)
    spec = gym.spec("Yamb-v0")
    
    def __init__(self):
        
                            #    1, 2, 3, 4, 5, 6, min, max, tris, straight, fh, carr, yamb
        one_column_max_scores = [5, 10, 15, 20, 25, 30, 5, 30, 38, 66, 58, 64, 80]
        # Note: observations are initialized to -1
        self.observation_space = gym.spaces.MultiDiscrete([value+1 for i in range(5) for value in one_column_max_scores] + [value+1 for i in range(5) for value in one_column_max_scores] + [6 for i in range(5)] + [3] + [14] * 2)
        
        # what action to play / no action, end_round, which dice to rethrow
        self.action_space = gym.spaces.MultiDiscrete([4 * 13 + 1] + [2] + [2] * 5)
    
    def step(self, action):
        self.action = action
        
        to_play = [-1, -1] if action[0] == 0 else [(action[0]-1) % 13, (action[0]-1) // 13]
        no_play = action[0] == 0
        end_round = bool(action[1])
        rethrow_dice = action[2:]
        
        my_state = self.state[:5*13].reshape(13, 5)
        dice_values = self.state[2*5*13:2*5*13+5]
        throw_number = self.state[2*5*13+5]
        forced_play = self.state[2*5*13+6]
        self_forced_play = self.state[2*5*13+7]
        
        reward, done, ending_message = self.evaluate_action(my_state, to_play, no_play, end_round, throw_number, forced_play)
        info = {"ending": ending_message,
                "forced_play_next": 0,
                "end_round": end_round}
        
        # was the play announced?
        if to_play[1] == 3 and throw_number == 0 and forced_play == 0:
            info["forced_play_next"] = self.state[2*5*13+7] = self_forced_play = to_play[0]
            
        if end_round or throw_number == 2:    
            if self_forced_play != 0:
                reward = self.get_reward(self_forced_play, dice_values, throw_number)
                self.state[int(self_forced_play * 5 + 3)] = reward

            elif forced_play != 0:
                reward = self.get_reward(forced_play, dice_values, throw_number)
                self.state[int(forced_play * 5 + 4)] = reward
                
            elif not done:
                reward = self.get_reward(to_play[0], dice_values, throw_number)
                self.state[to_play[0] * 5 + to_play[1]] = reward
                     
        if not done:
            self.state[2*5*13:2*5*13+5] = self.throw_dice(dice_values, rethrow_dice)
            self.state[2*5*13+5] += 1
            
        if np.all(my_state != -1):
            reward += self.get_final_reward(my_state)
            done = True
    
        return self.state, reward, done, info
    
    def get_final_reward(self, my_state):
        reward = 0
        
        # first, sum up 1-6; add +30 if the score in a column is >60
        reward += np.sum(my_state[:6, :])
        reward += np.sum(np.sum(my_state[:6, :], axis=0) > 60) * 30
        
        # second, calculate max-min * number of 1s in that column.
        reward += np.sum((my_state[7, :] - my_state[6, :]) * my_state[0, :])
        
        # sum up all remaining rows.
        reward += np.sum(my_state[8:, :])
        
        return reward
    
    def get_reward(self, row, dice_values, throw_number):
        reward = 0
        counted = Counter(dice_values)
        
        if 0 <= row < 6:
            # 1,2,3,4,5,6
            reward = np.sum(dice_values == (row + 1)) * (row + 1)
        
        elif 6 <= row < 8:
            # min, max
            reward = np.sum(dice_values)
            
        elif row == 8 and 3 in counted.values() and 2 not in counted.values():
            # tris: +10
            key = [k for k, v in counted.items() if v == 3][0]
            reward = key * 3 + 10
        
        elif row == 9 and (len({1,2,3,4,5} - set(dice_values)) == 0 or len({2,3,4,5,6} - set(dice_values))):
            # straight: 66, 56, 46
            reward = 66 - throw_number * 10
        
        elif row == 10 and 3 in counted.values() and 2 in counted.values():
            # full house: 3+2
            reward = np.sum(dice_values) + 30
        
        elif row == 11 and 4 in counted.values():
            # carriage: 4 of the same
            # key = [k for k, v in counted.items() if v == 4][0]
            reward = np.sum(dice_values) + 40
        
        elif row == 12 and 5 in counted.values():
            # yamb: all the same
            reward = np.sum(dice_values) + 50
            
        return reward
    
    def evaluate_action(self, my_state, to_play, no_play, end_round, throw_number, forced_play):
        ending_message = None
        reward = 0
        done = False
        
        # only check the action in case round is ending; otherwise we don't care what the agent wants to do.
        if (end_round or throw_number == 2) and forced_play == 0 :
            # is it trying to play an illegal move in 1st column?
            if to_play[1] == 0 and np.any(my_state[:to_play[0], to_play[1]] == -1):
                reward = -1
                done = True
                ending_message = "Illegal move was played in the 1st column."
            
            # is it trying to play an illegal move in 2nd column?
            if to_play[1] == 1 and np.any(my_state[to_play[0]+1:, to_play[1]] == -1):
                reward = -1
                done = True
                ending_message = "Illegal move was played in the 2nd column."
            
            # is it playing a forced_play in the last throw?
            if to_play[1] == 3:
                reward = -1
                done = True
                ending_message = "Trying to play forced_play in the last step of the round"
                
            # check if play is specified and the end of the round
            if no_play:
                reward = -1
                done = True
                ending_message = "Round is ending, but play not specified."
                
            # is the target cell still empty?
            if my_state[to_play[0], to_play[1]] != -1:
                reward = -1
                done = True
                ending_message = "Played cell is not empty."
        
        # is it playing 4th column after first round?
        # if to_play[1] == 3 and throw_number > 0:
        #     done = True
        #     reward = -1
        #     ending_message = "Forced move played and throw_number > 0"
            
        # is the agent performing forced_play in non-empty cell?
        if to_play[1] == 3 and throw_number == 0 and forced_play == 0 and my_state[to_play[0], to_play[1]] != -1:
            done = True
            reward = -1
            ending_message = "Player forced_play to non-empty cell."

        return reward, done, ending_message
        
    def throw_dice(self, dice_values=np.zeros(5), rethrow_dice=np.ones(5)):
        return np.where(rethrow_dice == 1, np.random.randint(1, 7, size=5), dice_values)
    
    def reset(self):
        self.match_number = 0
        
        self.state = np.ones(self.observation_space.shape[0]) * -1
        self.state[2*5*13:2*5*13+5] = self.throw_dice()
        self.state[2*5*13+5] = 0
        self.state[2*5*13+6] = 0
        self.state[2*5*13+7] = 0
        
        self.action = None
        
        return self.state
    
    def render(self, mode="ansi"):
        s = ""
        
        s += "ACTION\n"
        if self.action is not None:
            s += "Position: " + str([-1, -1] if self.action[0] == 0 else [(self.action[0]-1) % 13, (self.action[0]-1) // 13]) + "\n"
            s += "End round: " + str(bool(self.action[1])) + "\n"
            s += "Rethrow dice: " + str(self.action[2:])
        else:
            s += str(self.action)
        s += "\n\n"
        
        s += "STATE\n"
        s += "my:\n"
        s += str(self.state[:5*13].reshape(13, 5))
        s += "\n\n"
        
        s += "opponent's:\n"
        s += str(self.state[5*13:2*5*13].reshape(13, 5))
        s += "\n\n"
        
        s += "dice numbers: " + str(self.state[2*5*13:2*5*13+5]) + "\n"
        s += "throw #: " + str(self.state[2*5*13+5]) + "\n"
        s += "forced play: " + str(self.state[2*5*13+6]) + "\n"
        s += "self-forced play: " + str(self.state[2*5*13+7]) + "\n"
        
        print(s)
        time.sleep(0.5)
        return s
    
    def switch(self, forced_play_next):
        tmp_state = np.array(self.state)
        
        # switch my and opponent's state
        self.state[:5*13] = tmp_state[5*13:2*5*13]
        self.state[5*13:2*5*13] = tmp_state[:5*13]
        
        # reinitialize other state parameters
        self.state[2*5*13:2*5*13+5] = self.throw_dice()     # dice
        self.state[2*5*13+5] = 0                            # throw # 
        self.state[2*5*13+6] = forced_play_next             # forced_play
        self.state[2*5*13+7] = 0                            # self-forced play
        
        self.action = None
        