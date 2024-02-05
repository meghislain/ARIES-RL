from typing import Dict, Union

import gym
import os
import sys
import random
import numpy as np
from gym import spaces
from skimage.morphology import disk, square, rectangle, cube, octahedron, ball, octagon, star
#from skimage.morphology import square
import random
import wandb
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from stable_baselines3.common.type_aliases import GymStepReturn
from ARIES.trajectory import create_signal_position, create_noisy_signal, create_breathing_signals, create_breathing_signals_reel, create_breathing_signals_reel_3D
from scipy.interpolate import CubicSpline
from ARIES.dose_evaluation import compute_DVH, compute_DVH_OAR
from scipy.ndimage import gaussian_filter

# Application d'un filtre gaussien

# os.chdir("/linux/meghislain/Project/opentps/opentps_core")
# currentWorkingDir = os.getcwd()
# print(currentWorkingDir)
# sys.path.append(currentWorkingDir)
# from opentps.core.data.dynamicData._breathingSignals import SyntheticBreathingSignal


class TumorEnv(gym.Env):
    """
    Base class for GridWorld-based MultiObsJ Environments 4x4  grid world.

    """
    
    def __init__(
        self,
        num_col: int = 12,
        num_row: int = 12,
        size: int = 8,
        target_size: int = 2,
        n_test_episode: int = 5,
        n_train_episode: int = 40,
        n_epoch: int = 30,
        signal_length: int = 75,
        dose_quantity: float = 1.0,
        name: str = "1",
        saving_path: str = "None",
        mode: int = -1,
        discrete_actions: bool = True,
        channel_last: bool = True,
        breathingSignal: str = "None",
        save_gif: int = 1,
        moving: int = 1,
        amplitude: float = 2,
        form: str = "ball",
        frequency : int = 10,
        le : int = 2,
        inc : int = 1
    ):
        super().__init__()

        self.save_gif = save_gif
        self.targetSize = target_size
        self.n_obs = 4
        self.saving_path = saving_path
        self.name = name
        self.moving = moving
        self.amplitude = amplitude
        self.time_to_shuffle = 0
        self.count_time_shuffle = 0
        self.form = form
        self.inc = inc
        self.frequency = frequency
        self.gauss = self.gkern(long = (2*le) +1, sig = 0.55) #un au centre 0 ailleurs #remettre 0.55
        print(self.gauss)
        maximum_gauss = np.max(self.gauss)
        rapport = 1/maximum_gauss
        self.distr_tir = self.gauss*rapport
        self.le = int(np.floor(len(self.gauss)/2))
        if save_gif :
            if not os.path.exists(self.saving_path + "/" + self.name):
                 os.umask(0)
                 os.makedirs(self.saving_path + "/" + self.name) # Create a new directory because it does not exist
                 print("New directory created to save the data: ", self.saving_path + "/" + self.name)

        #self.img_size = [2, num_row, num_col]
        self.zone = 2 # zone considered around the tumor for the accumulated dosemap
        self.discrete_actions = discrete_actions
        self.SignalLength=int(signal_length)
        self.dose_size = 1#(2*self.targetSize)+1
        if discrete_actions:
            self.action_space = spaces.Discrete(5 + 1)
        else:
            self.action_space = spaces.Box(0, 1, (5+ 1,))
        self.l = [2,1]
        if self.n_obs == 3 :
            self.img_size = [3, num_row, num_col]
            self.doseMaps = np.zeros(self.img_size, dtype=np.float64)
            self.observation_space = spaces.Dict(
            spaces={
                #"percentage_recovery": spaces.Box(0,1, [1,1], dtype=np.float64),
                "doseMaps": spaces.Box(-1,1,self.img_size, dtype=np.float64)
            }
            )
        if self.n_obs == 4 :
            self.img_size = [2, num_row, num_col]
            self.doseMaps = np.zeros([self.dose_size, num_row, num_col], dtype=np.float64)
            self.incertitude = np.zeros(self.img_size, dtype=np.float64)
            self.observation_space = spaces.Dict(
            spaces={
                #"percentage_recovery": spaces.Box(0,1, [1,1], dtype=np.float64),
                "doseMaps": spaces.Box(-1,1,[self.dose_size, num_row, num_col], dtype=np.float64),
                "incert" : spaces.Box(-1,1,self.img_size, dtype=np.float64) #[1,num_row, num_col]
            }
            )   
        if self.n_obs == 2 :
            self.img_size = [2, num_row, num_col]
            self.doseMaps = np.zeros(self.img_size, dtype=np.float64)
            self.observation_space = spaces.Dict(
            spaces={
                "beam position": spaces.Box(0,1, [2,1], dtype=np.float64),
                "doseMaps": spaces.Box(-1,1,self.img_size, dtype=np.float64)
            }
            )
        self.num_col = num_col
        self.num_row = num_row
        self.depth = self.num_row
        self.ref_position = [int(self.depth / 2),int(self.num_row / 2),int(self.num_col / 2)]
        self.beam_pos = np.zeros((num_row, num_col), dtype=np.float64)
        self.beam_pos[self.ref_position[0], self.ref_position[1]] = 1
        self.perfect_DM = np.zeros((self.depth, num_row, num_col), dtype=np.float64)
        self.DM_i = np.zeros((self.depth, num_row, num_col), dtype=np.float64)
        # self.DM_i_noisy = np.zeros((self.depth,num_row, num_col), dtype=np.float64)
        self.PTV = self.observeEnvAs2DImage_plain(pos = self.ref_position, dose_q=1, targetSize = self.targetSize, form = self.form)
        self.number_pixel_to_touch = len(self.PTV[self.PTV==1])
        self.mask = self.observeEnvAs2DImage_plain(pos = self.ref_position, dose_q=1, targetSize = self.targetSize+self.zone, form = self.form)
        self.incert_image = self.mask
        self.n_test_episode = n_test_episode
        self.n_train_episode = n_train_episode
        self.count = 0
        self.max_count = 100
        self.state = 0
        #self.action2str = ["left", "up", "right", "down","shoot", "nothing", ""]

        self.observation = []
        self._beam_position = np.zeros((2, 1), dtype=np.uint8)
        self.n_targets = 9
        self.n_dose = self.n_targets*dose_quantity

        self.energy = int(self.depth/2) + self.targetSize
        #self.u = self.n_dose
        # self.envSize = size 
        #self.doseMaps[0] = self.perfect_DM
        if self.n_obs == 2 :
            self.doseMaps[0] = 5*(self.perfect_DM[self.energy] - self.DM_i[self.energy])/self.SignalLength
            self.doseMaps[1] = self.mask[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size), "beam position" : self._beam_position/self.num_col})#, "incert": self.mask.reshape([1,num_row, num_col])})
        if self.n_obs == 3 :
            self.doseMaps[0] = 5*(self.perfect_DM[self.energy] - self.DM_i[self.energy])/self.SignalLength
            self.doseMaps[1] = self.beam_pos
            self.doseMaps[2] = self.mask[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})#, "incert": self.mask.reshape([1,num_row, num_col])})
        if self.n_obs == 4 :
            self.doseMaps[:] = self.DM_i[self.energy]#self.ref_position[0]-self.targetSize:self.ref_position[0]+self.targetSize+1]#self.energy]#100*(self.perfect_DM[self.energy] - self.DM_i[self.energy])/self.SignalLength
            #self.doseMaps[1] = self.beam_pos
            #self.doseMaps[2] = self.mask
            self.incertitude[0] = self.mask[self.energy]
            self.incertitude[1] = self.beam_pos
            #self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})#, "incert": self.mask.reshape([1,num_row, num_col])})
            self.observation.append({"doseMaps": self.doseMaps.reshape([self.dose_size, self.num_row, self.num_col]), "incert": self.incertitude.reshape(self.img_size)})
        if breathingSignal == "None":

            grid = np.ones((self.n_train_episode+self.n_test_episode,3))*self.ref_position
            
            np.random.seed(0);np.random.shuffle(grid)

            #signal_matrice = create_signal_position(grid, self.amplitude, self.moving, self.SignalLength, self.frequency)
            signal_matrice = create_breathing_signals_reel_3D(grid, self.amplitude, self.moving, self.SignalLength)
            # z_position = np.ones((self.n_train_episode+self.n_test_episode,1))*self.energy
            # signal_matrice = np.concatenate((z_position, x_y_position), axis = 1)
            
            if int(self.n_test_episode+self.n_train_episode)==len(grid): #pour après sélectionner la proportion de data utilisée
                self.breathingSignal_training = signal_matrice[int(self.n_test_episode):]
            else :
                self.breathingSignal_training = signal_matrice[int(self.n_test_episode):int(self.n_test_episode+self.n_train_episode)]
            self.breathingSignal_validation = signal_matrice[:int(self.n_test_episode)]
            print(" training : ", self.breathingSignal_training[:,0,:], len(self.breathingSignal_training))
            print(" validation : ", self.breathingSignal_validation[:,0,:], len(self.breathingSignal_validation))
        
        self._beam_position = [int(self.num_row/ 2),int(self.num_col / 2)]
        self._beam_dx= self.targetSize + 1 
        self._beam_dy= self.targetSize + 1
        self.count_validation = 0
        self.count_training = 0
        self.n_training = 0
        self.n_validation = 0
        self.count_episode = 0
        self.count_epoch = 0
        self.action = 0
        self.dose_quantity = dose_quantity
        #self.envSize = size
        self.mode = mode
        self.curTimeStep = 0
        self.done = False
        

    def step(self, action):
        #print(action)
        self.observation = []
        self._tumor_position = self._signal_position[self.curTimeStep]
        #self._tumor_position #+ [random.choice(array_ran),random.choice(array_ran)]#+ np.random.normal(0,0,2)
        
        if self.inc == 1 :
            if self.curTimeStep % 10 == 0:
                self.noisy_tumor_position = self._signal_position[self.curTimeStep]
                self.noisy_pos[self.curTimeStep] = self.noisy_tumor_position[1:]
                self.mask_shifted = self.observeEnvAs2DImage_plain(pos = self.noisy_tumor_position, dose_q=1, targetSize = self.targetSize+self.zone-1, form = self.form)
                self.incert_image = self.mask_shifted
                #self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
                #self.noisy_perfect_DM = gaussian_filter(self.noisy_perfect_DM, 1.5)
            else :
                self.previous_noisy_position = self.noisy_tumor_position
                self.noisy_tumor_position[0] = self._signal_position[self.curTimeStep][0]
                self.noisy_tumor_position[1] = self._signal_position[self.curTimeStep][1] + np.random.normal(0, (self.curTimeStep % 10)*0.12, 1)
                self.noisy_tumor_position[2] = self._signal_position[self.curTimeStep][2] + np.random.normal(0, (self.curTimeStep % 10)*0.06, 1)
                #self.noisy_tumor_position = self._signal_position[self.curTimeStep] + np.random.normal(0, (self.curTimeStep % 10)*0.12, 2)
                self.noisy_pos[self.curTimeStep] = self.noisy_tumor_position[1:]
                sigma = 0.3
                shift = np.zeros(3)
                shift[0] = 0
                shift[2] = self.noisy_tumor_position[2] - self.previous_noisy_position[2]
                shift[1] = self.noisy_tumor_position[1] - self.previous_noisy_position[1]
                # shift[0] = self.noisy_tumor_position[0] - self._signal_position[self.curTimeStep-1][0]
                # shift[1] = self.noisy_tumor_position[1] - self._signal_position[self.curTimeStep-1][1]
                # self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
                #self.noisy_perfect_DM = gaussian_filter(self.noisy_perfect_DM, 1.5)
                self.incert_image = scipy.ndimage.shift(self.incert_image, shift, order=1)
                self.incert_image = gaussian_filter(self.incert_image, sigma)
        else :
            self.noisy_tumor_position = self._signal_position[self.curTimeStep]
            self.mask_shifted = self.observeEnvAs2DImage_plain(pos = self.noisy_tumor_position, dose_q=1, targetSize = self.targetSize+self.zone-1, form = self.form)
            self.incert_image = self.mask_shifted
            # self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
            
        self.perfect_DM = self.observeEnvAs2DImage(pos = np.round(self._tumor_position), dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        sigma = 1.3
        #self.PTV_tumpos = self.observeEnvAs2DImage_plain(pos = np.round(self._tumor_position), dose_q=1, targetSize = self.targetSize, form = self.form)
        # self.perfect_DM_ag = self.observeEnvAs2DImage_plain(pos = self._tumor_position, dose_q=1, targetSize = self.targetSize+self.zone, form = self.form)
        # self.autour_tumeur = self.perfect_DM_ag - self.PTV_tumpos
        #self.perfect_DM = gaussian_filter(self.perfect_DM, sigma)*self.perfect_DM_ag

        # self._targets_position = np.argwhere(self.perfect_DM!=0)#/self.envSize
        self.DM_i = np.zeros((self.depth,self.num_row,self.num_col))
        # self.DM_i_noisy = np.zeros((self.depth,self.num_row, self.num_col))
        # shift = np.zeros(3)
        # shift[0] = 0
        # shift[2] = self.noisy_tumor_position[2] - self.ref_position[2]
        # shift[1] = self.noisy_tumor_position[1] - self.ref_position[1]
        # self.DM_i_noisy = scipy.ndimage.shift(self.DMi_inRef_noisy, shift, order=1)
        shift = np.zeros(3)
        shift[0] = 0
        shift[2] = np.round(self._tumor_position[2]) - self.ref_position[2]
        shift[1] = np.round(self._tumor_position[1]) - self.ref_position[1]
        self.DM_i = scipy.ndimage.shift(self.DMi_inRef, shift, order=1)
        # for i in self.p[np.abs(self.active_targets-self.touched_targets)>0]:
        #     self.DM_i[self._targets_position[i,0],self._targets_position[i,1]] = self.active_targets[i]-self.touched_targets[i]
        self.curTimeStep += 1
        reward_dose = 0
        reward_distance = 0
        reward = 0
        tir_inside = False
        tir = False
        if action == 0: # move to the left
            self._beam_position[0] = max(self._beam_position[0]-self._beam_dx, self.le)
        elif action == 1: # move upward
            self._beam_position[1] = min(self._beam_position[1]+self._beam_dy, self.num_row-1-self.le)
        elif action == 2: # move to the right
            self._beam_position[0] = min(self._beam_position[0]+self._beam_dx, self.num_col-1-self.le)
        elif action == 3: # move downward
            self._beam_position[1] = max(self._beam_position[1]-self._beam_dy, self.le)
        elif action == 4 : # shoot a gaussian dose
            tir = True
            x_ = int(np.round(self._beam_position[0]))
            y_ = int(np.round(self._beam_position[1]))
            if self.perfect_DM[self.energy,x_,y_]-self.DM_i[self.energy,x_,y_] > 0.2 :
                reward_dose += 1
            else :
                reward_dose -= 1
            # print(reward_dose)
            self.DM_i[self.energy, x_ - self.le : x_ + self.le +1 , y_ - self.le : y_ + self.le + 1] += self.gauss # gaussian_filter(3) #gaussienne autour de ce pixel mais pas de 2 ou print les perfectdose map et dosemap
            #self.DM_i_noisy[self.energy, x_, y_] += 1

            # if self.mode != -1 :
            x_in_ref = int(np.round(self._beam_position[0] - self._tumor_position[1])) + int(self.num_row/2)
            y_in_ref = int(np.round(self._beam_position[1] - self._tumor_position[2])) + int(self.num_col/2)
            if x_in_ref >= self.le and x_in_ref < (self.num_row-self.le) :
                if y_in_ref >= self.le and y_in_ref < (self.num_col-self.le) :
                    self.DMi_inRef[self.energy, x_in_ref - self.le : x_in_ref + self.le +1 , y_in_ref - self.le : y_in_ref + self.le + 1] += self.gauss
     
            x_in_ref_noisy = int(np.round(self._beam_position[0] - self.noisy_tumor_position[1])) + int(self.num_row/2)
            y_in_ref_noisy = int(np.round(self._beam_position[1] - self.noisy_tumor_position[2])) + int(self.num_col/2)
            if x_in_ref_noisy >= self.le and x_in_ref_noisy < (self.num_row-self.le) :
                if y_in_ref_noisy >= self.le and y_in_ref_noisy < (self.num_col-self.le) :
                    self.DMi_inRef_noisy[self.energy, x_in_ref_noisy - self.le:x_in_ref_noisy + self.le+1, y_in_ref_noisy -self.le:y_in_ref_noisy +self.le+1] += self.distr_tir


            
        elif action == 5 : #5 étant ne rien faire, 6 7 8 9 10 11 12 etant les differentes energies du faisceau de protons
            reward_dose -= 5*(1-(len(self.DMi_inRef[(self.PTV[self.energy] == 1) & (self.DMi_inRef[self.energy] >= 2)])/len(self.PTV[self.PTV[self.energy] == 1])))
            if self.energy == (int(self.depth/2) - self.targetSize) :
                self.energy = int(self.depth/2) + self.targetSize
            else : 
                self.energy = self.energy-1#max(0,self.energy-1) #int(int(self.depth/2) + 6 + self.targetSize - action)
        # elif action == 6 :
        #     self.energy = min(int(self.depth/2) + self.targetSize + 1, self.energy+1)#min(self.depth-1,self.energy+1)
        #     #reward_dose -= 20
        self.distance = np.linalg.norm(self._beam_position - self._tumor_position[1:])
        if action != 5 and action != 4 :
            if self.distance > self.targetSize +1 :
                reward_distance -= self.distance/(self.num_row/1.5)

        self._beam_dx = self.targetSize + 1
        self._beam_dy = self.targetSize + 1

        self._last_ponctual_distance = np.linalg.norm(self._beam_position - self.noisy_tumor_position[1:])
        # beam_int_x, beam_int_y = int(np.round(self._beam_position[0])), int(np.round(self._beam_position[1]))
        
        if self._last_ponctual_distance <= (self.targetSize + 1):
            self._beam_dx = 1 #1/self.envSize
            self._beam_dy = 1 #1/self.envSize
            
            # reward_dose -= np.sum(np.abs(self.perfect_DM[beam_int_x - self.le : beam_int_x + self.le +1 , beam_int_y - self.le : beam_int_y + self.le + 1]-self.DM_i[beam_int_x - self.le : beam_int_x + self.le +1 , beam_int_y - self.le : beam_int_y + self.le + 1])/(self.SignalLength))
        # zone_PTV_OAR = np.ones_like(self.autour_tumeur) - self.autour_tumeur
        # zone_PTV = (self.perfect_DM-self.DM_i)*self.PTV_tumpos
        # zone_OAR = -(np.ones_like(self.perfect_DM)-self.perfect_DM_ag)*(self.perfect_DM-self.DM_i)
        # zone_autour = (self.DM_i)*self.autour_tumeur
        #if np.linalg.norm(self._beam_position - self._tumor_position[1:]) > self.targetSize + 1 :
        #    reward_distance -= np.linalg.norm(self._beam_position - self._tumor_position[1:])
        #rew = np.where(zone_autour>=2, ((zone_autour-2)**2), 0)
        #reward_dose -= 4*np.sum((np.abs(self.perfectDM_inref-self.DMi_inRef))/self.SignalLength)
        
        #reward_dose -= 5*np.sum((zone_PTV)**2)/self.SignalLength
        # if action == 4 :
            # if np.sum(self.perfect_DM[beam_int_x,beam_int_y]-self.DM_i[beam_int_x,beam_int_y]) >= 0 :
            #     reward_dose += 1
            # else :
            #     reward_dose -= 1
        #reward_dose -= np.sum((np.abs(self.perfect_DM-self.DM_i))**2)/self.SignalLength
        # reward_dose -= 5*np.sum((np.abs(self.perfect_DM-self.DM_i)*zone_PTV_OAR)**2)/self.SignalLength
        # reward_dose -= 5*np.sum(rew)/self.SignalLength 
        #reward_dose -= 3*self.weighted_squared_abs_value(self.perfect_DM-self.DM_i,2)/self.SignalLength
        #reward_dose -= 3*self.weighted_abs_value(zone_PTV, 3)/self.SignalLength
        #reward_dose -= self.weighted_zone(self.perfect_DM-self.DM_i, self.autour_tumeur, self.perfect_DM, self.perfect_DM_ag, 5)/self.SignalLength
        
        #reward_dose -= (self.weighted_abs_value(zone_OAR,5)+self.weighted_abs_value(zone_PTV,2))/(3*self.SignalLength)
        #reward_dose -= (self.weighted_abs_value(zone_OAR,5)+np.sum(zone_PTV**2))/(3*self.SignalLength)
        self.done = bool(self.curTimeStep >= self.SignalLength)
        # print(int(self.depth/2))
        # print("milieu", self.perfect_DM[int(self.depth/2)][self.perfect_DM[int(self.depth/2)]!=0])
        # print(int(self.depth/2)+ self.targetSize)
        # print("milieu+rayon", self.perfect_DM[int(self.depth/2)+ self.targetSize][self.perfect_DM[int(self.depth/2)+ self.targetSize]!=0])
        # print(int(self.depth/2)- self.targetSize)
        # print("milieu-rayon",self.perfect_DM[int(self.depth/2)- self.targetSize][self.perfect_DM[int(self.depth/2)- self.targetSize]!=0])
        # print(int(self.depth/2)+ self.targetSize+1)
        # print("milieu+rayon+1",self.perfect_DM[int(self.depth/2)+ self.targetSize+1][self.perfect_DM[int(self.depth/2)+ self.targetSize+1]!=0])
        # print(int(self.depth/2)- self.targetSize-1)
        # print("milieu-rayon-1",self.perfect_DM[int(self.depth/2)- self.targetSize-1][self.perfect_DM[int(self.depth/2)- self.targetSize-1]!=0])
        # print(int(self.depth/2)+ self.targetSize+2)
        # print("milieu+rayon+2",self.perfect_DM[int(self.depth/2)+ self.targetSize+2][self.perfect_DM[int(self.depth/2)+ self.targetSize+2]!=0])
        # print(int(self.depth/2)- self.targetSize-2)
        # print("milieu-rayon-2",self.perfect_DM[int(self.depth/2)- self.targetSize-2][self.perfect_DM[int(self.depth/2)- self.targetSize-2]!=0])
       
        reward = reward_distance + reward_dose  
        self.sum_reward += reward

        info = {}
        # if self.mode != -1 :
        #     self.allDoseMap[self.curTimeStep-1] = self.DM_i
        #     self.valid_beam_pos[self.curTimeStep-1] = self._beam_position
        #     self.sum_reward_b += reward_dose
        #     self.sum_reward_distance += reward_distance
        #     self.allCumReward_b[self.curTimeStep-1] = self.sum_reward_b
        #     self.allCumReward_distance[self.curTimeStep-1] = self.sum_reward_distance
        #     self.allCumReward[self.curTimeStep-1] = self.sum_reward
        #     self.allReward[self.curTimeStep-1] = reward
        #     self.all_noisy_Perfect_DM[self.curTimeStep-1] = self.noisy_perfect_DM
        #     self.allPerfect_DM[self.curTimeStep-1] = self.perfect_DM
        #     self.all_energies[self.curTimeStep-1] = self.energy
        #     self.AllDMi_inRef[self.curTimeStep-1] = self.DMi_inRef
        
        self.beam_pos = np.zeros((self.num_row,self.num_col), dtype=np.float64)
        self.beam_pos[self._beam_position[0],self._beam_position[1]] = 1
        if self.n_obs == 2 :
            self.doseMaps[0] = 5*(self.perfectDM_inref[self.energy]-self.DMi_inRef_noisy[self.energy])/self.SignalLength
            self.doseMaps[1] = self.incert_image[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size), "beam position" : self._beam_position.reshape([2,1])/self.num_col})#, "incert": self.mask.reshape([1,num_row, num_col])})
        # if self.n_obs == 3 :
        #     #self.doseMaps[0] = 5*(self.perfectDM_inref-self.DMi_inRef_noisy)/self.SignalLength
        #     self.doseMaps[0] = 5*(self.noisy_perfect_DM[self.energy] - self.DM_i_noisy[self.energy])/self.SignalLength
        #     self.doseMaps[1] = self.beam_pos
        #     self.doseMaps[2] = self.incert_image[self.energy]
        #     self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})
        if self.n_obs == 4 :
            self.doseMaps[:] = self.DMi_inRef_noisy[self.energy]/4#self.ref_position[0]-self.targetSize:self.ref_position[0]+self.targetSize+1]#[self.energy]/4#100*(self.perfectDM_inref[self.energy]-self.DMi_inRef_noisy[self.energy])/self.SignalLength
            self.incertitude[0] = self.incert_image[self.energy]
            self.incertitude[1] = self.beam_pos
            
            #self.doseMaps[2] = self.mask
            # self.doseMaps[2] = np.zeros((self.num_row, self.num_col))#np.where(self.noisy_perfect_DM != 0, 1, self.noisy_perfect_DM)
            self.observation.append({"doseMaps": self.doseMaps.reshape([self.dose_size, self.num_row, self.num_col]), "incert": self.incertitude.reshape(self.img_size)})
        # print(action)
        # print(reward_dose)
        # print(self.doseMaps[:][self.doseMaps[:]!=0])
        # print(self.incertitude[0][self.incertitude[0]!=0])
        # print(self.incertitude[1][self.incertitude[1]!=0])
        #self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})#, "incert": self.mask.reshape([1,self.num_row, self.num_col])})
        #self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size), "incert": self.incert_image.reshape([1,self.num_row, self.num_col])})
        return self.observation[0], reward, self.done, info

    def reset(self):
        self.noisy_pos = np.zeros((self.SignalLength,2))
        self.observation = []
        self.curTimeStep = 0
        self.count_episode += 1
        if self.mode == -1:
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle):
                self.count_time_shuffle += 1
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle) and self.count_time_shuffle == 1: 
                np.random.shuffle(self.breathingSignal_training)
                self.n_training = 0
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle) and self.count_time_shuffle == 2 :
                if self.time_to_shuffle == (self.n_train_episode-1):
                    self.time_to_shuffle = 0
                else :
                    self.time_to_shuffle += 1
                self.count_time_shuffle = 0
            self.positions = self.breathingSignal_training
            self._signal_position = self.positions[int(self.count_training % self.n_train_episode)]
            self._tumor_position = self._signal_position[self.curTimeStep]
            self.noisy_signal_position = self._signal_position #+ create_noisy_signal(self.moving, self.SignalLength, self.frequency)
            self.noisy_tumor_position = self.noisy_signal_position[self.curTimeStep]
            print('training mode : ', self.n_training, self._tumor_position)
            self.count_training += 1
            self.n_training += 1
            self.sum_reward = 0
        else:
            if int(self.n_validation % self.n_test_episode) == 0 :
                np.random.shuffle(self.breathingSignal_validation)
            # if self.save_gif == 1 :
            # #     #self.render()
            # #     # if self.n_validation == 1:
            # #     #     self.render()
            # #     # if self.n_validation == 3501:
            # #     #     self.render()
            # #     # # if self.n_validation == 9999:
            # #     # #     self.render()
            #      #if int(self.n_validation % self.n_test_episode) == 1:
            #     if (self.n_validation % 301) == 1 :
            #         self.render()
            # self.valid_beam_pos = np.zeros((self.SignalLength,2))
            # self.valid_tumor_pos = np.zeros((self.SignalLength,2))
            # self.allDoseMap = np.zeros((self.SignalLength, self.depth, self.num_row, self.num_col))
            # self.all_noisy_Perfect_DM = np.zeros((self.SignalLength, self.depth, self.num_row, self.num_col))
            # self.allPerfect_DM = np.zeros((self.SignalLength, self.depth, self.num_row, self.num_col))
            # self.allCumReward_b = np.zeros(self.SignalLength)
            # self.allCumReward_distance = np.zeros(self.SignalLength)
            # self.allCumReward = np.zeros(self.SignalLength)
            # self.allReward = np.zeros(self.SignalLength)
            self.sum_reward_b = 0
            self.sum_reward_distance = 0
            self.sum_reward = 0
            self.positions = self.breathingSignal_validation
            self._signal_position = self.positions[int(self.n_validation % self.n_test_episode)]
            self._tumor_position = self._signal_position[self.curTimeStep]
            self.noisy_signal_position = self._signal_position #+ create_noisy_signal(self.moving, self.SignalLength, self.frequency)
            self.noisy_tumor_position = self.noisy_signal_position[self.curTimeStep]
            # self.all_energies = np.zeros(self.SignalLength, dtype=int)
            print('validation mode : ' + str(self.count_validation), self._tumor_position)
            self.count_validation += 1
            self.n_validation += 1
        # self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        self.perfect_DM = self.observeEnvAs2DImage(pos = np.round(self._tumor_position), dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        sigma = 1.5
        # self.PTVag = self.observeEnvAs2DImage_plain(pos = self._tumor_position, dose_q=1, targetSize = self.targetSize+self.zone, form = self.form)
        # self.perfect_DM = gaussian_filter(self.perfect_DM, sigma)#*self.PTVag
        # self.noisy_perfect_DM = gaussian_filter(self.noisy_perfect_DM, sigma)
        self.n_dose = np.sum(self.perfect_DM)
        self._targets_position = np.argwhere(self.perfect_DM!=0)#/self.envSize
        self.n_targets = len(self._targets_position)
        # already_shoot = [0,0.5,1]
        # dose_already_shoot = already_shoot[random.randint(0, 2)]
        ref_position = [int(self.depth / 2),int(self.num_row / 2),int(self.num_col / 2)]
        # self.DM_i = self.observeEnvAs2DImage_plain(pos = self._tumor_position, dose_q=dose_already_shoot, targetSize = self.targetSize, form = self.form)
        # self.DM_i_noisy = self.observeEnvAs2DImage_plain(pos = self.noisy_tumor_position, dose_q=dose_already_shoot, targetSize = self.targetSize, form = self.form)
        # self.DMi_inRef = self.observeEnvAs2DImage_plain(pos = ref_position, dose_q=dose_already_shoot, targetSize = self.targetSize, form = self.form)
        # self.DMi_inRef_noisy = self.observeEnvAs2DImage_plain(pos = ref_position, dose_q=dose_already_shoot, targetSize = self.targetSize, form = self.form)
        # self.AllDMi_inRef = np.zeros((self.SignalLength, self.depth,self.num_row,self.num_col))
        self.DM_i = np.zeros((self.depth,self.num_row,self.num_col))
        # self.DM_i_noisy = np.zeros((self.depth,self.num_row,self.num_col))
        self.DMi_inRef = np.zeros((self.depth,self.num_row,self.num_col))
        self.DMi_inRef_noisy = np.zeros((self.depth,self.num_row,self.num_col))
        self.perfectDM_inref = self.observeEnvAs2DImage(pos = ref_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        #self.perfectDM_inref = gaussian_filter(self.perfectDM_inref, sigma)*self.mask
        self._beam_dx= self.targetSize + 1
        self._beam_dy= self.targetSize + 1
        self._beam_position = np.array((int(self.num_row / 2), int(self.num_col / 2)))
        self.beam_pos = np.zeros((self.num_row,self.num_col), dtype=np.float64)
        self.beam_pos[self._beam_position[0],self._beam_position[1]] = 1
        #self._last_ponctual_observation = self.perfect_DM - self.DM_i
        self.mask_shifted = self.observeEnvAs2DImage_plain(pos = self._tumor_position, dose_q=1, targetSize = self.targetSize+self.zone-1, form = self.form)
        self.energy = int(self.depth/2) + self.targetSize
        if self.n_obs == 2 :
            self.doseMaps[0] = 5*(self.perfectDM_inref[self.energy]-self.DMi_inRef_noisy[self.energy])/self.SignalLength
            self.doseMaps[1] = self.mask_shifted[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size), "beam position" : self._beam_position.reshape([2,1])/self.num_col})#, "incert": self.mask.reshape([1,num_row, num_col])})
        # if self.n_obs == 3 :
        #     #self.doseMaps[0] = 5*(self.perfectDM_inref-self.DMi_inRef_noisy)/self.SignalLength
        #     self.doseMaps[0] = 5*(self.noisy_perfect_DM[self.energy] - self.DM_i[self.energy])/self.SignalLength
        #     self.doseMaps[1] = self.beam_pos
        #     self.doseMaps[2] = self.mask_shifted[self.energy]
        #     self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})
        if self.n_obs == 4 :
            self.doseMaps[:] = self.DMi_inRef_noisy[self.energy]/4#self.ref_position[0]-self.targetSize:self.ref_position[0]+self.targetSize+1]#[self.energy]#100*(self.perfectDM_inref[self.energy]-self.DMi_inRef_noisy[self.energy])/self.SignalLength
            #self.doseMaps[0] = 5*(self.noisy_perfect_DM - self.DM_i)/self.SignalLength
            #self.doseMaps[1] = self.beam_pos
            #self.doseMaps[2] = self.mask_shifted
            self.incertitude[0] = self.mask_shifted[self.energy]
            self.incertitude[1] = self.beam_pos
            self.observation.append({"doseMaps": self.doseMaps.reshape([self.dose_size, self.num_row, self.num_col]), "incert": self.incertitude.reshape(self.img_size)})
        #self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})#, "incert": self.mask.reshape([1,self.num_row, self.num_col])})
        #self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size), "incert": self.mask_shifted.reshape([1,self.num_row, self.num_col])})
        return self.observation[0]  # reward, done, info can't be included

    def render(self):
        from mpl_toolkits.mplot3d import Axes3D


        fig = plt.figure(figsize=(10,10))
        XX = np.arange(self.num_row)
        YY = np.arange(self.num_col)
        
        def update(i):
                fig.clear()
                ax1 = fig.add_subplot(3, 2, 1)
                maxDose = self.dose_quantity + 2
                shw0 = ax1.imshow(self.allDoseMap[i][self.all_energies[i]], cmap='jet', vmin=0, vmax=maxDose)
                shw1 = ax1.imshow(self.allPerfect_DM[i][self.all_energies[i]], cmap='gray')
                shw2 = ax1.imshow(self.allDoseMap[i][self.all_energies[i]], cmap='jet', alpha=0.4*(self.allDoseMap[i][self.all_energies[i]] > 0), vmin=0, vmax=maxDose)

                plt.colorbar(shw0, shrink=0.7)

                # show plot with labels
                plt.scatter(self.valid_beam_pos[i,1],self.valid_beam_pos[i,0], color = "g", marker = "x")
                plt.title(str(self.n_validation)+"th validation episode, energy : " + str(self.all_energies[i]))

                fig.add_subplot(3,2,2)
                if (i == self.SignalLength - 1):
                    plt.plot(self.allCumReward_b, '.--', c='orange', label="doseMap")
                    plt.plot(self.allCumReward_distance, '.--', c='g', label="distance")
                    plt.plot(self.allCumReward, '.--', c='k', label='total reward')
                else :
                    plt.plot(self.allCumReward_b[:i+1], '.--', c='orange', label="doseMap")
                    plt.plot(self.allCumReward_distance[:i+1], '.--', c='g', label="distance")
                    plt.plot(self.allCumReward[:i+1], '.--', c='k', label='total reward')
                plt.xlabel("Time")
                plt.ylabel("Cumulative reward")
                plt.xlim([0, self.SignalLength])
                plt.ylim([min(self.allCumReward_b), max(self.allCumReward)])
                plt.legend()
                plt.title("Cum. reward as the episode progresses, R : "+ str(np.round(self.allReward[i],3)))
                
                ax4 = fig.add_subplot(3, 2, 3)
                plt.title("Noisy observation")
                shw0 = ax4.imshow(self.allDoseMap[i][self.all_energies[i]], cmap='jet', vmin=0, vmax=maxDose)
                shw1 = ax4.imshow(self.all_noisy_Perfect_DM[i][self.all_energies[i]], cmap='gray')
                shw2 = ax4.imshow(self.allDoseMap[i][self.all_energies[i]], cmap='jet', alpha=0.4*(self.allDoseMap[i][self.all_energies[i]] > 0), vmin=0, vmax=maxDose)

                plt.colorbar(shw0, shrink=0.7)

                # show plot with labels
                plt.scatter(self.valid_beam_pos[i,1],self.valid_beam_pos[i,0], color = "g", marker = "x")
                
                ax2 = fig.add_subplot(3,2,4, projection='3d')
                #shw1 = ax2.voxels(self.PTV, cmap='gray')
                shw2 = ax2.voxels(self.AllDMi_inRef[i], cmap='jet')#, vmin=0, vmax=maxDose, interpolation = "gaussian", alpha=0.4*(self.AllDMi_inRef[i] > 0),facecolors='jet',edgecolor='k')
                #plt.colorbar(shw2, shrink=0.7)
                ax2.set_xlim3d([self.ref_position[0]-self.targetSize-4, self.ref_position[0]+self.targetSize+3])
                ax2.set_xlabel('X')

                ax2.set_ylim3d([self.ref_position[1]-self.targetSize-4, self.ref_position[1]+self.targetSize+3])
                ax2.set_ylabel('Y')

                ax2.set_zlim3d([self.ref_position[2]-self.targetSize-4, self.ref_position[2]+self.targetSize+3])
                ax2.set_zlabel('Z')

                #ax2.set_title("Accum. doseMap in a reference position")
                ax = fig.add_subplot(3,2,5, projection='3d')
                shw0 = ax.voxels(-(self.perfectDM_inref-self.AllDMi_inRef[i]), cmap='bwr')#,vmin = -2, vmax=2)#, interpolation = "gaussian")
                #plt.colorbar(shw0, shrink=0.7)
                ax.set_title("Diff. between dose objective and rt doseMap")

                # Setting the axes properties
                ax.set_xlim3d([self.ref_position[0]-self.targetSize-6, self.ref_position[0]+self.targetSize+5])
                ax.set_xlabel('X')

                ax.set_ylim3d([self.ref_position[1]-self.targetSize-6, self.ref_position[1]+self.targetSize+5])
                ax.set_ylabel('Y')

                ax.set_zlim3d([self.ref_position[2]-self.targetSize-6, self.ref_position[2]+self.targetSize+5])
                ax.set_zlabel('Z')

                ax6 = fig.add_subplot(3,2,6)
                bin_edges,dvh, bin_edges_interpolate = compute_DVH(self.AllDMi_inRef[i], perfectDM_PTV_inref, self.dose_quantity, 20)
                spl = CubicSpline(bin_edges, dvh)
                ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTV")
                bin_edges,dvh, bin_edges_interpolate = compute_DVH(self.AllDMi_inRef[i], perfectDM_GTV_inref, self.dose_quantity, 20)
                spl = CubicSpline(bin_edges, dvh)
                ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="GTV")
                bin_edges,dvh, bin_edges_interpolate = compute_DVH(self.AllDMi_inRef[i], perfectDM_AroundTumor_inref, self.dose_quantity, 20)
                spl = CubicSpline(bin_edges, dvh)
                ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="Around Tumor")
                bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(self.AllDMi_inRef[i], perfectDM_inref, self.dose_quantity, 20)
                spl = CubicSpline(bin_edges, dvh)
                ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="OAR")
                ax6.axvline(self.dose_quantity, linestyle='dashed', label="Dose objective")
                ax6.set_xlabel("Dose deposited (Gy)")
                ax6.set_ylabel("Pourcentage of volume")
                ax6.set_title("DVH of PTV and OAR")
                plt.ylim([-5, 105])
                ax6.legend()

                
        ref_position = [int(self.depth / 2),int(self.num_row / 2), int(self.num_col / 2)]
        perfectDM_inref = self.observeEnvAs2DImage_plain(pos = ref_position, dose_q=self.dose_quantity, targetSize = self.targetSize+2, form = self.form)
        perfectDM_PTV_inref = self.observeEnvAs2DImage_plain(pos = ref_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        perfectDM_GTV_inref = self.observeEnvAs2DImage_plain(pos = ref_position, dose_q=self.dose_quantity, targetSize = self.targetSize-1, form = self.form)
        perfectDM_AroundTumor_inref = perfectDM_inref-perfectDM_PTV_inref
        #self.valid_beam_pos = self.valid_beam_pos[~np.all(self.valid_beam_pos == 0, axis=1)]

        # LEN = self.SignalLength
        # anim = FuncAnimation(fig, update, frames = LEN, interval=100)
        # anim.save(self.saving_path + "/" + self.name + "/epoch" +str((self.n_validation//self.n_test_episode))+'.gif') 

        # 23_05_100_LR0.00007+10_[32, 16]bs8
        # def gen(n):
        #     phi = 0
        #     while phi < 2*np.pi:
        #         yield np.array([np.cos(phi), np.sin(phi), phi])
        #         phi += 2*np.pi/n

        # def update(num, data, line):
        #     line.set_data(data[:2, :num])
        #     line.set_3d_properties(data[2, :num])

        N = self.SignalLength
        # data = np.array(list(gen(N))).T
        # line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

        
        anim = FuncAnimation(fig, update, frames = N, interval=100, blit=False) #fargs=(data, line)
        anim.save(self.saving_path + "/" + self.name + "/epoch" +str((self.n_validation//self.n_test_episode))+".gif")#, writer='imagemagick') 
        plt.close()

    def get_Signal_length(self,) :
        return self.SignalLength
    
    def observeEnvAs2DImage(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
        """
        The noise thing is a work in progress
        """

        envImg1 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
        
        if self.form == "cube":
            target = (dose_q/2)*cube(2*(targetSize+2) + 1) #square(2*self.targetSize + 1) #disk(self.targetSize) #[1, 1]#
        if self.form == "ball":
            target = (dose_q/2)*ball(targetSize+2)
        if pos is not None:
            targetCenter = ref_position
        # else:
        #     targetCenter = np.array(self.breathingSignal.breathingSignal[self.curTimeStep, 0], self.breathingSignal.breathingSignal[self.curTimeStep, 1])
        targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
        # envImg1[targetCenterInPixels[0] - int(targetSize + 2): targetCenterInPixels[0] + int(targetSize + 3),
        #         targetCenterInPixels[1] - int(targetSize + 2): targetCenterInPixels[1] + int(targetSize + 3), 
        #         targetCenterInPixels[2] - int(targetSize + 2): targetCenterInPixels[2] + int(targetSize + 3)] = target
        # envImg[targetCenterInPixels[0] ,
        #         targetCenterInPixels[1] : targetCenterInPixels[1] + 2] = target
        
        envImg2 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        if self.form == "cube":
            target = (dose_q)*cube(2*(targetSize+1) + 1) #square(2*self.targetSize + 1) #disk(self.targetSize) #[1, 1]#
        if self.form == "ball":
            target = (dose_q)*ball(targetSize+1)
        if pos is not None:
            targetCenter = ref_position
        # else:
        #     targetCenter = np.array(self.breathingSignal.breathingSignal[self.curTimeStep, 0], self.breathingSignal.breathingSignal[self.curTimeStep, 1])
        envImg2[targetCenterInPixels[0] - int(targetSize+1): targetCenterInPixels[0] + int(targetSize + 2),
                targetCenterInPixels[1] - int(targetSize+1): targetCenterInPixels[1] + int(targetSize + 2),
                targetCenterInPixels[2] - int(targetSize+1): targetCenterInPixels[2] + int(targetSize + 2)] = target
        
        
        envImg = envImg1+envImg2 #+ envImg3
        shift = np.zeros(3)
        shift[0] = pos[0] - targetCenterInPixels[0]
        shift[1] = pos[1] - targetCenterInPixels[1]
        shift[2] = pos[2] - targetCenterInPixels[2]
        envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
        
        return envImg_shifted
            
    def observeEnvAs2DImage_plain(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
        """
        The noise thing is a work in progress
        """
        envImg = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
        
        if self.form == "cube":
            target = (dose_q)*cube(2*targetSize + 1) #square(2*self.targetSize + 1) #disk(self.targetSize) #[1, 1]#
        if self.form == "ball":
            target = (dose_q)*ball(targetSize)
        if pos is not None:
            targetCenter = ref_position
        # else:
        #     targetCenter = np.array(self.breathingSignal.breathingSignal[self.curTimeStep, 0], self.breathingSignal.breathingSignal[self.curTimeStep, 1])
        targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
        envImg[targetCenterInPixels[0] - int(targetSize): targetCenterInPixels[0] + int(targetSize + 1),
                targetCenterInPixels[1] - int(targetSize): targetCenterInPixels[1] + int(targetSize + 1),
                targetCenterInPixels[2] - int(targetSize): targetCenterInPixels[2] + int(targetSize + 1)] = target
        # envImg[targetCenterInPixels[0] ,
        #         targetCenterInPixels[1] : targetCenterInPixels[1] + 2] = target
        shift = np.zeros(3)
        shift[0] = pos[0] - targetCenterInPixels[0]
        shift[1] = pos[1] - targetCenterInPixels[1]
        shift[2] = pos[2] - targetCenterInPixels[2]
        envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
        
        return envImg_shifted

    
    def gkern(self, long, sig):
        """
        creates gaussian kernel with side length `long` and a sigma of `sig`
        """
        ax = np.linspace(-(long - 1) / 2., (long - 1) / 2., long)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    
    def weighted_abs_value(self,x,a):
        positive_error = a*np.sum(x[x>=0])
        negative_error = -1*np.sum(x[x<0])
        return positive_error + negative_error
    
    def weighted_squared_abs_value(self,x,a):
        b = np.where(x>=0, a*(x**2), x**2)
        return np.sum(b)
    
    def weighted_zone(self, x, autour_tumeur, perfect_DM, perfectDM_ag, a):
        sum_autour = x[autour_tumeur != 0]
        sum_autours = -1*np.sum(sum_autour[sum_autour<-0.5])
        sum_PTV = x[perfect_DM != 0]
        sum_PTVS = a*np.sum(sum_PTV[sum_PTV>=0]) - np.sum(sum_PTV[sum_PTV<0])
        sum_OAR = -a*np.sum(x[perfectDM_ag == 0])
        return sum_autours + sum_PTVS + sum_OAR