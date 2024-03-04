from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
import sys
import torch as th
from ARIES.evaluation import *
from ARIES.network import Custom_CombinedExtractor, Custom_CNN
from ARIES.learning_rate import linear_schedule, stair_schedule
from ARIES.parser import parameters
from stable_baselines3.common.utils import get_linear_fn
import pickle

import os
import logging

import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

import numpy as np
sav = "/linux/meghislain/Results-RL/"

if parameters.dim =="sansdose" :
       from ARIES.env_sans_dose import TumorEnv
       saving_path = sav + "/2D_Sans_dose"
if parameters.dim =="3Dsansdose" :
       from ARIES.env_sans_dose3D import TumorEnv #ARIES.env_3D_inc
       saving_path = sav + "/3D_incert"
if parameters.dim == "2D" :
        from ARIES.Env_incertitude import TumorEnv
        saving_path = sav + "/2D_Results"
elif parameters.dim == "3D" :
       from ARIES.env_3D import TumorEnv
       saving_path = sav + "/3D_Results"

wandb.init(project="2DBilly", entity="super_aries")

# Instantiate the env
name = parameters.date +"_DQ" +str(parameters.dq) +"_tarSize"+str(parameters.ts)+"_SL" + str(parameters.sl)#"12_06_8X8_carre3X3_MOVING_DOSE1_L75"#"32X32_CARRE5X5"#"MOVING_CARRE3X3_K3:3:3"
num_col = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position
num_row = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position

n_value = []
n_cpu_cores = 2
th.set_num_threads(n_cpu_cores)
wandb.run.name = name

env = TumorEnv(n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)
env_eval = TumorEnv(mode=0, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)
env_test = TumorEnv(mode=1, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 64, 32], normalize_images=False, features_extractor_class=Custom_CombinedExtractor)
#model = DQN.load(sav + "/Model/test_moving", env, device="cuda:1")
if parameters.expl_rate != 1.0 :
       model = DQN("MultiInputPolicy", env, verbose=0, learning_rate=parameters.lr, batch_size=16, device=parameters.dev, policy_kwargs=policy_kwargs, exploration_fraction=0.8, exploration_initial_eps=parameters.eps, exploration_final_eps=parameters.eps, target_update_interval=parameters.sl)
else : 
       model = DQN("MultiInputPolicy", env, verbose=0, learning_rate=parameters.lr, batch_size=16, device=parameters.dev, policy_kwargs=policy_kwargs, exploration_fraction=0.8, exploration_initial_eps=parameters.eps, exploration_final_eps=0.3, target_update_interval=parameters.sl)
TIMESTEPS = parameters.sl*parameters.n_train
print(model.policy)
for i in range(parameters.n_epoch):
        print("epoch number : ", str(i))
        # if i > (int(n_epoch/2)):
        #         model.learning_rate = stair_schedule(initial_lr, int(n_epoch/2), i-int(n_epoch/2))
        #stair_schedule(initial_lr, n_epoch, i)

        model.learn(total_timesteps=TIMESTEPS)
        if parameters.dim == "2D" or parameters.dim == "sansdose" :
        # mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini = evaluate(model, env_eval, n_eval_episodes=n_test_episodes)
                mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV = evaluate_over_treatment_daily(model, env_eval, n_eval_episodes=parameters.n_test, epoch=i)
              #   columns = ["action", "beam pos x", "beam pos y", "real tum pos x", "real tum pos y", "noisy tum pos x", "noisy tum pos y"]
        elif parameters.dim == "3D" or parameters.dim == "3Dsansdose":
        # mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini = evaluate(model, env_eval, n_eval_episodes=n_test_episodes)
                mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV = evaluate_over_treatment_daily_3D(model, env_eval, n_eval_episodes=parameters.n_test, epoch=i)
              #   columns = ["action", "beam pos x", "beam pos y", "real tum pos z","real tum pos x", "real tum pos y", "noisy tum pos z", "noisy tum pos x", "noisy tum pos y"]
       #  table = np.concatenate((actions, beam_pos, real_pos, noisy_pos), axis=1)
       #  table = table.astype(str)
       #  tbl = wandb.Table(columns=columns, data=table)
        dic = {"Accumulation Dose": Accum_DM, "actions": actions, "Beam position": beam_pos, 
                   "real tumor position": real_pos, "noisy tumor position": noisy_pos, "target size": parameters.ts, "form" : parameters.form, "reward" : rewards, "Signal Length" : parameters.sl,
                   "D98_PTV" : D98_PTV_treat, "D80_PTV" : D80_PTV_treat, "D30_OAR" : D30_OAR_treat, "d_meanOAR" : mean_OAR_treat, "DMi_inRef_noisy" : DMi_inRef_noisy}
        if (D80_PTV_treat >= 50 and D30_PTV_treat <= 80) or (i%15 == 0):
              with open(saving_path + "/" + name + "/epoch" + str(i) + ".pickle", 'wb') as handle:
                     pickle.dump(dic, handle)
        wandb.log({"mean reward": mean_reward/(5*2*env.number_pixel_to_touch), "standard deviation of the reward": std_reward, "treatment time": time, 
                   "learning rate": model.learning_rate, "Difference of final DM": DM, "Difference on PTV": DM_PTV, "Dose moyenne PTV" : d_meanPTV, 
                   "Dose moyenne OAR":d_meanOAR, "Dose moyenne Autour":d_meanAutour, "D98 PTV":D98_PTV, "D30 OAR":D30_OAR, "minimum PTV" : mini, 
                   "D98_PTV_treat " : D98_PTV_treat, "D80_PTV_treat " : D80_PTV_treat, "D30_PTV_treat " : D30_PTV_treat, "D30_OAR_treat ": D30_OAR_treat, "mean_OAR_treat ": mean_OAR_treat, "exploration rate ": model.exploration_rate})#, "DM_Accum ": Accum_DM}) #, "Actions ": actions, "Real tumor positions ": real_pos, "Noisy tumor positions ": noisy_pos, "Beam position ": beam_pos})
        
        if (D80_PTV_treat >= (0.75*2*parameters.n_test)) and (D80_PTV_treat <=  (1.2*2*parameters.n_test)):
            if mean_OAR_treat <= 1.0 :
                if parameters.dim == "2D" or parameters.dim == "sansdose" :
                       _ = plot_treatment(env_eval,n_eval_episodes=parameters.n_test, D98_PTV_treat=D98_PTV_treat, D80_PTV_treat=D80_PTV_treat, D30_OAR_treat=D30_OAR_treat, mean_OAR_treat=mean_OAR_treat, Accum_DM=Accum_DM, epoch=i)
                elif parameters.dim == "3D" or parameters.dim == "3Dsansdose" :
                       _ = plot_treatment_3D(env_eval,n_eval_episodes=parameters.n_test, D98_PTV_treat=D98_PTV_treat, D80_PTV_treat=D80_PTV_treat, D30_OAR_treat=D30_OAR_treat, mean_OAR_treat=mean_OAR_treat, Accum_DM=Accum_DM, epoch=i)
        if i % 25 == 0 :
                if parameters.dim == "2D" or parameters.dim == "sansdose" :
                       _ = plot_treatment(env_eval,n_eval_episodes=parameters.n_test, D98_PTV_treat=D98_PTV_treat, D80_PTV_treat=D80_PTV_treat, D30_OAR_treat=D30_OAR_treat, mean_OAR_treat=mean_OAR_treat, Accum_DM=Accum_DM, epoch=i)
                elif parameters.dim == "3D" or parameters.dim == "3Dsansdose" :
                       _ = plot_treatment_3D(env_eval,n_eval_episodes=parameters.n_test, D98_PTV_treat=D98_PTV_treat, D80_PTV_treat=D80_PTV_treat, D30_OAR_treat=D30_OAR_treat, mean_OAR_treat=mean_OAR_treat, Accum_DM=Accum_DM, epoch=i)
        
        model.learning_rate = model.learning_rate*parameters.fct
        if parameters.expl_rate != 1.0 :
              model.exploration_initial_eps = parameters.eps*(parameters.expl_rate**(i+1))
              model.exploration_final_eps = parameters.eps*(parameters.expl_rate**(i+1))
              model.exploration_schedule = get_linear_fn(
              model.exploration_initial_eps,
              model.exploration_final_eps,
              model.exploration_fraction, 
              )
        print(D80_PTV_treat,D98_GTV_treat,D98_PTV_treat,D30_PTV_treat,maxPTVs,maxPTV)
          #D98_GTV_treat >= 56
        if (D80_PTV_treat >= 60) and (D98_GTV_treat >= 58) and (D98_PTV_treat >= 50) and (D30_PTV_treat <= 75) and (maxPTVs <= 85) and (maxPTV <= 87):
              for j in range(5):
                     mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat = evaluate_over_treatment_daily_3D(model, env_test, n_eval_episodes=parameters.n_test, epoch=j)
                     dic = {"Accumulation Dose": Accum_DM, "actions": actions, "Beam position": beam_pos, 
                     "real tumor position": real_pos, "noisy tumor position": noisy_pos, "target size": parameters.ts, "form" : parameters.form, "reward" : rewards, "Signal Length" : parameters.sl,"D98_PTV" : D98_PTV_treat, "D80_PTV" : D80_PTV_treat, "D30_OAR" : D30_OAR_treat, "d_meanOAR" : mean_OAR_treat, "DMi_inRef_noisy" : DMi_inRef_noisy}
                     with open(saving_path + "/" + name + "/testingepoch" + str(j) + ".pickle", 'wb') as handle:
                       pickle.dump(dic, handle)
              #model.save(sav + "/Model/" + name)
              break# if i % 50 == 0 :
        #        model.save(sav + "/Model/" + name)

        # if n == patience :
        #         print("Early stopping at epoch ", i)
        #         print(n_value)
        #         if save_gif == True :
        #                 env_eval.render()
        #         model.save(sav + "/Model/" + name)
        #         break
#mean_reward, std_reward, time, DM, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR = evaluate_over_treatment(model, env_eval, n_eval_episodes=30, epoch=i)
wandb.finish()