import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.collections import LineCollection
from dose_evaluation import *
from scipy.ndimage import gaussian_filter


def evaluate(model, env, n_eval_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    #tumor_recovery = []
    time = np.zeros(n_eval_episodes)
    DM = np.zeros(n_eval_episodes)
    DM_PTV = np.zeros(n_eval_episodes)
    d_meanOAR = np.zeros(n_eval_episodes)
    d_meanAutour = np.zeros(n_eval_episodes)
    d_meanPTV= np.zeros(n_eval_episodes)
    D98_PTV = np.zeros(n_eval_episodes)
    D80_PTV = np.zeros(n_eval_episodes)
    D30_OAR = np.zeros(n_eval_episodes)
    minim = np.zeros(n_eval_episodes)
    #all_tumor_recovery = []
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            #tumor_recovery.append(info)
            episode_rewards.append(reward)
        
        time[i] = env.curTimeStep
        
        ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
        perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+2, form = env.form)
        #perfectDM_PTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
        perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
        d_OAR = env.DMi_inRef[perfectDM_inref_eval!=2]
        d_PTV = env.DMi_inRef[env.PTV==1]
        d_GTV = env.DMi_inRef[perfectDM_GTV_inref==2]
        AutourPTV = env.mask - env.PTV
        d_Autour = env.DMi_inRef[AutourPTV!=0]

        d_meanOAR[i] = np.mean(d_OAR)
        d_meanAutour[i] = np.mean(d_Autour)
        d_meanPTV[i] = np.mean(d_PTV)
        D98_PTV[i] = computeDx(d_PTV, 98, 4)
        D80_PTV[i] = computeDx(d_PTV, 80, 4)
        D30_OAR[i] = computeDx(d_OAR,30, 4)
        diff = env.perfectDM_inref-env.DMi_inRef
        DM[i] = np.sum(np.abs(diff))
        DM_PTV[i] = np.sum(np.abs(diff[env.PTV==1]))
        all_episode_rewards.append(sum(episode_rewards))
        minim[i] = np.min(d_PTV)
        #all_tumor_recovery.append(tumor_recovery[-1])
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    #print("Mean reward:", mean_episode_reward, "Num episodes:", n_eval_episodes)

    return mean_episode_reward, std, np.mean(time), np.mean(DM), np.mean(DM_PTV), np.mean(d_meanPTV), np.mean(d_meanOAR), np.mean(d_meanAutour), np.mean(D98_PTV), np.mean(D80_PTV), np.mean(D30_OAR), np.mean(minim)

def computeDx(d, percentile, maxDVH):
    number_of_bins = 4096
    DVH_interval = [0, maxDVH]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1] + 0.5 * bin_size, bin_size)
    bin_edges[-1] = maxDVH + d.max()
    _dose = bin_edges[:number_of_bins] + 0.5 * bin_size
    h, _ = np.histogram(d, bin_edges)
    h = np.flip(h, 0)
    h = np.cumsum(h)
    h = np.flip(h, 0)
    _volume = h * 100 / len(d)  # volume in %
    index = np.searchsorted(-_volume, -percentile)
    if (index > len(_volume) - 2): 
        index = len(_volume) - 2
    volume = _volume[index]
    volume2 = _volume[index + 1]
    if (volume == volume2):
            Dx = _dose[index]
    else:
            w2 = (volume - percentile) / (volume - volume2)
            w1 = (percentile - volume2) / (volume - volume2)
            Dx = w1 * _dose[index] + w2 * _dose[index + 1]
            if Dx < 0: Dx = 0
    return Dx

def evaluate_over_treatment(model, env, n_eval_episodes=30, deterministic=True, epoch=0):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    #tumor_recovery = []
    time = np.zeros(n_eval_episodes)
    DM = np.zeros(n_eval_episodes)
    Accum_DM = np.zeros((env.num_row, env.num_col))
    d_meanOAR = np.zeros(n_eval_episodes)
    d_meanAutour = np.zeros(n_eval_episodes)
    d_meanPTV= np.zeros(n_eval_episodes)
    D98_PTV = np.zeros(n_eval_episodes)
    D80_PTV = np.zeros(n_eval_episodes)
    D30_OAR = np.zeros(n_eval_episodes)
    #all_tumor_recovery = []
    ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+2, form = env.form)
    #perfectDM_PTV_inref = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            #tumor_recovery.append(info)
            episode_rewards.append(reward)
        
        time[i] = env.curTimeStep
        Accum_DM += env.DMi_inRef
        ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
        d_OAR = env.DMi_inRef[perfectDM_inref_eval!=2]
        d_PTV = env.DMi_inRef[env.PTV==1]
        AutourPTV = env.mask - env.PTV
        d_Autour = env.DMi_inRef[AutourPTV!=0]

        d_meanOAR[i] = np.mean(d_OAR)
        d_meanAutour[i] = np.mean(d_Autour)
        d_meanPTV[i] = np.mean(d_PTV)
        D98_PTV[i] = computeDx(d_PTV, 98, 4)
        D80_PTV[i] = computeDx(d_PTV, 80, 4)
        D30_OAR[i] = computeDx(d_OAR,30, 4)
        diff = env.perfectDM_inref-env.DMi_inRef
        DM[i] = np.sum(np.abs(diff))
        all_episode_rewards.append(sum(episode_rewards))
        #all_tumor_recovery.append(tumor_recovery[-1])
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    #print("Mean reward:", mean_episode_reward, "Num episodes:", n_eval_episodes)
    d_PTV_all = Accum_DM[env.PTV==1]
    d30_OAR_all = Accum_DM[perfectDM_inref_eval==0]
    fig = plt.figure()
    plt.suptitle("D98_PTV : " + str(np.round(computeDx(d_PTV_all, 98, 90),2)) +"D80_PTV : " + str(np.round(computeDx(d_PTV_all, 80, 90),2)) + "   D30_OAR : " + str(np.round(computeDx(d30_OAR_all, 30, 90),2)) + "    d_meanOAR : " + str(np.round(np.mean(d30_OAR_all),2)))
    ax5 = plt.subplot(221)
    shw0 = ax5.imshow(-((n_eval_episodes*env.perfectDM_inref)-Accum_DM), cmap='bwr',vmin = -25, vmax=25, interpolation = "gaussian")
    plt.colorbar(shw0, shrink=0.7)
    plot_outlines(env.PTV.T, ax=ax5, lw=1.4, color='yellowgreen')
    ax5.set_title("(A) Difference between \n the desired and the actual doseMap")

    ax2 = plt.subplot(222)
    shw1 = ax2.imshow(env.PTV, cmap='gray')
    maxDose = n_eval_episodes*env.dose_quantity
    shw2 = ax2.imshow(Accum_DM, cmap='jet', alpha=0.4*(Accum_DM > 0), vmin=0, vmax=80, interpolation = "gaussian")
    plt.colorbar(shw2, shrink=0.7, ticks=[0,30,60])
    ax2.set_title("(B) Accumulated dose")

    ax6 = plt.subplot(212)
    perfectDM_PTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2, targetSize = env.targetSize, form = env.form)
    perfectDM_AroundTumor_inref = perfectDM_inref_eval-perfectDM_PTV_inref
    perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_PTV_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTV", color="yellowgreen")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_GTV_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="GTV", color="orange")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_AroundTumor_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTVs", color="steelblue")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(Accum_DM, perfectDM_inref_eval, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="OAR", color="firebrick")
    ax6.axvline(n_eval_episodes*2, linestyle='dashed', label="Dose obj.")
    ax6.set_xlabel("Dose deposited (Gy)")
    ax6.set_ylabel("Pourcentage of surface (%)")
    ax6.set_title("(C) Dose Surface Histogram of the different zones")
    plt.ylim([-5, 105])
    ax6.legend(loc='upper right', labelspacing=0.15)
    fig.tight_layout()
    plt.savefig(env.saving_path + "/" + env.name + "/final_epoch_all_treat" +str(epoch)+".png")
    plt.close

    return mean_episode_reward, std, np.mean(time), np.mean(DM), np.mean(d_meanPTV), np.mean(d_meanOAR), np.mean(d_meanAutour), np.mean(D98_PTV), np.mean(D80_PTV), np.mean(D30_OAR)

def evaluate_over_treatment_daily(model, env, n_eval_episodes=30, deterministic=True, epoch=0):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    #tumor_recovery = []
    time = np.zeros(n_eval_episodes)
    DM = np.zeros(n_eval_episodes)
    Accum_DM = np.zeros((env.num_row, env.num_col))
    d_meanOAR = np.zeros(n_eval_episodes)
    d_meanAutour = np.zeros(n_eval_episodes)
    d_meanPTV= np.zeros(n_eval_episodes)
    D98_PTV = np.zeros(n_eval_episodes)
    D80_PTV = np.zeros(n_eval_episodes)
    D30_OAR = np.zeros(n_eval_episodes)

    DM_PTV = np.zeros(n_eval_episodes)
    minim = np.zeros(n_eval_episodes)
    #all_tumor_recovery = []
    ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+env.zone, form = env.form)
    #perfectDM_PTV_inref = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            #tumor_recovery.append(info)
            episode_rewards.append(reward)
        
        time[i] = env.curTimeStep
        Accum_DM += env.DMi_inRef
        ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
        d_OAR = env.DMi_inRef[perfectDM_inref_eval!=2]
        d_PTV = env.DMi_inRef[env.PTV==1]
        AutourPTV = env.mask - env.PTV
        d_Autour = env.DMi_inRef[AutourPTV!=0]

        d_meanOAR[i] = np.mean(d_OAR)
        d_meanAutour[i] = np.mean(d_Autour)
        d_meanPTV[i] = np.mean(d_PTV)
        D98_PTV[i] = computeDx(d_PTV, 98, 4)
        D80_PTV[i] = computeDx(d_PTV, 80, 4)
        D30_OAR[i] = computeDx(d_OAR,30, 4)
        diff = env.perfectDM_inref-env.DMi_inRef
        DM[i] = np.sum(np.abs(diff))
        DM_PTV[i] = np.sum(np.abs(diff[env.PTV==1]))
        all_episode_rewards.append(sum(episode_rewards))
        minim[i] = np.min(d_PTV)#all_tumor_recovery.append(tumor_recovery[-1])
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    perfect_q = n_eval_episodes*2
    #print("Mean reward:", mean_episode_reward, "Num episodes:", n_eval_episodes)
    d_PTV_all = Accum_DM[env.PTV==1]
    d30_OAR_all = Accum_DM[perfectDM_inref_eval==0]
    D98_PTV_treat = computeDx(d_PTV_all, 98, perfect_q+30)
    D80_PTV_treat = computeDx(d_PTV_all, 80, perfect_q+30)
    D30_OAR_treat = computeDx(d30_OAR_all, 30, perfect_q+30)
    mean_OAR_treat = np.mean(d30_OAR_all)

    return mean_episode_reward, std, np.mean(time), np.mean(DM), np.mean(DM_PTV), np.mean(d_meanPTV), np.mean(d_meanOAR), np.mean(d_meanAutour), np.mean(D98_PTV), np.mean(D80_PTV), np.mean(D30_OAR), np.mean(minim), D98_PTV_treat, D80_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM

def plot_treatment(env, n_eval_episodes, D98_PTV_treat, D80_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, epoch):
    ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    perfectDM_inref_eval = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    #perfectDM_inref_eval = gaussian_filter(perfectDM_inref_eval, 1.5)*env.mask
    fig = plt.figure()
    plt.suptitle("D98_PTV : " + str(np.round(D98_PTV_treat,2)) +"D80_PTV : " + str(np.round(D80_PTV_treat,2)) + "   D30_OAR : " + str(np.round(D30_OAR_treat,2)) + "    d_meanOAR : " + str(np.round(mean_OAR_treat,2)))
    ax5 = plt.subplot(231)
    shw0 = ax5.imshow(-((n_eval_episodes*env.perfectDM_inref)-Accum_DM), cmap='bwr',vmin = -25, vmax=25, interpolation = "gaussian")
    plot_outlines(env.PTV.T, ax=ax5, lw=1.4, color='yellowgreen')
    ax5.set_title("(A) DoseMaps Difference")

    ax2 = plt.subplot(232)
    shw0 = ax2.imshow(-((n_eval_episodes*2*env.PTV)-Accum_DM), cmap='bwr',vmin = -25, vmax=25, interpolation = "gaussian")
    plt.colorbar(shw0, shrink=0.7)
    plot_outlines(env.PTV.T, ax=ax2, lw=1.4, color='yellowgreen')
    ax2.set_title("(B) Medical objective")
    perfect_q = n_eval_episodes*2
    ax2 = plt.subplot(233)
    shw1 = ax2.imshow(env.PTV, cmap='gray')
    maxDose = n_eval_episodes*env.dose_quantity
    shw2 = ax2.imshow(Accum_DM, cmap='jet', alpha=0.4*(Accum_DM > 0), vmin=0, vmax=perfect_q+20, interpolation = "gaussian")
    plt.colorbar(shw2, shrink=0.7, ticks=[0,perfect_q/2,perfect_q])
    ax2.set_title("(C) Accumulated dose")

    ax6 = plt.subplot(212)
    perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+env.zone, form = env.form)
    perfectDM_PTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2, targetSize = env.targetSize, form = env.form)
    perfectDM_AroundTumor_inref = perfectDM_inref_eval-perfectDM_PTV_inref
    perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_PTV_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTV", color="yellowgreen")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_GTV_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="GTV", color="orange")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_AroundTumor_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTVs", color="steelblue")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(Accum_DM, perfectDM_inref_eval, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="OAR", color="firebrick")
    ax6.axvline(n_eval_episodes*2, linestyle='dashed', label="Dose obj.")
    ax6.set_xlabel("Dose deposited (Gy)")
    ax6.set_ylabel("Pourcentage of surface (%)")
    ax6.set_title("(D) Dose Surface Histogram of the different zones")
    plt.ylim([-5, 105])
    ax6.legend(loc='upper right', labelspacing=0.15)
    fig.tight_layout()
    plt.savefig(env.saving_path + "/" + env.name + "/final_epoch_all_treat" +str(epoch)+".png")
    plt.close
    return 0

def evaluate_over_treatment_with_updates(model, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    #tumor_recovery = []
    time = np.zeros(n_eval_episodes)
    DM = np.zeros(n_eval_episodes)
    Accum_DM = np.zeros((env.num_row, env.num_col))
    d_meanOAR = np.zeros(n_eval_episodes)
    d_meanAutour = np.zeros(n_eval_episodes)
    d_meanPTV= np.zeros(n_eval_episodes)
    D98_PTV = np.zeros(n_eval_episodes)
    D30_OAR = np.zeros(n_eval_episodes)
    #all_tumor_recovery = []
    ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    perfectDM_inref_eval = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    
#PREMIER EPISODE
    done = False
    obs = env.reset()
    # print(env.noisy_perfect_DM[env.noisy_perfect_DM!=0])
    # print(env.perfect_DM[env.perfect_DM!=0])
    while not done:
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
        obs, reward, done, info = env.step(action)
            #tumor_recovery.append(info)        
    # time[i] = env.curTimeStep
    Accum_DM += env.DMi_inRef
    # print(env.noisy_perfect_DM[env.noisy_perfect_DM!=0])
    # print(env.perfect_DM[env.perfect_DM!=0])

    for i in range(1,n_eval_episodes):
        expected_dose = env.observeEnvAs2DImage(pos = ref_position, dose_q=(i+1)*env.dose_quantity, targetSize = env.targetSize, form = env.form)
        update = expected_dose - Accum_DM
        episode_rewards = []
        done = False
        obs = env.reset_update(update)
        # print(env.noisy_perfect_DM[env.noisy_perfect_DM!=0])
        # print(env.perfect_DM[env.perfect_DM!=0])
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step_update(action, update)
            #tumor_recovery.append(info)
            episode_rewards.append(reward)
        # print(env.noisy_perfect_DM[env.noisy_perfect_DM!=0])
        # print(env.perfect_DM[env.perfect_DM!=0])
        time[i] = env.curTimeStep
        Accum_DM += env.DMi_inRef
        ref_position = [int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
        perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+2, form = env.form)
        perfectDM_PTV_inref = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
        d_OAR = env.DMi_inRef[perfectDM_inref_eval!=2]
        d_PTV = env.DMi_inRef[perfectDM_PTV_inref==2]
        AutourPTV = perfectDM_inref_eval - perfectDM_PTV_inref
        d_Autour = env.DMi_inRef[AutourPTV!=0]

        d_meanOAR[i] = np.mean(d_OAR)
        d_meanAutour[i] = np.mean(d_Autour)
        d_meanPTV[i] = np.mean(d_PTV)
        D98_PTV[i] = computeDx(d_PTV, 98, 4)
        D30_OAR[i] = computeDx(d_OAR,30, 4)
        DM[i] = np.sum(np.abs(perfectDM_PTV_inref-env.DMi_inRef))
        all_episode_rewards.append(sum(episode_rewards))
        #all_tumor_recovery.append(tumor_recovery[-1])
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    #print("Mean reward:", mean_episode_reward, "Num episodes:", n_eval_episodes)

    d_PTV_all = Accum_DM[perfectDM_PTV_inref==2]
    plt.figure()
    plt.title("D98 : " + str(computeDx(d_PTV_all, 98, 90)))
    ax5 = plt.subplot(221)
    shw0 = ax5.imshow(-((n_eval_episodes*perfectDM_PTV_inref)-Accum_DM), cmap='bwr',vmin = -n_eval_episodes*env.dose_quantity, vmax=env.dose_quantity*n_eval_episodes, interpolation = "gaussian")
    plt.colorbar(shw0, shrink=0.7)

    ax2 = plt.subplot(222)
    shw1 = ax2.imshow(env.mask, cmap='gray')
    maxDose = n_eval_episodes*env.dose_quantity
    shw2 = ax2.imshow(Accum_DM, cmap='jet', alpha=0.4*(Accum_DM > 0), vmin=0, vmax=maxDose+5, interpolation = "gaussian")
    plt.colorbar(shw2, shrink=0.7)

    ax6 = plt.subplot(212)
    perfectDM_PTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2, targetSize = env.targetSize, form = env.form)
    perfectDM_AroundTumor_inref = perfectDM_inref_eval-perfectDM_PTV_inref
    perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_PTV_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTV")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_GTV_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="GTV")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_AroundTumor_inref, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="Around Tumor")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(Accum_DM, perfectDM_inref_eval, (n_eval_episodes*env.dose_quantity) + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="OAR")
    ax6.axvline(n_eval_episodes*2, linestyle='dashed', label="Dose objective")
    ax6.set_xlabel("Dose deposited (Gy)")
    ax6.set_ylabel("Pourcentage of volume")
    ax6.set_title("DVH of PTV and OAR")
    plt.ylim([-0.05, 1.05])
    ax6.legend(loc = "upper right")
    plt.savefig(env.saving_path + "/" + env.name + "/final_epoch_all_treat_update" +str((env.n_validation//env.n_test_episode))+".png")

    return mean_episode_reward, std, np.mean(time), np.mean(DM), np.mean(d_meanPTV), np.mean(d_meanOAR), np.mean(d_meanAutour), np.mean(D98_PTV), np.mean(D30_OAR)

def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)