
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#from opentps.opentps_core.opentps.core.data._dvh import DVH

def compute_DVH(dose, mask, maxDVH, number_of_bins):
    DVH_interval = [0, maxDVH + 2]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1], bin_size)
    # bin_edges = np.zeros(number_of_bins)
    # for i in range(number_of_bins): #tchevytchev
    # bin_edges[i] = (2-2*np.cos((2*i+1)*np.pi/(2*number_of_bins)))
    bin_edges_interpolate = np.arange(DVH_interval[0], DVH_interval[1], bin_size/10)
    d = dose[mask!=0]
    dvh = np.zeros(len(bin_edges))
    for i in range(len(dvh)) :
        dvh[i] = len(d[d>=bin_edges[i]])*100/len(d)

    return bin_edges,dvh, bin_edges_interpolate

def compute_DVH_OAR(dose, mask, maxDVH, number_of_bins):
    DVH_interval = [0, maxDVH + 2]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1], bin_size)
    bin_edges_interpolate = np.arange(DVH_interval[0], DVH_interval[1], bin_size/10)
    d = dose[mask==0]
    # print(d)
    dvh = np.zeros(len(bin_edges))
    for i in range(len(dvh)) :
        dvh[i] = len(d[d>=bin_edges[i]])*100/len(d)

    return bin_edges,dvh, bin_edges_interpolate