import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from ARIES.parser import parameters



class SyntheticBreathingSignal():
    def __init__(self, amplitude=10, breathingPeriod=4, meanNoise=0,
                 varianceNoise=1, samplingPeriod=0.2, simulationTime=100, coeffMin = 0.10, coeffMax = 0.15, meanEvent = 1/60, meanEventApnea=0/120, name="Breathing Signal"):

        self.amplitude = amplitude  # amplitude (mm)
        self.breathingPeriod = breathingPeriod  # periode respiratoire (s)
        self.meanNoise = meanNoise
        self.varianceNoise = varianceNoise
        self.samplingPeriod = samplingPeriod  # periode d echantillonnage
        self.simulationTime = simulationTime  # temps de simulation
        self.coeffMin = coeffMin #coefficient minimal pour le changement d amplitude
        self.coeffMax = coeffMax #coefficient maximal pour le changement d amplitude
        self.meanEvent = meanEvent #nombre moyen d evenements
        self.meanEventApnea = meanEventApnea #nombre moyen d apnees
        self.isNormalized = False


    def generate1DBreathingSignal(self):
        self.timestamps, self.breathingSignal = signalGeneration(self.amplitude, self.breathingPeriod, self.meanNoise, self.varianceNoise, self.samplingPeriod, self.simulationTime, self.coeffMin, self.coeffMax,self.meanEvent, self.meanEventApnea)
        return self.breathingSignal

    def generate2DBreathingSignal(self):
        """
        this can be improved to be a single function with a dimension parameter
        """
        self.timestamps, self.breathingSignal = signal2DGeneration(self.amplitude,self.breathingPeriod, self.meanNoise, self.varianceNoise, self.samplingPeriod, self.simulationTime, self.coeffMin, self.coeffMax,self.meanEvent, self.meanEventApnea)
        return self.breathingSignal

    def generate3DBreathingSignal(self):
        """
                this can be improved to be a single function with a dimension parameter
                """
        self.timestamps, self.breathingSignal = signal3DGeneration(self.amplitude,self.breathingPeriod, self.meanNoise, self.varianceNoise, self.samplingPeriod, self.simulationTime, self.coeffMin, self.coeffMax,self.meanEvent, self.meanEventApnea)
        return self.breathingSignal

    def normalize(self, bound = None):

        if bound is None :
            self.breathingSignal -= np.min(self.breathingSignal)
            self.breathingSignal = self.breathingSignal / np.max(self.breathingSignal)
        else :
            self.breathingSignal = (self.breathingSignal-np.min(self.breathingSignal))/(np.max(self.breathingSignal)-np.min(self.breathingSignal))
            self.breathingSignal = (1-2*bound)*self.breathingSignal + bound
        self.isNormalized = True
def create_signal_position(grid, amplitude, moving, SignalLength, nb) :
    matrix = np.zeros((len(grid), SignalLength, 2))
    number = SignalLength//nb
    vecteur = np.arange(1, number, 2)
    for i in range(len(grid)):
        matrix[i] = np.ones((SignalLength, 2))*grid[i]
        if moving == True :
            for j in range(len(vecteur)):
                matrix[i,int(vecteur[j]*nb):int((vecteur[j]+1)*nb), 1] += amplitude*np.ones(nb)*((-1)**j)
                matrix[i,int(vecteur[j]*nb):int((vecteur[j]+1)*nb), 0] += amplitude*np.ones(nb)*((-1)**j)
    return matrix

def create_noisy_signal(moving, SignalLength, nb) :
    number = SignalLength//nb
    vecteur = np.arange(1, number, 2)
    # for i in range(len(grid)):
    matrix = np.zeros((SignalLength, 2))
    if moving == True :
        for j in range(len(vecteur)):
            matrix[int(vecteur[j]*nb):int((vecteur[j]+1)*nb), 1] += np.random.randint(-1,2)
            matrix[int(vecteur[j]*nb):int((vecteur[j]+1)*nb), 0] += np.random.randint(-1,2)
    return matrix

def create_breathing_signals(grid, amplitude, moving, signalLength):
    matrix = np.zeros((len(grid), signalLength, 2))
    breath_sign = SyntheticBreathingSignal(amplitude=2*amplitude, breathingPeriod=4, meanNoise=1/20,
                 varianceNoise=0, samplingPeriod=parameters.sampperiod, simulationTime=signalLength*parameters.sampperiod, coeffMin = 0.10, coeffMax = 0.15, meanEvent = 0/60, meanEventApnea=0/120, name="Breathing Signal")
    for i in range(len(grid)):
        breath_signal = breath_sign.generate2DBreathingSignal()
        matrix[i] = np.ones((signalLength, 2))*grid[i] + np.round(breath_signal)
    return matrix

def create_breathing_signals_reel(grid, amplitude, moving, signalLength):
    matrix = np.zeros((len(grid), signalLength, 2))
    breath_sign = SyntheticBreathingSignal(amplitude=2*amplitude, breathingPeriod=4, meanNoise=1/20,
                 varianceNoise=0., samplingPeriod=parameters.sampperiod, simulationTime=signalLength*parameters.sampperiod, coeffMin = 0.10, coeffMax = 0.15, meanEvent = 0/60, meanEventApnea=0/120, name="Breathing Signal")
    for i in range(len(grid)):
        breath_signal = breath_sign.generate2DBreathingSignal()
        matrix[i] = np.ones((signalLength, 2))*grid[i] + breath_signal
    return matrix
def events(L,meanDurationEvents,varianceDurationEvents,Tend):
    timestamp = [0]
    U = np.random.uniform(0,1)
    if L == 0:
        return timestamp
    else:
        t1 = -np.log(U)/L
        while t1 <= Tend:
            timeEvents = np.random.normal(meanDurationEvents,varianceDurationEvents)
            timestamp.append(t1)
            t1 += timeEvents
            if t1 <= Tend:
                timestamp.append(t1)
            U = np.random.uniform(0,1)
            t1 += -np.log(U)/L
        return timestamp

#entre deux timestamps successifs, un event est cree
#Un event correspond a une fonction echellon 
def vectorSimulation(coeffMin,coeffMax,amplitude,frequency,timestamps,listOfEvents): 
    t = timestamps      
    y_amplitude = np.zeros(len(t))
    y_frequency = np.zeros(len(t))
    i = 0
    while i < len(listOfEvents):
        if i+2 < len(listOfEvents):
            dA = np.random.uniform(coeffMin,coeffMax)*amplitude #amplitude variation
            df = np.abs(frequency-(1/frequency+np.random.uniform(coeffMin,coeffMax))**-1) #frequency variation
            value_amplitude = np.random.uniform(-dA,dA)
            value_frequency = np.random.uniform(-df,df)
            t1 = listOfEvents[i+1]
            t2 = listOfEvents[i+2]
            y_amplitude[(t>=t1) & (t<=t2)] = value_amplitude
            y_frequency[(t>=t1) & (t<=t2)] = value_frequency
        i+=2
    return y_amplitude,y_frequency

#creation des donnees respiratoires
def signalGeneration(amplitude=10, period=4.0, mean=0, sigma=3, step=0.5, signalDuration=100, coeffMin = 0.10, coeffMax = 0.15, meanEvent = 1/20, meanEventApnea=1/120):
    amp = amplitude
    freq = 1 / period
    timestamps = np.arange(0,signalDuration,step)
    #creation des events
    #s il y a un changement d amplitude, alors il y a un changement de frequence
    meanDurationEvents = 10
    varianceDurationEvents = 5
    meanDurationEventsApnea = 15
    varianceDurationEventsApnea = 5
    listOfEvents = events(meanEvent,meanDurationEvents,varianceDurationEvents,signalDuration)
    listOfEventsApnea = events(meanEventApnea,meanDurationEventsApnea,varianceDurationEventsApnea,signalDuration)
    sigma *= amp/20
    
    y_amplitude, y_frequency = vectorSimulation(coeffMin,coeffMax,amp,freq,timestamps,listOfEvents)
    amplitude += y_amplitude
    freq += y_frequency
    noise = np.random.normal(loc=mean,scale=sigma,size=len(timestamps))
    phi = np.random.uniform(0,2*np.pi)
    signal = (amplitude / 2) * np.sin(2 * np.pi * freq * (timestamps % (1 / freq))+phi) ## we talk about breathing amplitude in mm so its more the total amplitude than the half one, meaning it must be divided by two here
    signal += noise
    
    #pour chaque event, la valeur min de tout le signal doit rester identique, meme s il y a un changement
    #d amplitude 
    i = 0
    while i < len(listOfEvents):
        if i+2 < len(listOfEvents):
            t1 = listOfEvents[i+1]
            t2 = listOfEvents[i+2]
            newAmplitude = amplitude[int(((t1+t2)/2)/step)]
            signal[(timestamps>=t1) & (timestamps<=t2)] += (-amp/2+newAmplitude/2) 
        i+= 2
    
    #pendant une apnea, le signal respiratoire ne varie quasi pas
    timeApnea = []
    i = 0
    while i < len(listOfEventsApnea):
        if i+2 < len(listOfEventsApnea):
            index = np.abs(timestamps - listOfEventsApnea[i+1])
            indexApnea = np.argmin(index)
            a = signal[indexApnea]
            if a < 0 and a < -0.8*amp/2:
                t1 = listOfEventsApnea[i+1]
            else:
                newIndexApnea = indexApnea + np.argmin(signal[indexApnea:int(indexApnea+period//step)]) #+ np.random.randint(-int(period/(2*step)),0)
                t1 = timestamps[newIndexApnea]
                a = signal[newIndexApnea]
                
            t2 = listOfEventsApnea[i+2]
            diff_i = np.argmin(np.abs(timestamps-t2))-np.argmin(np.abs(timestamps-t1))
            timeDec = np.arange(0,t2-t1,step)[0:diff_i]
            noiseApnea = np.random.normal(loc=0,scale=sigma/5,size=len(timeDec))
            signal[np.argmin(np.abs(timestamps-t1)):np.argmin(np.abs(timestamps-t2))] = -timeDec/(t2-t1)+a + noiseApnea
            timeApnea.append(np.argmin(np.abs(timestamps-t2)))
        i+=2
    
    #apres une apnee, le signal a une amplitude plus grande car le patient doit reprendre son souffle
    for timeIndex in timeApnea:
        timeAfterApnea = np.arange(0,np.random.normal(15,5),step)
        cst = np.random.uniform(1.4,2.0)
        ampSig = cst*(amp/2)
        noiseSig = np.random.normal(loc=mean,scale=sigma,size=len(timeAfterApnea))
        sig = ampSig*np.sin(2*np.pi*timeAfterApnea/period)+ (ampSig-amp/2) + noiseSig
        if timeIndex+len(timeAfterApnea) < len(signal):
            signal[timeIndex:timeIndex+len(timeAfterApnea)] = sig[:]
        else:
            signal[timeIndex::] = sig[0:len(signal)-timeIndex]
        
    
    return timestamps * 1000, signal


def signal3DGeneration(amplitude=20, period=4.0, mean=0, sigma=3, step=0.5, signalDuration=100, coeffMin = 0.10, coeffMax = 0.45, meanEvent = 1/20, meanEventApnea=1/120, otherDimensionsRatio = [0.3, 0.4], otherDimensionsNoiseVar = [0.1, 0.05]):

    timestamps, mainMotionSignal = signalGeneration(amplitude=amplitude, period=period, mean=mean, sigma=sigma, step=step, signalDuration=signalDuration, coeffMin=coeffMin, coeffMax=coeffMax, meanEvent=meanEvent, meanEventApnea=meanEventApnea)

    secondMotionSignal = mainMotionSignal * otherDimensionsRatio[0] + np.random.normal(loc=0, scale=otherDimensionsNoiseVar[0], size=mainMotionSignal.shape[0])
    thirdMotionSignal = mainMotionSignal * otherDimensionsRatio[1] + np.random.normal(loc=0, scale=otherDimensionsNoiseVar[1], size=mainMotionSignal.shape[0])

    signal3D = np.vstack((mainMotionSignal, secondMotionSignal, thirdMotionSignal))
    signal3D = signal3D.transpose(1, 0)

    # plt.figure()
    # plt.plot(signal3D[:, 0])
    # plt.plot(signal3D[:, 1])
    # plt.plot(signal3D[:, 2])
    # plt.show()

    return timestamps, signal3D

def signal2DGeneration(amplitude=20, period=4.0, mean=0, sigma=3, step=0.5, signalDuration=100, coeffMin = 0.10, coeffMax = 0.45, meanEvent = 1/20, meanEventApnea=1/120, otherDimensionsRatio = [0.3, 0.4], otherDimensionsNoiseVar = [0.1, 0.05]):

    timestamps, mainMotionSignal = signalGeneration(amplitude=amplitude, period=period, mean=mean, sigma=sigma, step=step, signalDuration=signalDuration, coeffMin=coeffMin, coeffMax=coeffMax, meanEvent=meanEvent, meanEventApnea=meanEventApnea)

    secondMotionSignal = mainMotionSignal * otherDimensionsRatio[0] + np.random.normal(loc=0, scale=otherDimensionsNoiseVar[0], size=mainMotionSignal.shape[0])

    signal2D = np.vstack((mainMotionSignal, secondMotionSignal))
    signal2D = signal2D.transpose(1, 0)

    # plt.figure()
    # plt.plot(signal3D[:, 0])
    # plt.plot(signal3D[:, 1])
    # plt.show()

    return timestamps, signal2D

