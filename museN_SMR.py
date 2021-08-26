# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers
This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.
The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al
Adapted from https://github.com/NeuroTechX/bci-workshop
"""

from typing import Text
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
from os import system

# Handy little enum to make code more readable

#my dificult
#level 1
#theta < 60
#smr(alpha) > 60
#betha < 50

f = open('param.csv','r')
message = f.read()

theta_threshold = (float(message[0:4]))
alpha_threshold = (float(message[5:9])) #SMR 
beta_threshold = (float(message[10:14]))

theta_cal = np.array([])
alpha_cal= np.array([])
beta_cal = np.array([])

theta_average = 0
alpha_average = 0
beta_average = 0

calibration_time = 60
error_percentage = 3

f.close()

sec_cal = 0
sec_op = 0

cal = False


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def seconds(sec):
    return (sec*0.2)

def put_10():
    system("brightness 0.01")

def put_50():
    system("brightness 0.35")

def put_100():
    system("brightness 0.85")



""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [1]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """
    put_100()

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            try:
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                # Only keep the channel we're interested in
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            except:
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                # Only keep the channel we're interested in
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces


            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            
            theta_percentage = ((smooth_band_powers[Band.Theta] + 2)*25) #theta represent theta
            alpha_percentage = ((smooth_band_powers[Band.Alpha] + 2)*25) #alpha represent SMR wave
            beta_percentage = ((smooth_band_powers[Band.Beta] + 2)*25)  #beta represent high beta

            if(sec_cal == 0):
                option = input("Do you want to calibrate?(y o n) : ")
                if( option == "y"):
                    cal = False
                elif(option == "n"):
                    cal = True
                    sec_cal = -1

            if(cal == False):
                if(seconds(sec_cal) < calibration_time):
                    theta_cal = np.append(theta_cal, theta_percentage)
                    alpha_cal = np.append(alpha_cal, alpha_percentage)
                    beta_cal = np.append(beta_cal, beta_percentage)
                    sec_cal += 1
                else:
                    cal = True
                    theta_average = theta_cal.mean() + error_percentage
                    alpha_average = alpha_cal.mean() - error_percentage
                    beta_average = beta_cal.mean() + error_percentage
                    sec_cal = -1

                    option = input("Use " + str(alpha_average)[0:5] + " instead of " + str(alpha_threshold) +"? (y o n) higher better : " )
                    if(option == "y"):
                        alpha_threshold = alpha_average

                    option = input("Use " + str(theta_average)[0:5] + " instead of " + str(theta_threshold) +"? (y o n) lower better : " )
                    if(option == "y"):
                        theta_threshold = theta_average

                    option = input("Use " + str(beta_average)[0:5] + " instead of " + str(beta_threshold) +"? (y o n) lower better : " )
                    if(option == "y"):
                        beta_threshold = beta_average

                    #clean the arrays
                    theta_cal = np.array([])
                    alpha_cal= np.array([]) 
                    beta_cal = np.array([])

            elif(cal == True):
                sec_op += 1
                if(sec_op % 6 == 0):
                    print(" ")
                    theta_avg = theta_cal.mean()
                    alpha_avg = alpha_cal.mean()
                    beta_avg = beta_cal.mean()

                    if( alpha_avg > alpha_threshold and theta_avg < theta_threshold and beta_avg < beta_threshold):
                        put_100()
                            
                    elif( alpha_avg > alpha_threshold and theta_avg < theta_threshold ):
                        put_50()
                        
                    else:
                        put_10()


                    if(theta_avg < theta_threshold):
                        print( bcolors.OKGREEN +"theta: " + str(theta_threshold) + bcolors.OKGREEN, end=' ')
                    else:
                        print( bcolors.FAIL +"theta: " + str(theta_threshold) + bcolors.FAIL, end=' ')

                    if(alpha_avg > alpha_threshold):
                        print(bcolors.OKGREEN + "SMR" + str(alpha_threshold) + bcolors.OKGREEN, end=' ')
                    else:
                        print(bcolors.FAIL + "SMR" + str(alpha_threshold) + bcolors.FAIL, end=' ')

                    if(beta_avg < beta_threshold):
                        print(bcolors.OKGREEN + "beta" + str(beta_threshold) + bcolors.OKGREEN, end='\n')
                    else:
                        print(bcolors.FAIL + "beta" + str(beta_threshold) + bcolors.FAIL, end='\n')

                    #Clean the arrays
                    theta_cal = np.array([])
                    alpha_cal= np.array([]) 
                    beta_cal = np.array([])
                    sec_op = 0
                else:
                    theta_cal = np.append(theta_cal, theta_percentage)
                    alpha_cal = np.append(alpha_cal, alpha_percentage)
                    beta_cal = np.append(beta_cal, beta_percentage)

                    
    except KeyboardInterrupt:
        print('Closing!')
        f = open('param.csv','w')
        f.write(str(theta_threshold)[0:4] + "," + str(alpha_threshold)[0:4] + "," + str(beta_threshold)[0:4] )
        f.close()