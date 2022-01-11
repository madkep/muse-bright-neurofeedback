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

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt
from numpy.random.mtrand import beta  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
from os import system
from datetime import date

# Handy little enum to make code more readable

from pynput import keyboard
from pynput.keyboard import Key, Controller

keyboard = Controller()


today = str(date.today())
count = 0
count_sec = 0
count_record_aux = 0
count_record = 0
play = False
bright = 1
error_percentage = 2 # The calibration error

f = open('param.csv','r')
message = f.read()

theta_threshold = (float(message[0:4])) + error_percentage
alpha_threshold = (float(message[5:9])) - error_percentage#SMR 
beta_threshold = (float(message[10:14])) + error_percentage
f.close()

class Band:
    Delta_AF7 = 0
    Delta_AF8 = 1
    Theta_AF7 = 2
    Theta_AF8 = 3
    Alpha_AF7 = 4
    Alpha_AF8 = 5
    Beta_AF7 = 6
    Beta_AF8 = 7

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




theta_cal = np.array([]) #Arrays for calibration and use
alpha_cal= np.array([]) 
beta_cal = np.array([])

def put_b(bright):
    system(f"brightness {bright}")



""" EXPERIMENTAL PARAMEßERS """
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
INDEX_CHANNEL = [1,2]
ch_names = ['AF7', 'AF8']
n_channels = 2

feature_names = ['delta-AF7', 'delta-AF8', 'theta-AF7', 'theta-AF8', 'alpha-AF7', 'alpha-AF8','beta-AF7', 'beta-AF8']

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """
    put_b(100)
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
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), n_channels))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, len(feature_names)))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            count += 1
            count_sec += 1
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
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
            #print(band_buffer)
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            
            theta_percentage = (((smooth_band_powers[Band.Theta_AF7] + 2)*25) +  ((smooth_band_powers[Band.Theta_AF8] + 2)*25) )/2#theta represent theta
            alpha_percentage = (((smooth_band_powers[Band.Alpha_AF7] + 2)*25) + ((smooth_band_powers[Band.Alpha_AF8] + 2)*25) )/2 #alpha represent SMR wave
            beta_percentage = (((smooth_band_powers[Band.Beta_AF7] + 2)*25)  +  ((smooth_band_powers[Band.Beta_AF8] + 2)*25) )/2#beta represent high beta

            if(count % 6 == 0 and count_sec > 30):
                theta_avg = np.mean(theta_cal)
                alpha_avg = np.mean(alpha_cal)
                beta_avg = np.mean(beta_cal)
                
                theta_cal = np.array([])
                alpha_cal= np.array([]) 
                beta_cal = np.array([])
                count = 0

                if( alpha_avg > alpha_threshold and theta_avg < theta_threshold and beta_avg < beta_threshold):
                    put_b(0.85)
                    if(play == False):
                        keyboard.press(" ")
                        keyboard.release(" ")
                        play = True
                    count_record_aux += 1
                    if(count_record_aux > count_record):
                        count_record = count_record_aux

                elif( alpha_avg > (alpha_threshold - 1) and theta_avg < (theta_threshold + 1) and beta_avg < (beta_threshold + 1) ):
                    put_b(0.75)
                    count_record_aux = 0

                elif( alpha_avg > (alpha_threshold - 2) and theta_avg < (theta_threshold + 2) and beta_avg < (beta_threshold + 2) ):
                    put_b(0.60)
                    count_record_aux = 0

                

                else:
                    put_b(0.2)
                    if(play == True):
                        keyboard.press(" ")
                        keyboard.release(" ")
                        play = False
                    count_record_aux = 0

                print(bcolors.OKCYAN + "    " + str(theta_avg)[0:5] + "    " + str(alpha_avg)[0:5]  + "    " + str(beta_avg)[0:5]  + " rc:" + str(count_record) + bcolors.OKCYAN)

                if(theta_avg < theta_threshold):
                    print( bcolors.OKGREEN +"theta: " + str(theta_threshold) + bcolors.OKGREEN , end=' ')
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



            elif(count_sec > 30):
                theta_cal = np.append(theta_cal, theta_percentage)
                alpha_cal = np.append(alpha_cal, alpha_percentage)
                beta_cal = np.append(beta_cal, beta_percentage)



    except KeyboardInterrupt:
        with open('record.csv','r') as f:
            for line in f:
                pass
            last_line = line
        x = last_line.split(",")
        if(x[1] == today+"\n"):
            print("Mejoraste la puntuación diaria")
            if(count_record > int(x[0])):
                readFile = open('record.csv')
                lines = readFile.readlines()
                readFile.close()
                w = open('record.csv','w')
                w.writelines([item for item in lines[:-1]])
                w.close()
                with open('record.csv','a+') as f:
                    f.write(str(count_record)+","+today+"\n")
        else:
            with open('record.csv','a+') as f:
                    f.write(str(count_record)+","+today+"\n")