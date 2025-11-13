import json
import click
from pathlib import Path
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import statistics
from scipy import signal
import scipy.linalg
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad
import glob
import os
import math
import csv
from collections import Counter

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

# cathode channels
channelsofinterest = [32, 33, 30, 31, 34, 35, 36, 37, 0, 2, 1, 3, 5, 7, 4, 6]
#channelsofinterest = [32]
#channelsofinterest = [32, 31, 34, 36, 2, 1, 5, 4]
# membrane channels: M1-M4 very noisy, shouldn't trust noise count
#channelsofinterest = [45, 42, 44, 41, 0, 20, 30, 10]
#channelsofinterest = [20, 30] # if quartz window is M7 (30)
#channelsofinterest = [0, 10] # if quartz window is M8 (10)
#channelsofinterest = [45, 42, 44, 41] # HD style HPK

Beamrun = True
# Beam coincidence cathode
beamcoinstart = 1800
beamcoinstop = 2000
# region to find peak adc
prebeamtrigtick = 1700
postbeamtrigtick = 2400 # if saturated more than this, you can't recover charge under ToT
checkToT_start = 1700
checkToT_end = 2400
peakadccut_max = 14000
peakadccut_min = 8000
beam_coincidence_cut = 20 # pd ticks

filterlength = 10
percentile4baseline = 10
tick_2_ns = 16
plotwfms = False
plotpersistence = True

nadcthrs = 30 # number of ADC thresholds
countToT = np.zeros((len(channelsofinterest),nadcthrs), dtype=np.int32) # number of time ticks above a certain threshold, per tick 16ns
ThresholdStep = np.zeros((nadcthrs,), dtype=np.int32) # 100 adc ~ 10PE step, 3500 - 35 step
ThresholdStep[0] = 2000 # if this is hitting 1000, you have noise effect
for ithres in range(1, nadcthrs):
    ThresholdStep[ithres] = ThresholdStep[ithres-1] + 500

##################################
# 3rd beam period Aug 22
##################################
#++++++++++++++++++++++++++++++++++++
# Pure electrons: all ask for HL trig
#++++++++++++++++++++++++++++++++++++
# run 39183, 8 GeV, L - 0.1b, H - 1b, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039183_cathode/"
# run 39108, 5 GeV, L - 0.25bar, H - 5bar, Cu target - done
dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039108_cathode/"

BaselineADCAllWfms = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

modules = ["None"] * 100

for ich in range(len(channelsofinterest)):
    # depends on channel, set at roughly SPE amplitude
    if channelsofinterest[ich] == 32:
        modules[channelsofinterest[ich]]= "C1"
    if channelsofinterest[ich] == 33:
        modules[channelsofinterest[ich]]= "C1"
    if channelsofinterest[ich] == 30:
        modules[channelsofinterest[ich]]= "C2"
    if channelsofinterest[ich] == 31:
        modules[channelsofinterest[ich]]= "C2"
    if channelsofinterest[ich] == 34:
        modules[channelsofinterest[ich]]= "C3"
    if channelsofinterest[ich] == 35:
        modules[channelsofinterest[ich]]= "C3"
    if channelsofinterest[ich] == 36:
        modules[channelsofinterest[ich]]= "C4"
    if channelsofinterest[ich] == 37:
        modules[channelsofinterest[ich]]= "C4"
    if channelsofinterest[ich] == 0:
        modules[channelsofinterest[ich]]= "C5"
    if channelsofinterest[ich] == 2:
        modules[channelsofinterest[ich]]= "C5"
    if channelsofinterest[ich] == 1:
        modules[channelsofinterest[ich]]= "C6"
    if channelsofinterest[ich] == 3:
        modules[channelsofinterest[ich]]= "C6"
    if channelsofinterest[ich] == 5:
        modules[channelsofinterest[ich]]= "C7"
    if channelsofinterest[ich] == 7:
        modules[channelsofinterest[ich]]= "C7"
    if channelsofinterest[ich] == 4:
        modules[channelsofinterest[ich]]= "C8"
    if channelsofinterest[ich] == 6:
        modules[channelsofinterest[ich]]= "C8"

############################
# Load waveforms
############################

waveforms =[]
file_pattern = os.path.join(dirpath, "*.hdf5")
hdf5_files = glob.glob(file_pattern)
for filepath in hdf5_files:
    print(filepath)
    iwfset = load_structured_waveformset(str(filepath))
    waveforms.extend(iwfset.waveforms)

wfset = WaveformSet(*waveforms)

print("file path: ", dirpath)
print("run number: ", wfset.waveforms[0].run_number)
print("tot waveforms from all channels: ", len(wfset.waveforms))
print("1st wfm attributes: ", vars(wfset.waveforms[0]))
print("1st wfm adcs: ", wfset.waveforms[0].adcs)
print("1st wfm number of ticks: ", len(wfset.waveforms[0].adcs))
if Beamrun == True: print("1st wfm trigger types: ", wfset.waveforms[0].trigger_type_names)
if Beamrun == True: print("1st wfm trigger type name: ", wfset.waveforms[0].trigger_type_names[0])
print("1st wfm channel: ", wfset.waveforms[0].channel)

avf_wfm_max_tick = len(wfset.waveforms[0].adcs) - filterlength
avg_wfm          = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
avg_wfm_HL       = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
avg_wfm_HLx      = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
avg_wfm_HxLx     = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
bslin_subtracted_filtered_wfm  = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
countwfms        = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HL     = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HLx    = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HxLx   = np.zeros((len(channelsofinterest),), dtype=np.int32)
mode_value_max_count = np.zeros((len(channelsofinterest),), dtype=np.int32)
ToT_start        = np.zeros((nadcthrs,), dtype=np.int32)
ToT_stop         = np.zeros((nadcthrs,), dtype=np.int32)
ToT              = np.zeros((nadcthrs,), dtype=np.int32)
QoT              = np.zeros((nadcthrs,), dtype=np.float32)

# Main loop
#for iwfm in range(len(wfset.waveforms)):
for iwfm in range(40000):
#for iwfm in range(1000):

        if iwfm % 10000 == 0:
            print(iwfm)

        for ich in range(len(channelsofinterest)):
            #print("ich: ", ich)
            channelofinterest = channelsofinterest[ich]

            if wfset.waveforms[iwfm].channel == channelofinterest:

                # find baseline of the wfm
                baseline_ADC = 0
                peak_adc = 0
                peak_mode_adc = 0

                # smooth it out
                wfset.waveforms[iwfm].filtered = np.convolve(wfset.waveforms[iwfm].adcs, np.ones(filterlength), 'valid') / filterlength
                # use the mode of adcs of a wfm as baseline
                #baseline_ADC = statistics.mode(wfset.waveforms[iwfm].filtered)
                # use average from certain lowest percentile
                # need x <= since in some wfms has same adcs
                baseline_ADC = statistics.mean(filter(lambda x: x <= np.percentile(wfset.waveforms[iwfm].filtered, percentile4baseline), wfset.waveforms[iwfm].filtered))

                for itick in range(avf_wfm_max_tick):
                    bslin_subtracted_filtered_wfm[ich][itick] = wfset.waveforms[iwfm].filtered[itick] - baseline_ADC

                peak_adc = np.max(bslin_subtracted_filtered_wfm[ich][prebeamtrigtick:postbeamtrigtick])
                peak_mode_adc = statistics.mode(bslin_subtracted_filtered_wfm[ich][prebeamtrigtick:postbeamtrigtick]) # check for saturation
                # Create a Counter object
                counts = Counter(bslin_subtracted_filtered_wfm[ich][prebeamtrigtick:postbeamtrigtick])
                # Find the most common element(s) and their counts
                # most_common(1) returns a list of tuples: [(element, count)]
                most_common_element = counts.most_common(1)[0]

                mode_value = most_common_element[0]
                mode_count = most_common_element[1]

                if peak_mode_adc != mode_value:
                    print("WARNING: statistics.mode returns a different mode to Counter")
                    print("statistics.mode:", peak_mode_adc)
                    print("Counter mode", mode_value)

                # control plot: overlay wfms
                if plotpersistence == True:
                    plt.figure("bigunsaturated"+str(channelofinterest))
                    if peak_adc > peakadccut_min and peak_adc < peakadccut_max and peak_adc != peak_mode_adc: # if peak equals mode, it's saturated
                        xaxis = [x for x in range(len(bslin_subtracted_filtered_wfm[ich]))]
                        plt.plot(xaxis, bslin_subtracted_filtered_wfm[ich])

                    plt.figure("supersaturated"+str(channelofinterest))
                    if peak_adc == peak_mode_adc and peak_adc > peakadccut_min: # saturated
                        if mode_count > mode_value_max_count[ich]: # only keep larger saturation
                            mode_value_max_count[ich] = mode_count
                            xaxis = [x for x in range(len(bslin_subtracted_filtered_wfm[ich]))]
                            plt.plot(xaxis, bslin_subtracted_filtered_wfm[ich])

                # MAIN SELECTION For cathode modules
                if abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < beamcoinstop and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) > beamcoinstart and peak_adc > peakadccut_min and peak_adc < peakadccut_max and peak_adc != peak_mode_adc:
                    #print("beamtrue: ", iwfm)

                    # Fill baseline ADC of all waveforms in the data (a distribution of baselines)
                    BaselineADCAllWfms.append(baseline_ADC)

                    countwfms[ich] = countwfms[ich] + 1
                    for itick in range(avf_wfm_max_tick):
                        # sum up wfms for avg later
                        avg_wfm[ich][itick] = avg_wfm[ich][itick] + bslin_subtracted_filtered_wfm[ich][itick]
                        # typically electron
                        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHL':
                            avg_wfm_HL[ich][itick] = avg_wfm_HL[ich][itick] + bslin_subtracted_filtered_wfm[ich][itick]
                        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHLx':
                            # typically pion
                            avg_wfm_HLx[ich][itick] = avg_wfm_HLx[ich][itick] + bslin_subtracted_filtered_wfm[ich][itick]
                        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHxLx':
                            avg_wfm_HxLx[ich][itick] = avg_wfm_HxLx[ich][itick] + bslin_subtracted_filtered_wfm[ich][itick]
                    # Here for different particles
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHL':
                        countwfms_HL[ich] = countwfms_HL[ich] + 1
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHLx':
                        # typically pion
                        countwfms_HLx[ich] = countwfms_HLx[ich] + 1
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHxLx':
                        # k/proton- need ToF
                        countwfms_HxLx[ich] = countwfms_HxLx[ich] + 1

                    # validation plot for selected evts
                    if iwfm < 500 and plotwfms == True:
                        xaxis = [x for x in range(len(wfset.waveforms[iwfm].adcs))]
                        plt.figure(num="raw adc")
                        plt.plot(xaxis, wfset.waveforms[iwfm].adcs)
                        plt.savefig("plots/"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_adcs.pdf")
                        plt.clf() # important to clear figure
                        plt.close()

                        xaxis = [x for x in range(len(wfset.waveforms[iwfm].filtered))]
                        plt.figure(num="filtered adc")
                        plt.plot(xaxis, wfset.waveforms[iwfm].filtered)
                        plt.hlines(y=[baseline_ADC], xmin=0, xmax=len(wfset.waveforms[0].adcs), colors=['r'], linestyles=['--']) # also plot the calculated baseline in red for each wfm
                        plt.savefig("plots/"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_movingavged.pdf")
                        plt.clf() # important to clear figure
                        plt.close()


if plotpersistence == True:
    for ich in range(len(channelsofinterest)):
        plt.figure("bigunsaturated"+str(channelsofinterest[ich]))
        plt.savefig("./persistent_big_unsaturated_wfms_"+str(wfset.waveforms[0].run_number)+"_cathode_ch_"+str(channelsofinterest[ich])+".pdf")
        plt.clf() # important to clear figure
        plt.close()
        plt.figure("supersaturated"+str(channelsofinterest[ich]))
        plt.savefig("./persistent_super_saturated_wfms_"+str(wfset.waveforms[0].run_number)+"_cathode_ch_"+str(channelsofinterest[ich])+".pdf")
        plt.clf() # important to clear figure
        plt.close()

print("iwfm at end: ", iwfm)
print("run number: ", wfset.waveforms[0].run_number)

# avg wfm
print("================= Avg Wfm Report =================  ")
for ich in range(len(channelsofinterest)):
    print("==== module ", modules[channelsofinterest[ich]], " (ch ", channelsofinterest[ich], ") ==== ")
    print("tot wfms for avg: ", countwfms[ich])
    print("tot wfms for avg - HL: ", countwfms_HL[ich])
    print("tot wfms for avg - HLx: ", countwfms_HLx[ich])
    print("tot wfms for avg - HxLx: ", countwfms_HxLx[ich])
    # loop over ticks
    for itick in range(avf_wfm_max_tick):
        if countwfms[ich]>0: avg_wfm[ich][itick] = avg_wfm[ich][itick]*1.0 / countwfms[ich]
        if countwfms_HL[ich]>0: avg_wfm_HL[ich][itick] = avg_wfm_HL[ich][itick]*1.0 / countwfms_HL[ich]
        if countwfms_HLx[ich]>0: avg_wfm_HLx[ich][itick] = avg_wfm_HLx[ich][itick]*1.0 / countwfms_HLx[ich]
        if countwfms_HxLx[ich]>0: avg_wfm_HxLx[ich][itick] = avg_wfm_HxLx[ich][itick]*1.0 / countwfms_HxLx[ich]

# Subtract basline of avg wfm
for ich in range(len(channelsofinterest)):

    # Overall
    if countwfms[ich]>0:
        with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm.txt", "w") as f:

            # subtract basline of avg wfm for deconvolve
            baseline_ADC_avg_wfm = statistics.mean(filter(lambda x: x <= np.percentile(avg_wfm[ich], percentile4baseline), avg_wfm[ich]))
            #print("ch", channelsofinterest[ich], " avg wfm baseline: ", baseline_ADC_avg_wfm)
            for itick in range(avf_wfm_max_tick):
                avg_wfm[ich][itick] = avg_wfm[ich][itick] - baseline_ADC_avg_wfm
                # store avg wfm in txt
                f.write(str(avg_wfm[ich][itick])+ "\n")

        xaxis = [x for x in range(len(avg_wfm[ich]))]
        plt.plot(xaxis, avg_wfm[ich], 'blue', label=str(modules[channelsofinterest[ich]]))
        plt.grid(True)
        plt.legend()
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm.pdf")
        plt.clf() # important to clear figure
        plt.close()

    # HL - typically electron
    if countwfms_HL[ich]>0:
        with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HL.txt", "w") as f:

            # subtract basline of avg wfm for deconvolve
            baseline_ADC_avg_wfm_HL = statistics.mean(filter(lambda x: x <= np.percentile(avg_wfm_HL[ich], percentile4baseline), avg_wfm_HL[ich]))
            #print("ch", channelsofinterest[ich], " avg wfm baseline: ", baseline_ADC_avg_wfm)
            for itick in range(avf_wfm_max_tick):
                avg_wfm_HL[ich][itick] = avg_wfm_HL[ich][itick] - baseline_ADC_avg_wfm_HL
                # store avg wfm in txt
                f.write(str(avg_wfm_HL[ich][itick])+ "\n")

        xaxis = [x for x in range(len(avg_wfm_HL[ich]))]
        plt.plot(xaxis, avg_wfm_HL[ich], 'blue', label=str(modules[channelsofinterest[ich]]))
        plt.grid(True)
        plt.legend()
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HL.pdf")
        plt.clf() # important to clear figure
        plt.close()

    # HLx - typically pi
    if countwfms_HLx[ich]>0:
        with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HLx.txt", "w") as f:

            # subtract basline of avg wfm for deconvolve
            baseline_ADC_avg_wfm_HLx = statistics.mean(filter(lambda x: x <= np.percentile(avg_wfm_HLx[ich], percentile4baseline), avg_wfm_HLx[ich]))
            #print("ch", channelsofinterest[ich], " avg wfm baseline: ", baseline_ADC_avg_wfm)
            for itick in range(avf_wfm_max_tick):
                avg_wfm_HLx[ich][itick] = avg_wfm_HLx[ich][itick] - baseline_ADC_avg_wfm_HLx
                # store avg wfm in txt
                f.write(str(avg_wfm_HLx[ich][itick])+ "\n")

        xaxis = [x for x in range(len(avg_wfm_HLx[ich]))]
        plt.plot(xaxis, avg_wfm_HLx[ich], 'blue', label=str(modules[channelsofinterest[ich]]))
        plt.grid(True)
        plt.legend()
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HLx.pdf")
        plt.clf() # important to clear figure
        plt.close()

    # HxLx - typically k/p
    if countwfms_HxLx[ich]>0:
        with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HxLx.txt", "w") as f:

            # subtract basline of avg wfm for deconvolve
            baseline_ADC_avg_wfm_HxLx = statistics.mean(filter(lambda x: x <= np.percentile(avg_wfm_HxLx[ich], percentile4baseline), avg_wfm_HxLx[ich]))
            #print("ch", channelsofinterest[ich], " avg wfm baseline: ", baseline_ADC_avg_wfm)
            for itick in range(avf_wfm_max_tick):
                avg_wfm_HxLx[ich][itick] = avg_wfm_HxLx[ich][itick] - baseline_ADC_avg_wfm_HxLx
                # store avg wfm in txt
                f.write(str(avg_wfm_HxLx[ich][itick])+ "\n")

        xaxis = [x for x in range(len(avg_wfm_HxLx[ich]))]
        plt.plot(xaxis, avg_wfm_HxLx[ich], 'blue', label=str(modules[channelsofinterest[ich]]))
        plt.grid(True)
        plt.legend()
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HxLx.pdf")
        plt.clf() # important to clear figure
        plt.close()

    ##################################
    # Time over threshold calculation
    ##################################
    ToT_final = []
    QoT_final = []
    for itrg in range(4):
        # Append sublists
        ToT_final.append([itrg] * 0)
        QoT_final.append([itrg] * 0)

    for itrg in range(4):
        if itrg == 0:
            avg_wfm_final = avg_wfm_HL[ich]
            trgname = "HL"
        if itrg == 1:
            avg_wfm_final = avg_wfm_HLx[ich]
            trgname = "HLx"
        if itrg == 2:
            avg_wfm_final = avg_wfm_HxLx[ich]
            trgname = "HxLx"
        if itrg == 3:
            avg_wfm_final = avg_wfm[ich]
            trgname = "OR"

        print("=== trg type:", trgname, " ===")

        # For each avg wfm (baseline subtracted), project a line across any y value (ADC), integrate the lost charge (ADCxtime area above the projected line)
        # at zero projection, it equals the total charge of the wfm

        # Count time over threshold in the wfm waveform
        for ithres in range(nadcthrs):
            # Loop over beam time region in the avg wfm of a specific channel
            skip_uprise = 0
            for itick in range(checkToT_start, checkToT_end):
                # rising edge has to be below 2000 ticks in beam wfm
                if ( ( avg_wfm_final[itick] - ThresholdStep[ithres] ) < 0 and ( avg_wfm_final[itick+1] - ThresholdStep[ithres] ) > 0 ) and itick < 2000:
                    # automatically pick up the earlist uprise
                    if skip_uprise == 0:
                        ToT_start[ithres] = itick
                        skip_uprise = 1
                if ( ( avg_wfm_final[itick] - ThresholdStep[ithres] ) > 0 and ( avg_wfm_final[itick+1] - ThresholdStep[ithres] ) < 0 ):
                    # automatically pick up the latest downturn
                    ToT_stop[ithres] = itick

        # report ToT for this ch
        print("ToT [ticks] for channel: ", channelsofinterest[ich])
        for ithres in range(nadcthrs):

            ToT[ithres] = ToT_stop[ithres] - ToT_start[ithres]

            # Calculate lost charge (integral above adc threshold) and tot charge
            QoT[ithres] = np.sum(avg_wfm_final[ToT_start[ithres]:ToT_stop[ithres]] - ThresholdStep[ithres])*tick_2_ns

            # for threshold beyond peak adc, set zero
            if ThresholdStep[ithres] > peak_adc:
                ToT[ithres] = 0
                QoT[ithres] = 0

            # do not plot zero ToT
            if ToT[ithres] > 0 and QoT[ithres] > 0:
                ToT_final[itrg].append(ToT[ithres])
                QoT_final[itrg].append(QoT[ithres])
                print("Thrshold [adc]", ThresholdStep[ithres], " ToT: ", ToT[ithres], " ticks, QoT: ", QoT[ithres], "[Adc x ns]")

    # End loop over trigger

    # plot QoT vs ToT for each channel
    plt.plot(ToT_final[0], QoT_final[0], label='HL',   marker='o', markerfacecolor='green', markeredgecolor='green', color='green', linestyle='-')
    plt.plot(ToT_final[1], QoT_final[1], label='HLx',  marker='o', markerfacecolor='blue',  markeredgecolor='blue',  color='blue',  linestyle='-')
    plt.plot(ToT_final[2], QoT_final[2], label='HxLx', marker='o', markerfacecolor='pink',  markeredgecolor='pink',  color='pink',  linestyle='-')
    plt.plot(ToT_final[3], QoT_final[3], label='OR',   marker='o', markerfacecolor='black', markeredgecolor='black', color='black', linestyle='-')
    plt.legend(loc='upper left')

    # Adding labels and a title
    plt.xlabel("ToT [ticks]")
    plt.ylabel("QoT [Adc x ns]")
    plt.title("QoT vs ToT - "+str(modules[channelsofinterest[ich]])+" ch "+str(channelsofinterest[ich]))

    # Displaying the graph
    plt.savefig("QoT_ToT_"+str(modules[channelsofinterest[ich]])+"_ch_"+str(channelsofinterest[ich])+".pdf")
    plt.clf() # important to clear figure
    plt.close()

    # Write to CSV
    # Combine the lists into a list of lists, padding shorter ones
    # Find the maximum length among all lists
    max_len = max(len(ToT_final[0]), len(ToT_final[1]), len(ToT_final[2]), len(ToT_final[3]))

    # Pad shorter lists with empty strings to match the maximum length
    data_rows = []
    for i in range(max_len):
        row = []
        row.append(         str(ToT_final[0][i]) if i < len(ToT_final[0]) else '')
        row.append('      '+str(QoT_final[0][i]) if i < len(ToT_final[0]) else '')
        row.append('      '+str(ToT_final[1][i]) if i < len(ToT_final[1]) else '')
        row.append('      '+str(QoT_final[1][i]) if i < len(ToT_final[1]) else '')
        row.append('      '+str(ToT_final[2][i]) if i < len(ToT_final[2]) else '')
        row.append('      '+str(QoT_final[2][i]) if i < len(ToT_final[2]) else '')
        row.append('      '+str(ToT_final[3][i]) if i < len(ToT_final[3]) else '')
        row.append('      '+str(QoT_final[3][i]) if i < len(ToT_final[3]) else '')
        data_rows.append(row)

    # Define column headers
    headers = ['HL ToT', '      HL QoT', '      HLx ToT', '      HLx QoT', '      HxLx ToT', '      HxLx QoT', '      OR ToT', '      OR QoT']

    with open(f'{str(modules[channelsofinterest[ich]])}_ch_{str(channelsofinterest[ich])}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write headers
        writer.writerows(data_rows) # Write the data rows
