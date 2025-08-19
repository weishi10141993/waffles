import json
import click
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import signal
import scipy.linalg
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
import glob
import os

from waffles.input_output.hdf5_structured import load_structured_waveformset
import waffles.plotting.drawing_tools as draw

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

# cathode channels
# all cathode except C3-ch2 not working, use C3-ch1 x2
channelsofinterest = [32, 33, 30, 31, 34, 35, 36, 37, 0, 2, 1, 3, 5, 7, 4, 6]
# membrane channels: M1-M4 very noisy, shouldn't trust noise count
#channelsofinterest = [45, 42, 44, 41, 0, 20, 30, 10]
#channelsofinterest = [20, 30] # if quartz window is M7 (30)
#channelsofinterest = [0, 10] # if quartz window is M8 (10)
#channelsofinterest = [45, 42, 44, 41] # HD style HPK

# region to find peak adc
prebeamtrigtick = 1700
postbeamtrigtick = 2300
filterlength = 10
percentile4baseline = 10
plotwfms = False

daq_pd_dt = []
PE = []
ePE = []
pPE = []
kPE = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

modules = ["None"] * 100
adc_2_spe = [0] * 100

# set beam SPE ADC
# per: https://indico.fnal.gov/event/70796/contributions/321711/attachments/190670/263594/LED_calibration_updates.pdf
for ich in range(len(channelsofinterest)):
    # depends on channel, set at roughly SPE amplitude
    if channelsofinterest[ich] == 32:
        modules[channelsofinterest[ich]]= ["C1"]
        adc_2_spe[channelsofinterest[ich]] = 20
    if channelsofinterest[ich] == 33:
        modules[channelsofinterest[ich]]= ["C1"]
        adc_2_spe[channelsofinterest[ich]] = 8
    if channelsofinterest[ich] == 30:
        modules[channelsofinterest[ich]]= ["C2"]
        adc_2_spe[channelsofinterest[ich]] = 10
    if channelsofinterest[ich] == 31:
        modules[channelsofinterest[ich]]= ["C2"]
        adc_2_spe[channelsofinterest[ich]] = 15
    if channelsofinterest[ich] == 34:
        modules[channelsofinterest[ich]]= ["C3"]
        adc_2_spe[channelsofinterest[ich]] = 10
    if channelsofinterest[ich] == 35:
        modules[channelsofinterest[ich]]= ["C3"]
        adc_2_spe[channelsofinterest[ich]] = 12
    if channelsofinterest[ich] == 36:
        modules[channelsofinterest[ich]]= ["C4"]
        adc_2_spe[channelsofinterest[ich]] = 12
    if channelsofinterest[ich] == 37:
        modules[channelsofinterest[ich]]= ["C4"]
        adc_2_spe[channelsofinterest[ich]] = 12
    if channelsofinterest[ich] == 0:
        modules[channelsofinterest[ich]]= ["C5"]
        adc_2_spe[channelsofinterest[ich]] = 18
    if channelsofinterest[ich] == 2:
        modules[channelsofinterest[ich]]= ["C5"]
        adc_2_spe[channelsofinterest[ich]] = 20
    if channelsofinterest[ich] == 1:
        modules[channelsofinterest[ich]]= ["C6"]
        adc_2_spe[channelsofinterest[ich]] = 12.5
    if channelsofinterest[ich] == 3:
        modules[channelsofinterest[ich]]= ["C6"]
        adc_2_spe[channelsofinterest[ich]] = 12
    if channelsofinterest[ich] == 5:
        modules[channelsofinterest[ich]]= ["C7"]
        adc_2_spe[channelsofinterest[ich]] = 20
    if channelsofinterest[ich] == 7:
        modules[channelsofinterest[ich]]= ["C7"]
        adc_2_spe[channelsofinterest[ich]] = 18
    if channelsofinterest[ich] == 4:
        modules[channelsofinterest[ich]]= ["C8"]
        adc_2_spe[channelsofinterest[ich]] = 10
    if channelsofinterest[ich] == 6:
        modules[channelsofinterest[ich]]= ["C8"]
        adc_2_spe[channelsofinterest[ich]] = 8

# 2nd beam period Aug 5
# run 38563, 2 GeV incluide all particles with High Presion Cherenkov off
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038563_cathode/"
# run 38564, 2 GeV incluide all particles with the optimal Cherenkov configuration
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038564_cathode/"
# run 38565, 2 GeV incluide all particles with the optimal Cherenkov configuration
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038565_cathode/"


# 0.5 GeV - run038648  no cathode file??

# 1 GeV - HP 14 Bar, LP 5 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038645_cathode/"
# 1.5 GeV - HP 14 Bar, LP 4 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038670_cathode/"
# 2 GeV - Run 038563 is in self trig mode or full stream? HighP Cerenkov OFF. Full ticks is 7.3ms, check beam trig time.!!! run 38651 has no cathode file.
dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038563_cathode/"
# 3 GeV - HP 14 Bar, LP 1.2 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038658_cathode/"
# 4 GeV - HP 14 Bar, LP 1.2 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038660_cathode/"
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
print("1st wfm trigger types: ", wfset.waveforms[0].trigger_type_names)
print("1st wfm trigger type name: ", wfset.waveforms[0].trigger_type_names[0])
print("1st wfm number of ticks: ", len(wfset.waveforms[0].adcs))
print("1st wfm channel: ", wfset.waveforms[0].channel)


# Get of daq trigs
daqstamps = []
for iwfm in range(len(wfset.waveforms)):
#for iwfm in range(50000):
    if wfset.waveforms[iwfm].daq_window_timestamp in daqstamps: continue
    daqstamps.append(wfset.waveforms[iwfm].daq_window_timestamp)

# of daq trigs must be smaller thsn number of wfms
PE =  [0] * len(wfset.waveforms)
ePE = [0] * len(daqstamps)
pPE = [0] * len(daqstamps)
kPE = [0] * len(daqstamps)
perdaqtrigchannellist = []
for idaq in range(len(daqstamps)):
    # Append sublists
    perdaqtrigchannellist.append([idaq] * 0)

# loop to calculate PE
for iwfm in range(len(wfset.waveforms)):
#for iwfm in range(50000):
    if iwfm % 10000 == 0:
        print(iwfm)

    channelofinterest = wfset.waveforms[iwfm].channel

    # find baseline of the wfm
    baseline_ADC = 0
    peak_adc = 0
    # smooth it out
    wfset.waveforms[iwfm].filtered = np.convolve(wfset.waveforms[iwfm].adcs, np.ones(filterlength), 'valid') / filterlength
    # use the mode of adcs of a wfm as baseline
    #baseline_ADC = statistics.mode(wfset.waveforms[iwfm].filtered)
    # use average from certain lowest percentile
    # need x <= since in some wfms has same adcs
    baseline_ADC = statistics.mean(filter(lambda x: x <= np.percentile(wfset.waveforms[iwfm].filtered, percentile4baseline), wfset.waveforms[iwfm].filtered))
    peak_adc = np.max(wfset.waveforms[iwfm].filtered[prebeamtrigtick:postbeamtrigtick])

    # get PE
    PE[iwfm] = (peak_adc-baseline_ADC)/adc_2_spe[channelofinterest]


    ###############
    # control plot
    ###############
    daq_pd_dt.append(abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp))

    ###############
    # control plot
    ###############
    if iwfm < 1000 and plotwfms == True:
        xaxis = [x for x in range(len(wfset.waveforms[iwfm].adcs))]
        plt.plot(xaxis, wfset.waveforms[iwfm].adcs)
        plt.savefig("plots/"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_adcs.pdf")
        plt.clf() # important to clear figure
        plt.close()

        xaxis = [x for x in range(len(wfset.waveforms[iwfm].filtered))]
        plt.plot(xaxis, wfset.waveforms[iwfm].filtered)
        plt.hlines(y=[baseline_ADC], xmin=0, xmax=len(wfset.waveforms[0].adcs), colors=['r'], linestyles=['--']) # also plot the calculated baseline in red for each wfm
        plt.savefig("plots/"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_movingavged.pdf")
        plt.clf() # important to clear figure
        plt.close()

dt_allwfms = [(x) for x in daq_pd_dt]
plt.hist(dt_allwfms, range=(0,2500), bins=500, log=True)
plt.xlabel('|t_daq - t_pd|')
plt.draw()
plt.savefig("./dt_"+str(wfset.waveforms[0].run_number)+"_zoomin.pdf")
plt.clf() # important to clear figure
plt.close()

pe_allcathode = [(x) for x in PE]
plt.hist(pe_allcathode, range=(0,4000), bins=800, log=True)
plt.xlabel('PE')
plt.draw()
plt.savefig("./PE_per_wfm_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()


# loop to add PE
for iwfm in range(len(wfset.waveforms)):
#for iwfm in range(50000):
    index = daqstamps.index(wfset.waveforms[iwfm].daq_window_timestamp)
    #print("iwfm:", iwfm, "daqtime index:", index)

    # SELECT BEAM COINCIDENCE
    if abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < 2000 and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) > 1750:

        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHL':
            # e/mu/pi
            ePE[index] = ePE[index] + PE[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel) # this needs to have all ch
        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHxLx':
            # proton
            pPE[index] = pPE[index] + PE[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel)
        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHLx':
            # kaon
            kPE[index] = kPE[index] + PE[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel)


# check filled info
"""
print("ePE[0]: ", ePE[0])
print("ePE[1]: ", ePE[1])
print("pPE[0]: ", pPE[0])
print("pPE[1]: ", pPE[1])
print("kPE[0]: ", kPE[0])
print("kPE[1]: ", kPE[1])
print("perdaqtrigchannellist[0]: ", perdaqtrigchannellist[0])
print("perdaqtrigchannellist[1]: ", perdaqtrigchannellist[1])
"""

final_ePE = []
final_pPE = []
final_kPE = []
for idaq in range(len(daqstamps)):
    if len(perdaqtrigchannellist[idaq]) == 16:
        if ePE[idaq] != 0: final_ePE.append(ePE[idaq])
        if pPE[idaq] != 0: final_pPE.append(pPE[idaq])
        if kPE[idaq] != 0: final_kPE.append(kPE[idaq])


print(" === FINAL REPORT === ")
ePE_allwfms = [(x) for x in final_ePE] # also need to require each channel present
plt.hist(ePE_allwfms, range=(0,20000), bins=200, log=True)
plt.xlabel('e trig PE')
plt.draw()
plt.savefig("./e_totPE_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()
if len(ePE_allwfms) > 0:
    ePE_mean = statistics.mean(ePE_allwfms)
    ePE_std = statistics.stdev(ePE_allwfms)
    print("ePE_mean: ", ePE_mean)
    print("ePE_std: ", ePE_std)

pPE_allwfms = [(x) for x in final_pPE]
plt.hist(pPE_allwfms, range=(0,20000), bins=200, log=True)
plt.xlabel('proton trig PE')
plt.draw()
plt.savefig("./p_totPE_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()
if len(pPE_allwfms) > 0:
    pPE_mean = statistics.mean(pPE_allwfms)
    pPE_std = statistics.stdev(pPE_allwfms)
    print("pPE_mean: ", pPE_mean)
    print("pPE_std: ", pPE_std)

kPE_allwfms = [(x) for x in final_kPE]
plt.hist(kPE_allwfms, range=(0,20000), bins=200, log=True)
plt.xlabel('kaon trig PE')
plt.draw()
plt.savefig("./k_totPE_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()
if len(kPE_allwfms) > 0:
    kPE_mean = statistics.mean(kPE_allwfms)
    kPE_std = statistics.stdev(kPE_allwfms)
    print("kPE_mean: ", kPE_mean)
    print("kPE_std: ", kPE_std)
