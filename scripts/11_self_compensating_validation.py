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
from scipy.integrate import trapezoid
import glob
import os

from waffles.input_output.hdf5_structured import load_structured_waveformset

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

TotWfmsToRun = 50000
# region to find peak adc
prebeamtrigtick = 1700
postbeamtrigtick = 2300
# region to integrate charge 1000 ticks
qintegrate_start = 1700
qintegrate_end = 2700
tick_2_ns = 16
filterlength = 10
percentile4baseline = 10
plotwfms = False

daq_pd_dt = []

PE_peak = []
PE_charge = []

HLPE_peak = []
HLxPE_peak = []
HxLxPE_peak = []

HLPE_charge = []
HLxPE_charge = []
HxLxPE_charge = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

modules = ["None"] * 100
adc_2_spe = [0] * 100
charge_2_spe = [0] * 100

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# set beam SPE ADC
# per: https://indico.fnal.gov/event/70796/contributions/321711/attachments/190670/263594/LED_calibration_updates.pdf
for ich in range(len(channelsofinterest)):
    # depends on channel, set at roughly SPE amplitude
    if channelsofinterest[ich] == 32:
        modules[channelsofinterest[ich]]= ["C1"]
        adc_2_spe[channelsofinterest[ich]] = 20
        charge_2_spe[channelsofinterest[ich]] = 3450.06 # adc x ns , not ticks!!!
    if channelsofinterest[ich] == 33:
        modules[channelsofinterest[ich]]= ["C1"]
        adc_2_spe[channelsofinterest[ich]] = 8
        charge_2_spe[channelsofinterest[ich]] = 1950.04
    if channelsofinterest[ich] == 30:
        modules[channelsofinterest[ich]]= ["C2"]
        adc_2_spe[channelsofinterest[ich]] = 10
        charge_2_spe[channelsofinterest[ich]] = 2138.67
    if channelsofinterest[ich] == 31:
        modules[channelsofinterest[ich]]= ["C2"]
        adc_2_spe[channelsofinterest[ich]] = 15
        charge_2_spe[channelsofinterest[ich]] = 4251.92
    if channelsofinterest[ich] == 34:
        modules[channelsofinterest[ich]]= ["C3"]
        adc_2_spe[channelsofinterest[ich]] = 10
        charge_2_spe[channelsofinterest[ich]] = 1799.49
    if channelsofinterest[ich] == 35:
        modules[channelsofinterest[ich]]= ["C3"]
        adc_2_spe[channelsofinterest[ich]] = 12
        charge_2_spe[channelsofinterest[ich]] = 2159.39 # multiplied 1.2 from ch34 since 35 wasn't cali
    if channelsofinterest[ich] == 36:
        modules[channelsofinterest[ich]]= ["C4"]
        adc_2_spe[channelsofinterest[ich]] = 12
        charge_2_spe[channelsofinterest[ich]] = 2336.25
    if channelsofinterest[ich] == 37:
        modules[channelsofinterest[ich]]= ["C4"]
        adc_2_spe[channelsofinterest[ich]] = 12
        charge_2_spe[channelsofinterest[ich]] = 2197.08
    if channelsofinterest[ich] == 0:
        modules[channelsofinterest[ich]]= ["C5"]
        adc_2_spe[channelsofinterest[ich]] = 18
        charge_2_spe[channelsofinterest[ich]] = 5225.09
    if channelsofinterest[ich] == 2:
        modules[channelsofinterest[ich]]= ["C5"]
        adc_2_spe[channelsofinterest[ich]] = 20
        charge_2_spe[channelsofinterest[ich]] = 5185.5
    if channelsofinterest[ich] == 1:
        modules[channelsofinterest[ich]]= ["C6"]
        adc_2_spe[channelsofinterest[ich]] = 12.5
        charge_2_spe[channelsofinterest[ich]] = 2426.86
    if channelsofinterest[ich] == 3:
        modules[channelsofinterest[ich]]= ["C6"]
        adc_2_spe[channelsofinterest[ich]] = 12
        charge_2_spe[channelsofinterest[ich]] = 2685.4
    if channelsofinterest[ich] == 5:
        modules[channelsofinterest[ich]]= ["C7"]
        adc_2_spe[channelsofinterest[ich]] = 20
        charge_2_spe[channelsofinterest[ich]] = 3831.38
    if channelsofinterest[ich] == 7:
        modules[channelsofinterest[ich]]= ["C7"]
        adc_2_spe[channelsofinterest[ich]] = 18
        charge_2_spe[channelsofinterest[ich]] = 3567.7
    if channelsofinterest[ich] == 4:
        modules[channelsofinterest[ich]]= ["C8"]
        adc_2_spe[channelsofinterest[ich]] = 10
        charge_2_spe[channelsofinterest[ich]] = 1526.98
    if channelsofinterest[ich] == 6:
        modules[channelsofinterest[ich]]= ["C8"]
        adc_2_spe[channelsofinterest[ich]] = 8
        charge_2_spe[channelsofinterest[ich]] = 1258.39

##################################
# 3rd beam period Aug 22
##################################
#++++++++++++++++++++++++++++++++++++
# Pure electrons: all ask for HL trig
#++++++++++++++++++++++++++++++++++++
# run 39110, 0.2 GeV, L - 5bar, H - 10bar, Cu target - EXIST
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039110_cathode/"
# run 39111, 0.3 GeV, L - 5bar, H - 10bar, Cu target - EXIST
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039111_cathode/"
# run 39163, 0.4 GeV, L - 5bar, H - 10bar, Cu target - DOESN"T EXIST

# run 39046, 0.5 GeV, L - 5bar, H - 14bar, Cu target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039046_cathode/"
# run 39136, 0.7 GeV, L - 5bar, H - 8bar, Cu target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039136_cathode/"
# run 38930, 1 GeV, L - 4.5bar, H - 14bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038930_cathode/"
# run 39047, 1.5 GeV, L - 4bar, H - 14bar, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039047_cathode/"
# run 39030, 2 GeV, L - 2bar, H - 10bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039030_cathode/"
# run 39105, 2.5 GeV, L - 1.3bar, H - 10bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039105_cathode/"
# run 39007, 3 GeV, L - 1bar, H - 10bar, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039007_cathode/"
# run 39106, 4 GeV, L - 0.5bar, H - 8bar, Cu target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039106_cathode/"
# run 39108, 5 GeV, L - 0.25bar, H - 5bar, Cu target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039108_cathode/"
# run 39183, 8 GeV, L - 0.1b, H - 1b, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039183_cathode/"


#+++++++++++++++++++++++++++++++++
# Pure pions: all ask for HLx trig
#+++++++++++++++++++++++++++++++++
# run 39047, 1.5 GeV, L - 4bar, H - 14bar, W target (mu) + pi
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039047_cathode/"
# run 39031, share with (K)+P, 2 GeV, L - 2bar, H - 10bar, W target (mu) + pi
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039031_cathode/"
# run 39026, 3 GeV, L - 1.8bar, H - 5bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039026_cathode/"
# run 39027, 6 GeV, L - 0.45bar, H - 5bar, Cu target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039027_cathode/"


#+++++++++++++++++++++++++++++++++
# Pure mu: all ask for HLx trig
#+++++++++++++++++++++++++++++++++
# run 38930 share with pure electron, 1 GeV, L - 4.5bar, H - 14bar, W target
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038930_cathode/"

#+++++++++++++++++++++++++++++++++
# Pure proton: all ask for HxLx trig
#+++++++++++++++++++++++++++++++++
# run 39047, 1.5 GeV, L - 4bar, H - 14bar, W target
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039047_cathode/"
# run 39031, 2 GeV, L - 2bar, H - 10bar, W target  - (K)+P
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039031_cathode/"
# run 39006, 3 GeV, L - 1bar, H - 10bar, W target  - mostly P (+2% K)
dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039006_cathode/"


#----------------------------------------------------------------------------------------------------------------------------------------


##################################
# 2nd beam period
##################################
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
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038563_cathode/"
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

if TotWfmsToRun > len(wfset.waveforms): TotWfmsToRun = len(wfset.waveforms)

# Get of daq trigs
daqstamps = []
#for iwfm in range(len(wfset.waveforms)):
for iwfm in range(TotWfmsToRun):
    if wfset.waveforms[iwfm].daq_window_timestamp in daqstamps: continue
    daqstamps.append(wfset.waveforms[iwfm].daq_window_timestamp)

# of daq trigs must be smaller thsn number of wfms
PE_peak   =  [0] * len(wfset.waveforms)
PE_charge =  [0] * len(wfset.waveforms)

HLPE_peak = [0] * len(daqstamps)
HLxPE_peak = [0] * len(daqstamps)
HxLxPE_peak = [0] * len(daqstamps)

HLPE_charge = [0] * len(daqstamps)
HLxPE_charge = [0] * len(daqstamps)
HxLxPE_charge = [0] * len(daqstamps)

perdaqtrigchannellist = []
for idaq in range(len(daqstamps)):
    # Append sublists
    perdaqtrigchannellist.append([idaq] * 0)

# loop to calculate PE
#for iwfm in range(len(wfset.waveforms)):
for iwfm in range(TotWfmsToRun):
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
    #print("ch: ", channelofinterest, "baseline_ADC:", baseline_ADC)

    # get PE based on adc amplitude
    PE_peak[iwfm] = (peak_adc-baseline_ADC)/adc_2_spe[channelofinterest]
    # get PE based on integrated charge (with baseline subtracted)
    PE_charge[iwfm] = np.sum(wfset.waveforms[iwfm].filtered[qintegrate_start:qintegrate_end]-baseline_ADC)*tick_2_ns/charge_2_spe[channelofinterest]

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

pe_amp_allcathode = [(x) for x in PE_peak]
plt.hist(pe_amp_allcathode, range=(0,4000), bins=20, log=True)
plt.xlabel('PE (amplitude)')
plt.draw()
plt.savefig("./PE_peak_per_wfm_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()

pe_charge_allcathode = [(x) for x in PE_charge]
plt.hist(pe_charge_allcathode, range=(0,18000), bins=20, log=True)
plt.xlabel('PE (charge)')
plt.draw()
plt.savefig("./PE_charge_per_wfm_"+str(wfset.waveforms[0].run_number)+".pdf")
plt.clf() # important to clear figure
plt.close()

# loop to add PE
#for iwfm in range(len(wfset.waveforms)):
for iwfm in range(TotWfmsToRun):
    # daq time stamp index
    index = daqstamps.index(wfset.waveforms[iwfm].daq_window_timestamp)
    #print("iwfm:", iwfm, "daqtime index:", index)

    # SELECT BEAM COINCIDENCE
    if abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < 2000 and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) > 1750:

        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHL':
            # typically electron
            HLPE_peak[index] = HLPE_peak[index] + PE_peak[iwfm]
            HLPE_charge[index] = HLPE_charge[index] + PE_charge[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel) # this needs to have all ch
        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHLx':
            # typically pion
            HLxPE_peak[index] = HLxPE_peak[index] + PE_peak[iwfm]
            HLxPE_charge[index] = HLxPE_charge[index] + PE_charge[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel)
        if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHxLx':
            # k/proton- need ToF
            HxLxPE_peak[index] = HxLxPE_peak[index] + PE_peak[iwfm]
            HxLxPE_charge[index] = HxLxPE_charge[index] + PE_charge[iwfm]
            perdaqtrigchannellist[index].append(wfset.waveforms[iwfm].channel)


final_HLPE_peak = []
final_HLxPE_peak = []
final_HxLxPE_peak = []

final_HLPE_charge = []
final_HLxPE_charge = []
final_HxLxPE_charge = []

for idaq in range(len(daqstamps)):
    if len(perdaqtrigchannellist[idaq]) == 16:
        if HLPE_peak[idaq] != 0: final_HLPE_peak.append(HLPE_peak[idaq])
        if HLxPE_peak[idaq] != 0: final_HLxPE_peak.append(HLxPE_peak[idaq])
        if HxLxPE_peak[idaq] != 0: final_HxLxPE_peak.append(HxLxPE_peak[idaq])

        if HLPE_charge[idaq] != 0: final_HLPE_charge.append(HLPE_charge[idaq])
        if HLxPE_charge[idaq] != 0: final_HLxPE_charge.append(HLxPE_peak[idaq])
        if HxLxPE_charge[idaq] != 0: final_HxLxPE_charge.append(HxLxPE_peak[idaq])


print(" === FINAL REPORT === ")

HLPE_peak_tot = [(x) for x in final_HLPE_peak] # also need to require each channel present
if len(HLPE_peak_tot) > 0:
    counts, bins, _ = plt.hist(HLPE_peak_tot, range=(0,20000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HLPE_peak_tot), np.std(HLPE_peak_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HLPE_peak_mean: ", mean_fit)
    print("HLPE_peak_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HLPE_peak_tot, range=(0,20000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HL trig PE (amplitude)')
    plt.savefig("./HL_totPE_"+str(wfset.waveforms[0].run_number)+"_adcAmplitude.pdf")
    plt.clf() # important to clear figure
    plt.close()

HLxPE_peak_tot = [(x) for x in final_HLxPE_peak]
if len(HLxPE_peak_tot) > 0:
    counts, bins, _ = plt.hist(HLxPE_peak_tot, range=(0,20000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HLxPE_peak_tot), np.std(HLxPE_peak_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HLxPE_peak_mean: ", mean_fit)
    print("HLxPE_peak_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HLxPE_peak_tot, range=(0,20000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HLx trig PE (amplitude)')
    plt.savefig("./HLx_totPE_"+str(wfset.waveforms[0].run_number)+"_adcAmplitude.pdf")
    plt.clf() # important to clear figure
    plt.close()

HxLxPE_peak_tot = [(x) for x in final_HxLxPE_peak]
if len(HxLxPE_peak_tot) > 0:
    counts, bins, _ = plt.hist(HxLxPE_peak_tot, range=(0,20000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HxLxPE_peak_tot), np.std(HxLxPE_peak_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HxLxPE_peak_mean: ", mean_fit)
    print("HxLxPE_peak_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HxLxPE_peak_tot, range=(0,20000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HxLx trig PE (amplitude)')
    plt.savefig("./HxLx_totPE_"+str(wfset.waveforms[0].run_number)+"_adcAmplitude.pdf")
    plt.clf() # important to clear figure
    plt.close()

HLPE_charge_tot = [(x) for x in final_HLPE_charge] # also need to require each channel present
if len(HLPE_charge_tot) > 0:
    counts, bins, _ = plt.hist(HLPE_charge_tot, range=(0,100000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HLPE_charge_tot), np.std(HLPE_charge_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HLPE_charge_mean: ", mean_fit)
    print("HLPE_charge_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HLPE_charge_tot, range=(0,100000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HL trig PE (charge)')
    plt.savefig("./HL_totPE_"+str(wfset.waveforms[0].run_number)+"_IntegrateQ.pdf")
    plt.clf() # important to clear figure
    plt.close()


HLxPE_charge_tot = [(x) for x in final_HLxPE_charge]
if len(HLxPE_charge_tot) > 0:
    counts, bins, _ = plt.hist(HLxPE_charge_tot, range=(0,100000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HLxPE_charge_tot), np.std(HLxPE_charge_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HLxPE_charge_mean: ", mean_fit)
    print("HLxPE_charge_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HLxPE_charge_tot, range=(0,100000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HLx trig PE (charge)')
    plt.savefig("./HLx_totPE_"+str(wfset.waveforms[0].run_number)+"_IntegrateQ.pdf")
    plt.clf() # important to clear figure
    plt.close()

HxLxPE_charge_tot = [(x) for x in final_HxLxPE_charge] # also need to require each channel present
if len(HxLxPE_charge_tot) > 0:
    counts, bins, _ = plt.hist(HxLxPE_charge_tot, range=(0,100000), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=(np.max(counts), statistics.mode(HxLxPE_charge_tot), np.std(HxLxPE_charge_tot)/2), maxfev=1000)
    amplitude_fit, mean_fit, std_dev_fit = params
    print("HxLxPE_charge_mean: ", mean_fit)
    print("HxLxPE_charge_std: ", std_dev_fit)
    fitfunc = gaussian(bin_centers, amplitude_fit, mean_fit, std_dev_fit)
    #plt.hist(HxLxPE_charge_tot, range=(0,100000), bins=100)
    plt.draw()
    plt.plot(bin_centers, fitfunc, 'red', label='Gaussian fit')
    plt.xlabel('HxLx trig PE (charge)')
    plt.savefig("./HxLx_totPE_"+str(wfset.waveforms[0].run_number)+"_IntegrateQ.pdf")
    plt.clf() # important to clear figure
    plt.close()
