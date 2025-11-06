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

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

# cathode channels
#channelsofinterest = [32, 31, 34, 36, 2, 1, 5, 4]
# membrane channels: M1-M4 very noisy, shouldn't trust noise count
#channelsofinterest = [45, 42, 44, 41, 0, 20, 30, 10]
channelsofinterest = [20, 30] # if quartz window is M7 (30)
#channelsofinterest = [0, 10] # if quartz window is M8 (10)
#channelsofinterest = [45, 42, 44, 41] # HD style HPK

Beamrun = True
beam_coincidence_cut = 20 # pd ticks

sampling_rate = 62500000 # PDS tick 16ns, 62.5 M
cutoff_frequency_fft = 10000000 # 10 MHz
FFTGaussFilter = True # Add Gauss Filter to smear fft
# Gauss filter to smear fft
sigma_freq = 5000000 # 5 MHz is a good option, 10MHz doesn't provide enough smear, 1MHz too much smear

filterlength = 10
percentile4baseline = 10

plotwfms = False
plotpersistence = False
SPETemplateHighPassFilter = False
SPETemplateGaussFilter = False
SPETemplateMovingAvg = False
AvgWfmHighPassFilter = False
AvgWfmGaussFilter = False

# July 7 LED calib run: digital compensator off
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037066_membrane/"

##################################
# 3rd beam period Aug 22
##################################
#++++++++++++++++++++++++++++++++++++
# Pure electrons: all ask for HL trig
#++++++++++++++++++++++++++++++++++++
# run 39046, 0.5 GeV, L - 5bar, H - 14bar, Cu target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039046_membrane/"
# run 38930, 1 GeV, L - 4.5bar, H - 14bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038930_membrane/"
# run 39047, 1.5 GeV, L - 4bar, H - 14bar, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039047_membrane/"
# run 39030, 2 GeV, L - 2bar, H - 10bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039030_membrane/"
# run 39105, 2.5 GeV, L - 1.3bar, H - 10bar, W target - exist
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039105_membrane/"
# run 39007, 3 GeV, L - 1bar, H - 10bar, W target - exists
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039007_membrane/"
# run 39106, 4 GeV, L - 0.5bar, H - 8bar, Cu target
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039106_membrane/"
# run 39108, 5 GeV, L - 0.25bar, H - 5bar, Cu target
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039108_membrane/"
# 6 GeV Cu target unknown run #??

#+++++++++++++++++++++++++++++++++
# Pure pions: all ask for HLx trig
#+++++++++++++++++++++++++++++++++
#- NO MEMBRANE PROCESSES FILE
# run 39026, 3 GeV, L - 1.8bar, H - 5bar, W target
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039026_membrane/"
# run 39027, 6 GeV, L - 0.45bar, H - 5bar, Cu target
dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run039027_membrane/"


#----------------------------------------------------------------------------------------------------------------------------------------


##################################
# 2nd beam period
##################################
# 0.5 GeV - HP 14 Bar, LP 5 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038648_membrane/"
# 1 GeV - HP 14 Bar, LP 5 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038645_membrane/"
# 1.5 GeV - HP 14 Bar, LP 4 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038670_membrane/"
# 2 GeV - HP 14 Bar, LP 5 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038651_membrane/"
# 3 GeV - HP 14 Bar, LP 5 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038658_membrane/"
# 4 GeV - HP 14 Bar, LP 1.2 Bar (Good Run) - HLx+HxLx+HLx+HL triggers
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038660_membrane/"

#Aug 5
# run 38563, 2 GeV incluide all particles with High Presion Cherenkov off
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038563_membrane/"
# run 38564, 2 GeV incluide all particles with the optimal Cherenkov configuration
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038564_membrane/"
# run 38565, 2 GeV incluide all particles with the optimal Cherenkov configuration
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run038565_membrane/"


# July 8, cathode no HV, cathode PD modules off, membrane PD modules only
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037089_membrane/processed_merged_run037089_structured_membrane.hdf5"
# July 8, cathode no HV, cathode + membrane PD modules ON
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037094_cathode/processed_merged_run037094_structured_cathode.hdf5"
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037094_membrane/processed_merged_run037094_structured_membrane.hdf5"
# July 9, cathode no HV, cathode + membrane PD modules ON, LHU2-L5 OFF --> C6 module only operated by L6
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037130_cathode/processed_merged_run037130_structured_cathode.hdf5"
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037130_membrane/processed_merged_run037130_structured_membrane.hdf5"
# July 10, cathode HV @ 154 kV, cathode + membrane PD modules ON, LHU2-L5 OFF --> C6 module only operated by L6
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037210_membrane/processed_merged_run037210_structured_membrane.hdf5"
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037210_cathode/processed_merged_run037210_structured_cathode.hdf5"
# July 10, cathode HV @ 154 kV, ONLY membrane PD modules ON
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037212_membrane/processed_merged_run037212_structured_membrane.hdf5"
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037213_membrane/processed_merged_run037213_structured_membrane.hdf5"
# July 10: first beam run +12 GeV
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037218_membrane/processed_merged_run037218_structured_membrane.hdf5"
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037218_cathode/processed_merged_run037218_structured_cathode.hdf5"
# July 13: beam trig +5 GeV
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037249_membrane/processed_np02vd_raw_run037249_0000_df-s05-d0_dw_0_20250713T094009.hdf5.copied_structured_membrane.hdf5"
# July 14: beam run +2 GeV
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037260_membrane/processed_np02vd_raw_run037260_0000_df-s05-d0_dw_0_20250714T164808.hdf5.copied_structured_membrane.hdf5"
# July 15: beam +5 GeV, cathode full stream, + memb, Beam low cherenkov 2.3 bar (normal should be 4 bar), high cherenkov 14 bar
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037275_membrane/processed_np02vd_raw_run037275_0000_df-s05-d0_dw_0_20250715T154533.hdf5.copied_structured_membrane.hdf5"
# July 15: beam +5 GeV, NO cathode PDS, membrane PD only, Beam low cherenkov 2.3 bar (normal should be 4 bar), high cherenkov 14 bar
#dirpath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037276_membrane/processed_np02vd_raw_run037276_0000_df-s05-d0_dw_0_20250715T190106.hdf5.copied_structured_membrane.hdf5"

#####################################
# common setting for high pass filter
#####################################
order = 1

###########################################
# common setting for Gaussian kernel filter
###########################################
sigma = 2 # larger sigma results in more smoothing.

# typical LAr two exponentials
# normalization will be Rs*taus + Rt*taut
def LArTwoTimeConstants(t, Rs, taus, Rt, taut):
    return Rs * np.exp(-1.0*t/taus) + Rt * np.exp(-1.0*t/taut)

# This assumption of only two components and normalize to 1 doesn't necessarily fit
# has to fit to a normalized shape
def LArTwoTimeConstantsPDF(t, Rs, taus, taut):
    return (Rs/taus) * np.exp(-1.0*t/taus) + ((1-Rs)/taut) * np.exp(-1.0*t/taut)

# normalization constant of this func is?
# R* is A* x tau*
def LArThreeTimeConstants(t, Rs, taus, Rt, taut, Rrec, taurec): # per Eq. 1 in arXiv: 2507.08887 and Eur. Phys. J. C (2020) 80:303
    return Rs * np.exp(-1.0*t/taus) + Rt * np.exp(-1.0*t/taut) + Rrec/((1+t/taurec)**2)

def round_to_n_significant_digits(number, n_digits=1):
    """Rounds a number to n significant digits."""
    if number == 0:
        return 0.0

    # Determine the order of magnitude.
    order_of_magnitude = math.floor(math.log10(abs(number)))

    # Calculate the number of decimal places needed.
    # We round to the nearest (n_digits) after shifting the decimal point.
    decimal_places = (n_digits - 1) - order_of_magnitude

    return round(number, decimal_places)

BaselineADCAllWfms = []
daq_pd_dt = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

modules = ["None"] * 100

for ich in range(len(channelsofinterest)):
    # depends on channel, set at roughly SPE amplitude
    if channelsofinterest[ich] == 30:
        modules[channelsofinterest[ich]]= ["M7"]
    if channelsofinterest[ich] == 10:
        modules[channelsofinterest[ich]]= ["M8"]
    if channelsofinterest[ich] == 0:
        modules[channelsofinterest[ich]]= ["M5"]
    if channelsofinterest[ich] == 20:
        modules[channelsofinterest[ich]]= ["M6"]
    if channelsofinterest[ich] == 41:
        modules[channelsofinterest[ich]]= ["M4"]
    if channelsofinterest[ich] == 42:
        modules[channelsofinterest[ich]]= ["M2"]
    if channelsofinterest[ich] == 44:
        modules[channelsofinterest[ich]]= ["M3"]
    if channelsofinterest[ich] == 45:
        modules[channelsofinterest[ich]]= ["M1"]
    if channelsofinterest[ich] == 1:
        modules[channelsofinterest[ich]]= ["C6"]
    if channelsofinterest[ich] == 2:
        modules[channelsofinterest[ich]]= ["C5"]
    if channelsofinterest[ich] == 4:
        modules[channelsofinterest[ich]]= ["C8"]
    if channelsofinterest[ich] == 5:
        modules[channelsofinterest[ich]]= ["C7"]
    if channelsofinterest[ich] == 31:
        modules[channelsofinterest[ich]]= ["C2"]
    if channelsofinterest[ich] == 32:
        modules[channelsofinterest[ich]]= ["C1"]
    if channelsofinterest[ich] == 34:
        modules[channelsofinterest[ich]]= ["C3"]
    if channelsofinterest[ich] == 36:
        modules[channelsofinterest[ich]]= ["C4"]

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
filter_wfm       = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
countwfms        = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HL     = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HLx    = np.zeros((len(channelsofinterest),), dtype=np.int32)
countwfms_HxLx   = np.zeros((len(channelsofinterest),), dtype=np.int32)

# Outer loop to create sublists
for ich in range(len(channelsofinterest)):
    # Append sublists
    daq_pd_dt.append([ich] * 0)

# overlay raw adc waveforms from same trigger event based on daq time
daq_trigger_time = []
count_overlay_plot = 0
for iwfm in range(len(wfset.waveforms)):
    # get the daq time stamp
    if count_overlay_plot > 5: break
    if wfset.waveforms[iwfm].channel == 30 and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < beam_coincidence_cut:
        # this is the list of events we want to check all channels' wfms together
        daq_trigger_time.append(wfset.waveforms[iwfm].daq_window_timestamp)
        count_overlay_plot = count_overlay_plot +1

# Main loop
for iwfm in range(len(wfset.waveforms)):

        if iwfm % 10000 == 0:
            print(iwfm)

        for ich in range(len(channelsofinterest)):
            #print("ich: ", ich)
            channelofinterest = channelsofinterest[ich]

            if wfset.waveforms[iwfm].channel == channelofinterest:

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
                peak_adc = statistics.mode(wfset.waveforms[iwfm].filtered) # CAUTION !!! - this looks for saturated wfm, so mode is used, in principle should use np.max

                # control plot
                daq_pd_dt[ich].append(abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp))

                # control plot: overlay wfms
                if plotpersistence == True:
                    for itick in range(avf_wfm_max_tick):
                        filter_wfm[ich][itick] = wfset.waveforms[iwfm].filtered[itick] - baseline_ADC

                    plt.figure(channelofinterest)
                    if peak_adc > 15000:
                        xaxis = [x for x in range(len(filter_wfm[ich]))]
                        plt.plot(xaxis, filter_wfm[ich])

                # control plot: Overlay raw adc waveforms from same trigger event based on daq time
                for idaqtime in range(len(daq_trigger_time)):
                    if wfset.waveforms[iwfm].daq_window_timestamp == daq_trigger_time[idaqtime] and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < beam_coincidence_cut:
                        # multiple wfm from 1 channel can satisfy this condition
                        # use the min dt????
                        plt.figure(idaqtime)
                        xaxis = [x for x in range(len(wfset.waveforms[iwfm].adcs))]
                        plt.plot(xaxis, wfset.waveforms[iwfm].adcs, color=colors[ich], label=[modules[channelsofinterest[ich]]])

                # MAIN SELECTION
                # For membrane modules: requiring PD time stamp and DAQ time stamp within certain range to make sure it's selecting beam event
                #print("Beamrun: ", Beamrun, "abs dt: ", abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp))
                if (Beamrun == True and abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < beam_coincidence_cut) or Beamrun == False:
                    #print("beamtrue: ", iwfm)

                    # Fill baseline ADC of all waveforms in the data (a distribution of baselines)
                    BaselineADCAllWfms.append(baseline_ADC)

                    countwfms[ich] = countwfms[ich] + 1
                    for itick in range(avf_wfm_max_tick):
                        # sum up wfms for avg later
                        avg_wfm[ich][itick] = avg_wfm[ich][itick] + (wfset.waveforms[iwfm].filtered[itick] - baseline_ADC)
                    # Here for different particles
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHL':
                        countwfms_HL[ich] = countwfms_HL[ich] + 1
                        for itick in range(avf_wfm_max_tick):
                            # typically electron
                            avg_wfm_HL[ich][itick] = avg_wfm_HL[ich][itick] + (wfset.waveforms[iwfm].filtered[itick] - baseline_ADC)
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHLx':
                        # typically pion
                        countwfms_HLx[ich] = countwfms_HLx[ich] + 1
                        for itick in range(avf_wfm_max_tick):
                            avg_wfm_HLx[ich][itick] = avg_wfm_HLx[ich][itick] + (wfset.waveforms[iwfm].filtered[itick] - baseline_ADC)
                    if wfset.waveforms[iwfm].trigger_type_names[0]  == 'kCTBBeamChkvHxLx':
                        # k/proton- need ToF
                        countwfms_HxLx[ich] = countwfms_HxLx[ich] + 1
                        for itick in range(avf_wfm_max_tick):
                            avg_wfm_HxLx[ich][itick] = avg_wfm_HxLx[ich][itick] + (wfset.waveforms[iwfm].filtered[itick] - baseline_ADC)

                    # validation plot for selcted evts
                    if iwfm < 10000 and plotwfms == True:
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


#Baselines_allwfms = [(x) for x in BaselineADCAllWfms]
#plt.hist(Baselines_allwfms, range=(0,10000), bins=1000)
#plt.xlabel('ADC')
#plt.draw()
#plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_baselines.pdf")
#plt.clf() # important to clear figure
#plt.close()

# Save overlay wfms of all ch
for idaqtime in range(len(daq_trigger_time)):
    plt.figure(idaqtime)
    plt.legend(loc="upper right")
    plt.savefig("./overlay_wfm_memb_"+str(wfset.waveforms[0].run_number)+"_daqT_"+str(daq_trigger_time[idaqtime])+".pdf")
    plt.clf() # important to clear figure
    plt.close()

if plotpersistence == True:
    for ich in range(len(channelsofinterest)):
        plt.figure(channelsofinterest[ich])
        plt.savefig("./persistent_big_wfms_memb_"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+".pdf")
        plt.clf() # important to clear figure
        plt.close()

for ich in range(len(channelsofinterest)):
    channelofinterest = channelsofinterest[ich]
    dt_allwfms = [(x) for x in daq_pd_dt[ich]]
    plt.hist(dt_allwfms, range=(0,400000), bins=1000, log=True)
    plt.xlabel('|t_daq - t_pd|')
    plt.draw()
    plt.savefig("./dt_"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fullDAQwindow.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.hist(dt_allwfms, range=(0,500), bins=100, log=True)
    plt.xlabel('|t_daq - t_pd|')
    plt.draw()
    plt.savefig("./dt_"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_zoomin.pdf")
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

# Subtract basline of avg wfm for deconvolve
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

    #################
    # Deconvolution
    #################
    spe_response = np.loadtxt("ch"+str(channelsofinterest[ich])+"_avg_spe_waveform.txt", usecols=0)

    spe_template_final = spe_response
    avg_wfm_final = avg_wfm[ich]
    avg_wfm_HL_final = avg_wfm_HL[ich]
    avg_wfm_HLx_final = avg_wfm_HLx[ich]
    avg_wfm_HxLx_final = avg_wfm_HxLx[ich]

    # method 1: deconvolve - doesn't work out of box
    #source, remainder = signal.deconvolve(avg_wfm[ich], spe_response)

    # method 2: Least square method - works okay but very noisy
    #A = scipy.linalg.convolution_matrix(spe_response, len(avg_wfm[ich]), 'same')
    #A = scipy.linalg.convolution_matrix(spe_template_final, len(avg_wfm_final)+1-len(spe_template_final)) # default full mode
    #source, _, _, _ = scipy.linalg.lstsq(A, avg_wfm_final)

    # method 3: fft with frequency cutoff
    # first pad to same Length
    # Determine the target length (length of the longer signal)
    avg_wfm_length = len(avg_wfm_final)
    avg_wfm_freq = fft(avg_wfm_final)
    avg_wfm_HL_freq = fft(avg_wfm_HL_final)
    avg_wfm_HLx_freq = fft(avg_wfm_HLx_final)
    avg_wfm_HxLx_freq = fft(avg_wfm_HxLx_final)

    # cut high frequency
    frequencies = fftfreq(avg_wfm_length, 1/sampling_rate)
    filter_mask = np.abs(frequencies) <= cutoff_frequency_fft
    # Pad the shorter signal with zeros
    spe_template_final_padded = np.pad(spe_template_final, (0, avg_wfm_length - len(spe_template_final)), 'constant')
    spe_template_freq = fft(spe_template_final_padded)
    # Add a small epsilon to avoid division by zero
    spe_template_freq_safe = spe_template_freq + 1e-10

    # Apply deconvolution and the cutoff filter
    if FFTGaussFilter == True:
        gaussian_filter_freq = np.exp(-0.5 * (frequencies / sigma_freq)**2)
        source_freq_filtered = (avg_wfm_freq * gaussian_filter_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HL = (avg_wfm_HL_freq * gaussian_filter_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HLx = (avg_wfm_HLx_freq * gaussian_filter_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HxLx = (avg_wfm_HxLx_freq * gaussian_filter_freq / spe_template_freq_safe) * filter_mask
    else:
        source_freq_filtered = (avg_wfm_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HL = (avg_wfm_HL_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HLx = (avg_wfm_HLx_freq / spe_template_freq_safe) * filter_mask
        source_freq_filtered_HxLx = (avg_wfm_HxLx_freq / spe_template_freq_safe) * filter_mask

    source = ifft(source_freq_filtered).real #  ifft can have small imaginary components due to numerical precision.
    source_HL = ifft(source_freq_filtered_HL).real
    source_HLx = ifft(source_freq_filtered_HLx).real
    source_HxLx = ifft(source_freq_filtered_HxLx).real

    # plot fft of SPE template
    plt.plot(frequencies, np.abs(spe_template_freq))
    plt.title('Frequency Spectrum of the Avg SPE template')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_SPE_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    # plot fft of Gauss filter
    plt.plot(frequencies, np.abs(gaussian_filter_freq))
    plt.title('Frequency Spectrum of the Gaussian filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_Gaussian_filter_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    # plot fft of avg wfm
    plt.plot(frequencies, np.abs(avg_wfm_freq))
    plt.title('Frequency Spectrum of the Avg Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HL_freq))
    plt.title('Frequency Spectrum of the Avg Cerenkov HL Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_fft_Cerenkov_HL.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HLx_freq))
    plt.title('Frequency Spectrum of the Avg Cerenkov HLx Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_fft_Cerenkov_HLx.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HxLx_freq))
    plt.title('Frequency Spectrum of the Avg Cerenkov HxLx Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_fft_Cerenkov_HxLx.pdf")
    plt.clf() # important to clear figure
    plt.close()
    # plot fft of Gauss filter x wfmfft
    plt.plot(frequencies, np.abs(avg_wfm_freq * gaussian_filter_freq))
    plt.title('Frequency Spectrum of avg signal x Gaussian filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_fft_x_Gaussian_filter_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HL_freq * gaussian_filter_freq))
    plt.title('Frequency Spectrum of Avg Cerenkov HL Signal x Gaussian filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HL_fft_x_Gaussian_filter_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HLx_freq * gaussian_filter_freq))
    plt.title('Frequency Spectrum of Avg Cerenkov HLx Signal x Gaussian filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HLx_fft_x_Gaussian_filter_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()
    plt.plot(frequencies, np.abs(avg_wfm_HxLx_freq * gaussian_filter_freq))
    plt.title('Frequency Spectrum of Avg Cerenkov HxLx Signal x Gaussian filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_Cerenkov_HxLx_fft_x_Gaussian_filter_fft.pdf")
    plt.clf() # important to clear figure
    plt.close()

    # plot deconvolved source
    xaxis = [x for x in range(len(source))]
    plt.plot(xaxis, source)
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_Gaussfilter_"+str(FFTGaussFilter)+".pdf")
    plt.clf() # important to clear figure
    plt.close()
    xaxis = [x for x in range(len(source_HL))]
    plt.plot(xaxis, source_HL)
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_Gaussfilter_"+str(FFTGaussFilter)+"_Cerenkov_HL.pdf")
    plt.clf() # important to clear figure
    plt.close()
    xaxis = [x for x in range(len(source_HLx))]
    plt.plot(xaxis, source_HLx)
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_Gaussfilter_"+str(FFTGaussFilter)+"_Cerenkov_HLx.pdf")
    plt.clf() # important to clear figure
    plt.close()
    xaxis = [x for x in range(len(source_HxLx))]
    plt.plot(xaxis, source_HxLx)
    plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_Gaussfilter_"+str(FFTGaussFilter)+"_Cerenkov_HxLx.pdf")
    plt.clf() # important to clear figure
    plt.close()

    # smooth the deconvolved source
    source_filtered = np.convolve(source, np.ones(filterlength), 'valid') / filterlength
    #print("ch", channelsofinterest[ich], " source filtered: ", source_filtered)
    with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_mvavgsmoothed.txt", "w") as f:
        # loop over ticks
        for itick in range(len(source_filtered)):
            # store avg wfm in txt
            f.write(str(source_filtered[itick])+ "\n")
    # HL trig
    source_HL_filtered = np.convolve(source_HL, np.ones(filterlength), 'valid') / filterlength
    with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_mvavgsmoothed_Cerenkov_HL.txt", "w") as f:
        for itick in range(len(source_HL_filtered)):
            f.write(str(source_HL_filtered[itick])+ "\n")
    # HLx trig
    source_HLx_filtered = np.convolve(source_HLx, np.ones(filterlength), 'valid') / filterlength
    with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_mvavgsmoothed_Cerenkov_HLx.txt", "w") as f:
        for itick in range(len(source_HLx_filtered)):
            f.write(str(source_HLx_filtered[itick])+ "\n")
    # HxLx trig
    source_HxLx_filtered = np.convolve(source_HxLx, np.ones(filterlength), 'valid') / filterlength
    with open(str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_mvavgsmoothed_Cerenkov_HxLx.txt", "w") as f:
        for itick in range(len(source_HxLx_filtered)):
            f.write(str(source_HxLx_filtered[itick])+ "\n")

    #####################################################
    # Fit with standard two exponentials
    #####################################################
    if channelsofinterest[ich] == 20:
        fit_start_tick = 7
    if channelsofinterest[ich] == 30:
        fit_start_tick = 14
    else:
        fit_start_tick = 14

    xaxis = [x for x in range(len(source_filtered))]
    #print("xaxis:", xaxis)

    for isource in range(4):
        if isource == 0:
            source_filtered_final = source_filtered
            Cerenkov_status = "ORofAll"
        if isource == 1:
            source_filtered_final = source_HL_filtered
            Cerenkov_status = "HL"
        if isource == 2:
            source_filtered_final = source_HLx_filtered
            Cerenkov_status = "HLx"
        if isource == 3:
            source_filtered_final = source_HxLx_filtered
            Cerenkov_status = "HxLx"

        xaxis_fit = xaxis[fit_start_tick:]
        source_filtered_fit = source_filtered_final[fit_start_tick:]
        popt, pcov = curve_fit(LArTwoTimeConstants, np.array(xaxis_fit), source_filtered_fit, p0=(0.5,10.0,0.2,200.0), maxfev=5000)
        Rs   = popt[0]
        taus = popt[1]
        Rt   = popt[2]
        taut = popt[3]
        fitresult1 = LArTwoTimeConstants(np.array(xaxis_fit), Rs, taus, Rt, taut)
        residual1 = source_filtered_fit - fitresult1
        chi_squared1 = np.sum(residual1**2)

        # plot filtered source
        plt.plot(xaxis, source_filtered_final, label=str(modules[channelsofinterest[ich]])+': run'+str(wfset.waveforms[0].run_number))
        plt.plot(xaxis_fit, fitresult1, 'red', label=f'Fit f(t)= Rs * exp(-t/taus) + Rt * exp(-t/taut):\n Rs={round_to_n_significant_digits(Rs,2)}, taus={taus:.2f}, \n Rt={round_to_n_significant_digits(Rt,2)}, taut={taut:.2f} [x 16ns], \n chi2 = {round_to_n_significant_digits(chi_squared1,2)}')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_mvavgsmoothed_Cerenkov_"+str(Cerenkov_status)+"_fitwith2T.pdf")
        plt.clf() # important to clear figure
        plt.close()

        # Report fraction of prompt from the fit (assume 2 components)
        print("*** Assume 2 exponentials FIT WELL *** ")
        print("ch ", channelsofinterest[ich], " Cerenkov ", Cerenkov_status, ": A_prompt (abundance) = ", round_to_n_significant_digits((Rs*taus)/((Rs*taus)+(Rt*taut)),2) )

        # Report F_prompt, integrate fit func up to ~6 ticks ~100ns? per PRC 91, 035503 (2015)
        def FitLArTwoTimeConstants(t):
            return Rs * np.exp(-1.0*t/taus) + Rt * np.exp(-1.0*t/taut)
        prompt_integral, prompt_integral_err = quad(FitLArTwoTimeConstants, fit_start_tick, fit_start_tick+6) # integrate 6 ticks
        tot_integral, tot_integral_err = quad(FitLArTwoTimeConstants, fit_start_tick, avf_wfm_max_tick)
        print("ch ", channelsofinterest[ich], " Cerenkov ", Cerenkov_status, ": F_prompt (6 ticks) = ", round_to_n_significant_digits(prompt_integral/tot_integral) )

        #####################################################
        # Fit with three components
        #####################################################
        """popt, pcov = curve_fit(LArThreeTimeConstants, np.array(xaxis_fit), source_filtered_fit, p0=(0.5,10.0,0.2,200.0,1.0,4.0), maxfev=5000)
        Rs   = popt[0]
        taus = popt[1]
        Rt   = popt[2]
        taut = popt[3]
        Rrec = popt[4]
        taurec = popt[5]
        fitresult2 = LArThreeTimeConstants(np.array(xaxis_fit), Rs, taus, Rt, taut, Rrec, taurec)
        residual2 = source_filtered_fit - fitresult2
        chi_squared2 = np.sum(residual2**2)

        # plot filtered source
        plt.plot(xaxis, source_filtered, label=str(modules[channelsofinterest[ich]])+': run'+str(wfset.waveforms[0].run_number))
        plt.plot(xaxis_fit, fitresult2, 'red', label=f'Fit f(t)= Rs * exp(-t/taus) + Rt * exp(-t/taut) + \n Rrec/(1+t/taurec)^2:\n Rs={round_to_n_significant_digits(Rs,2)}, taus={taus:.2f}, \n Rt={round_to_n_significant_digits(Rt,2)}, taut={taut:.2f}, \n Rrec={round_to_n_significant_digits(Rrec,2)}, taurec={taurec:.2f} [x 16ns], chi2 = {round_to_n_significant_digits(chi_squared2,2)}')
        plt.legend(loc='upper right')
        plt.savefig("./"+str(wfset.waveforms[0].run_number)+"_ch_"+str(channelsofinterest[ich])+"_fft_source_smoothed_fitwith3T.pdf")
        plt.clf() # important to clear figure
        plt.close()"""
