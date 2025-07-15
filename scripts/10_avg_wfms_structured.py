import json
import click
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statistics

from waffles.input_output.hdf5_structured import load_structured_waveformset
import waffles.plotting.drawing_tools as draw

# cathode channels
#channelsofinterest = [32, 31, 34, 36, 2, 1, 5, 4]
# membrane channels: M1-M4 very noisy, shouldn't trust noise count
#channelsofinterest = [45, 42, 44, 41, 0, 20, 30, 10]
channelsofinterest = [20, 30] # if quartz window is M7 (30)
#channelsofinterest = [0, 10] # if quartz window is M8 (10)
#channelsofinterest = [45, 42, 44, 41] # HD style HPK

avf_wfm_max_tick = 800
# beam run
#wfm_peak_adc_low = 3500
#wfm_peak_adc_high = 4000
# LED run mask 8
wfm_peak_adc_low = 25
wfm_peak_adc_high = 40
# beam run
#trg_time_low = 0
#trg_time_high = 100
# LED run
trg_time_low = 220
trg_time_high = 290

filterlength = 10
percentile4baseline = 10
plotwfms = True
nadcthrs = 8 # number of ADC thresholds

avg_wfm = np.zeros((len(channelsofinterest),avf_wfm_max_tick), dtype=np.float32)
countwfms = np.zeros((len(channelsofinterest),), dtype=np.int32)
delta_ADC = np.zeros((len(channelsofinterest),nadcthrs), dtype=np.float32)

modules = ["None"] * 100

for ich in range(len(channelsofinterest)):
    # depends on channel, set at roughly SPE amplitude
    if channelsofinterest[ich] == 30: # SPE ADC is 35 per Jacob
        delta_ADC[ich][0] = 25
        step_size = 30
        modules[channelsofinterest[ich]]= ["M7"]
    if channelsofinterest[ich] == 10:
        delta_ADC[ich][0] = 20
        step_size = 25
        modules[channelsofinterest[ich]]= ["M8"]
    if channelsofinterest[ich] == 0: # SPE ADC is 22-23 per Jacob
        delta_ADC[ich][0] = 20
        step_size = 25
        modules[channelsofinterest[ich]]= ["M5"]
    if channelsofinterest[ich] == 20:
        delta_ADC[ich][0] = 30
        step_size = 35
        modules[channelsofinterest[ich]]= ["M6"]
    if channelsofinterest[ich] == 41: # M4, very noisy, shouldn't trust noise count
        delta_ADC[ich][0] = 30
        step_size = 35
        modules[channelsofinterest[ich]]= ["M4"]
    if channelsofinterest[ich] == 42: # M2, very noisy, shouldn't trust noise count
        delta_ADC[ich][0] = 15
        step_size = 20
        modules[channelsofinterest[ich]]= ["M2"]
    if channelsofinterest[ich] == 44: # M3, very noisy, shouldn't trust noise count
        delta_ADC[ich][0] = 20
        step_size = 25
        modules[channelsofinterest[ich]]= ["M3"]
    if channelsofinterest[ich] == 45: # M1, very noisy, shouldn't trust noise count
        delta_ADC[ich][0] = 20
        step_size = 25
        modules[channelsofinterest[ich]]= ["M1"]
    if channelsofinterest[ich] == 1:
        delta_ADC[ich][0] = 10
        step_size = 15
        modules[channelsofinterest[ich]]= ["C6"]
    if channelsofinterest[ich] == 2:
        delta_ADC[ich][0] = 15
        step_size = 20
        modules[channelsofinterest[ich]]= ["C5"]
    if channelsofinterest[ich] == 4:
        delta_ADC[ich][0] = 5
        step_size = 10
        modules[channelsofinterest[ich]]= ["C8"]
    if channelsofinterest[ich] == 5:
        delta_ADC[ich][0] = 15
        step_size = 20
        modules[channelsofinterest[ich]]= ["C7"]
    if channelsofinterest[ich] == 31:
        delta_ADC[ich][0] = 20
        step_size = 25
        modules[channelsofinterest[ich]]= ["C2"]
    if channelsofinterest[ich] == 32: # SPE is 32 ADC per Jacob analysis Jul 10
        delta_ADC[ich][0] = 27
        step_size = 32
        modules[channelsofinterest[ich]]= ["C1"]
    if channelsofinterest[ich] == 34:
        delta_ADC[ich][0] = 10
        step_size = 15
        modules[channelsofinterest[ich]]= ["C3"]
    if channelsofinterest[ich] == 36:
        delta_ADC[ich][0] = 10
        step_size = 15
        modules[channelsofinterest[ich]]= ["C4"]

    for ithres in range(nadcthrs):
        delta_ADC[ich][ithres] = delta_ADC[ich][0] + step_size*ithres

############################
# Before black blanket cover
# Stored: /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt
############################
# cosmic run, self triggered (before cover NP02)
#filepath="/pnfs/dune/persistent/users/weishi/PDVDNoiseHunt/processed_np02vd_raw_run036362_0000_df-s04-d0_dw_0_20250507T145213.hdf5.copied_structured.hdf5"
# random trigger no led
#filepath="processed_np02vd_raw_run036019_0000_df-s04-d0_dw_0_20250425T090046.hdf5_structured.hdf5"

############################
# After black blanket cover
# Stored: /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt
############################
# RANDOM TRIGGER (Cover PrM fibers + temperature monitor light 1500nm on, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036400_0000_df-s04-d0_dw_0_20250512T121635.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036412_0000_df-s04-d0_dw_0_20250513T142358.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036413_0000_df-s04-d0_dw_0_20250513T144818.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036414_0000_df-s04-d0_dw_0_20250513T145205.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036419_0000_df-s04-d0_dw_0_20250513T151530.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Remove PrM fibers cover + temperature monitor light 1500nm off, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036434_0000_df-s04-d0_dw_0_20250514T084924.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036435_0000_df-s04-d0_dw_0_20250514T085249.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Remove PrM fibers cover + temperature monitor light 1500nm off and fibers unplugged and flange covered, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036445_0000_df-s04-d0_dw_0_20250514T093130.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036449_0000_df-s04-d0_dw_0_20250514T094448.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036463_0000_df-s04-d0_dw_0_20250514T103210.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036457_0000_df-s04-d0_dw_0_20250514T101218.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Manhole not covered)
#filepath="processed_np02vd_raw_run036481_0000_df-s04-d0_dw_0_20250515T085932.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036482_0000_df-s04-d0_dw_0_20250515T090254.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Manhole covered with copper foil and black blankets)
#filepath="processed_np02vd_raw_run036489_0000_df-s04-d0_dw_0_20250515T103041.hdf5.copied_structured.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/precommissioning/check_light_leakage/processed_np02vd_raw_run036498_0000_df-s04-d0_dw_0_20250515T105742.hdf5.copied_structured.hdf5"
# Random trigger: EHN1 lights off
#filepath="processed_np02vd_raw_run036577_0000_df-s04-d0_dw_0_20250519T165114.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036578_0000_df-s04-d0_dw_0_20250519T165430.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036583_0000_df-s04-d0_dw_0_20250519T180550.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036584_0000_df-s04-d0_dw_0_20250519T181700.hdf5.copied_structured.hdf5"
# July 7 LED calib run
filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037066_membrane/processed_np02vd_raw_run037066_0000_df-s04-d0_dw_0_20250707T145032.hdf5.copied_structured_membrane.hdf5"
# July 8, cathode no HV, cathode PD modules off, membrane PD modules only
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037089_membrane/processed_merged_run037089_structured_membrane.hdf5"
# July 8, cathode no HV, cathode + membrane PD modules ON
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037094_cathode/processed_merged_run037094_structured_cathode.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037094_membrane/processed_merged_run037094_structured_membrane.hdf5"
# July 9, cathode no HV, cathode + membrane PD modules ON, LHU2-L5 OFF --> C6 module only operated by L6
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037130_cathode/processed_merged_run037130_structured_cathode.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037130_membrane/processed_merged_run037130_structured_membrane.hdf5"
# July 10, cathode HV @ 154 kV, cathode + membrane PD modules ON, LHU2-L5 OFF --> C6 module only operated by L6
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037210_membrane/processed_merged_run037210_structured_membrane.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037210_cathode/processed_merged_run037210_structured_cathode.hdf5"
# July 10, cathode HV @ 154 kV, ONLY membrane PD modules ON
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037212_membrane/processed_merged_run037212_structured_membrane.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037213_membrane/processed_merged_run037213_structured_membrane.hdf5"
# July 10: first beam run
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037218_membrane/processed_merged_run037218_structured_membrane.hdf5"
#filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/processed/run037218_cathode/processed_merged_run037218_structured_cathode.hdf5"

# Cosmic trigger
# /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt/processed_np02vd_raw_run036401_0000_df-s04-d0_dw_0_20250512T122851.hdf5.copied_structured.hdf5
#filepath="processed_np02vd_raw_run036405_0000_df-s04-d0_dw_0_20250513T123222.hdf5.copied_structured.hdf5" - same condition as cosmic run 36362
#filepath="processed_np02vd_raw_run036401_0000_df-s04-d0_dw_0_20250512T122851.hdf5.copied_structured.hdf5"

wfset = load_structured_waveformset(str(filepath))

print("file path: ", filepath)
print("run number: ", wfset.runs)
print("available channels: ", wfset.available_channels)
print("tot waveforms from all channels: ", len(wfset.waveforms))
print("1st wfm attributes: ", vars(wfset.waveforms[0]))
print("1st wfm adcs: ", wfset.waveforms[0].adcs)
print("1st wfm number of ticks: ", len(wfset.waveforms[0].adcs))
print("1st wfm channel: ", wfset.waveforms[0].channel)
if avf_wfm_max_tick >= len(wfset.waveforms[0].adcs):
    print("ERROR: !!! ************************************** !!!")
    print("ERROR: !!! avf_wfm_max_tick exceeds max wfm ticks !!!")
    print("ERROR: !!! ************************************** !!!")


BaselineADCAllWfms = []

for iwfm in range(len(wfset.waveforms)):

    # requiring PD time stamp and DAQ time stamp within 10 ticks makes sure it's selecting beam event
    #if abs(wfset.waveforms[iwfm].daq_window_timestamp - wfset.waveforms[iwfm].timestamp) < 10:
        # TCO side lower XA closer to cathode, VD style DVDC-DVDM
        if iwfm % 10000 == 0:
            print(iwfm)

        for ich in range(len(channelsofinterest)):
            channelofinterest = channelsofinterest[ich]


            if wfset.waveforms[iwfm].channel == channelofinterest:

                # find baseline of the wfm
                baseline_ADC = 0
                peak_adc = 0

                wfset.waveforms[iwfm].filtered = np.convolve(wfset.waveforms[iwfm].adcs, np.ones(filterlength), 'valid') / filterlength
                # use the mode of adcs of a wfm as baseline
                #baseline_ADC = statistics.mode(wfset.waveforms[iwfm].filtered)
                # use average from certain lowest percentile
                # need x <= since in some wfms has same adcs
                baseline_ADC = statistics.mean(filter(lambda x: x <= np.percentile(wfset.waveforms[iwfm].filtered, percentile4baseline), wfset.waveforms[iwfm].filtered))

                # Fill baseline ADC of all waveforms in the data (a distribution of baselines)
                BaselineADCAllWfms.append(baseline_ADC)

                # Loop over points in the waveform
                for itick in range(avf_wfm_max_tick):
                    # find peak adc in beam trig run
                    if itick < trg_time_high and itick > trg_time_low and wfset.waveforms[iwfm].filtered[itick] - baseline_ADC > peak_adc:
                        peak_adc = wfset.waveforms[iwfm].filtered[itick] - baseline_ADC

                # avg with certain range adcs relative to baseline
                if peak_adc >= wfm_peak_adc_low and peak_adc <= wfm_peak_adc_high:
                    #print("peak_adc: ", peak_adc)
                    countwfms[ich] = countwfms[ich] + 1
                    for itick in range(avf_wfm_max_tick):
                        # sum up wfms for avg later
                        avg_wfm[ich][itick] += wfset.waveforms[iwfm].filtered[itick] - baseline_ADC


                    if iwfm < 1000 and plotwfms == True:
                        # plot the wfm individually
                        xaxis = [x for x in range(len(wfset.waveforms[iwfm].adcs))]
                        plt.plot(xaxis, wfset.waveforms[iwfm].adcs)
                        plt.savefig("plots/"+str(wfset.runs)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_adcs.pdf")
                        plt.clf() # important to clear figure
                        plt.close()

                        xaxis = [x for x in range(len(wfset.waveforms[iwfm].filtered))]
                        plt.plot(xaxis, wfset.waveforms[iwfm].filtered)
                        plt.hlines(y=[baseline_ADC], xmin=0, xmax=len(wfset.waveforms[0].adcs), colors=['r'], linestyles=['--']) # also plot the calculated baseline in red for each wfm
                        plt.savefig("plots/"+str(wfset.runs)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+"_filtered.pdf")
                        plt.clf() # important to clear figure
                        plt.close()


#Baselines_allwfms = [(x) for x in BaselineADCAllWfms]
#plt.hist(Baselines_allwfms, range=(0,10000), bins=1000)
#plt.xlabel('ADC')
#plt.draw()
#plt.savefig("./"+str(wfset.runs)+"_ch_"+str(channelsofinterest[ich])+"_baselines.pdf")
#plt.clf() # important to clear figure
#plt.close()


print("iwfm at end: ", iwfm)
print("run number: ", wfset.runs)
# avg wfm
print("================= Avg Wfm Report =================  ")
for ich in range(len(channelsofinterest)):
    print("==== module ", modules[channelsofinterest[ich]], " (ch ", channelsofinterest[ich], ") ==== ")
    print("tot wfms for avg: ", countwfms[ich])

    # loop over ticks
    for itick in range(avf_wfm_max_tick):
        avg_wfm[ich][itick] = avg_wfm[ich][itick]*1.0 / countwfms[ich]


for ich in range(len(channelsofinterest)):
    xaxis = [x for x in range(len(avg_wfm[ich]))]
    plt.plot(xaxis, avg_wfm[ich])
    plt.savefig("./"+str(wfset.runs)+"_ch_"+str(channelsofinterest[ich])+"_AVG_wfm_filtered.pdf")
    plt.clf() # important to clear figure
    plt.close()
