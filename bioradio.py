import clr
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from livefilter import LiveLFilter
from scipy.signal import iirfilter, find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

path = os.getcwd() + "\\BioRadioSDK.dll"
clr.AddReference(path)

# Import necessary classes from BioRadioSDK
from GLNeuroTech.Devices.BioRadio import BioRadioDeviceManager

# Initialize the device
device_manager = BioRadioDeviceManager()
devices = device_manager.DiscoverBluetoothDevices()
print("devices: ", devices)

# Check if devices are found
if len(devices) == 0:
    print("No BioRadio device found.")
    exit()

device_info = devices[0]
print(f"Connecting to device: {device_info.DeviceId}")
print(device_info.MacId, type(device_info.MacId))
device = device_manager.GetBluetoothDevice(int(device_info.MacId, 16))

# Connect and configure
device.Connect()
device.StartAcquisition()
signals = [signal for sg in device.SignalGroups for signal in sg]

butter_b, butter_a = iirfilter(4, Wn=[5, 20], fs=500, btype="band", ftype="butter")
eeg_names = ['HEAD_R', 'HEAD_L', 'BACK_L', 'BACK_R']
filters = [LiveLFilter(b = butter_b, a = butter_a) if signal.Name in eeg_names else lambda x: x for signal in signals]
rhythms=[
    {'label':'δ','start':1,'end':4},
    {'label':'θ','start':4,'end':8},
    {'label':'α','start':8,'end':14},
    {'label':'β','start':14,'end':30},
    {'label':'γ','start':30,'end':40}
]
rlabels = [r['label'] for r in rhythms]

fig, axes = plt.subplots(4, 4, figsize=(12, 7))  # Adjust rows and columns as needed
plt.subplots_adjust(wspace=0.2,hspace=0.8)
start = time.time()
lines = []
xdata = [[0] for _ in range(10)]
ydata = [[0] for _ in range(10)]
referent = {'HEAD_R':None,'HEAD_L':None,'BACK_L':None,'BACK_R':None,'PULSE':None,'HR':None}
EEG_FREQ = 500
PULSE_FREQ = 250
VISIBLE_WINDOW = 3 # должно быть меньше REFERENT_WINDOW
START_SHIFT = 10
REFERENT_WINDOW = 30
ANALYSYS_WINDOW = 30

for i, signal in enumerate(signals):
    sig_ax = [axes[i][0], axes[i][1], axes[i][2]]  if i < 4 else [axes[i-4][3]]
    sig_ax[0].set_xlim(0, VISIBLE_WINDOW)
    sig_ax[0].set_ylim(-0.01, 0.02)

def find_first_visible(x, window):
    t = len(x) - 1
    while x[-1] - x[t] <= window and t >= 0:
        t -= 1
    return t if t >= 0 else 0 

def fourie(time, sig):
    t = [t - time[0] for t in time]
    T = 1 / 500
    N = int(max(t) / T)
    linTime = np.linspace(0, max(t), N)
    x = rfftfreq(N, T)[1 : int(N // 2)]
    inter_sig = interp1d(t, sig, kind='cubic', fill_value='extrapolate')(linTime)
    y = 2 / N * np.abs(rfft(inter_sig)[1 : int(N // 2)])
    return x, y

def rhythm_ampl(x, y, start, end):
    return np.mean([y[i] for i in range(len(x)) if start <= x[i] <= end])


def update(frame):
    # Retrieve data from the device
    for i, signal in enumerate(signals):
        data = [filters[i](d) for d in signal.GetScaledValueArray()]
        ydata[i].extend(data)
        xdata[i].extend(np.linspace(xdata[i][-1], time.time() - start, len(data) + 1)[1:])
        index = find_first_visible(xdata[i], VISIBLE_WINDOW)

        sig_ax = [axes[i][0], axes[i][1], axes[i][2]]  if i < 4 else [axes[i-4][3]]
        for ax in sig_ax: ax.clear()
        sig_ax[0].set_title(signal.Name)
        if signal.Name in eeg_names:
            sig_ax[1].set_title(signal.Name + ' bars (ref)')
            sig_ax[2].set_title(signal.Name + ' bars')

        color = 'g' if xdata[i][-1] > REFERENT_WINDOW else 'b'
        sig_ax[0].plot(xdata[i][index:], ydata[i][index:], c=color, lw=0.5)

        if xdata[i][-1] > VISIBLE_WINDOW:
            sig_ax[0].set_xlim(xdata[i][-1] - VISIBLE_WINDOW, xdata[i][-1])
            if signal.Name in eeg_names:
                x_f, y_f = fourie(xdata[i][index:], ydata[i][index:])
                # sig_ax[1].plot(x_f, y_f, c=color, lw=0.5)
                # sig_ax[1].set_xlim(5, 45)
                # sig_ax[1].set_ylim(0, max(y_f))
                bars = [rhythm_ampl(x_f, y_f, r['start'], r['end']) for r in rhythms]
                sig_ax[2].bar(rlabels, bars, color=color)
                if len(xdata[i]) > EEG_FREQ*REFERENT_WINDOW:
                    if not referent[signal.Name]:
                        x_f, y_f = fourie(xdata[i][EEG_FREQ*START_SHIFT:EEG_FREQ*REFERENT_WINDOW], ydata[i][EEG_FREQ*START_SHIFT:EEG_FREQ*REFERENT_WINDOW])
                        bars = [rhythm_ampl(x_f, y_f, r['start'], r['end']) for r in rhythms]
                        referent[signal.Name] = bars
                    sig_ax[1].bar(rlabels, referent[signal.Name], color='b')

        try:
            y_min, y_max = np.min(ydata[i][index:]), np.max(ydata[i][index:])
            if y_min and y_max and y_min != y_max:
                middle = (y_max + y_min) / 2
                diff = 1.1 * (y_max - y_min) / 2
                sig_ax[0].set_ylim(middle - diff, middle + diff) 
        except: pass


    axes[3][3].clear()
    axes[3][3].set_xticks([])
    axes[3][3].set_yticks([])
    axes[3][3].spines['top'].set_visible(False)
    axes[3][3].spines['right'].set_visible(False)
    axes[3][3].spines['bottom'].set_visible(False)
    axes[3][3].spines['left'].set_visible(False)
    axes[3][3].set_xlim(0,12)
    axes[3][3].set_ylim(0,12)
    axes[3][3].text(0,10,f'Референс')
    axes[3][3].text(10,10,f'Текущее')
    if len(xdata[4]) > PULSE_FREQ * ANALYSYS_WINDOW:
        axes[3][3].text(10,8,f'ЧСС: {np.mean(ydata[4][-PULSE_FREQ*ANALYSYS_WINDOW:]):.2f}')
    if len(xdata[4]) > PULSE_FREQ * REFERENT_WINDOW:
        if not referent['HR']:
            referent['HR'] = np.mean(ydata[4][PULSE_FREQ*START_SHIFT:PULSE_FREQ*REFERENT_WINDOW])
        axes[3][3].text(0,8,f"ЧСС: {referent['HR']:.2f}")

    if len(xdata[6]) > ANALYSYS_WINDOW * PULSE_FREQ:
        p, _ = find_peaks(ydata[i][-ANALYSYS_WINDOW*PULSE_FREQ:], distance=125)
        sig_ax[0].scatter(np.array(xdata[i][-ANALYSYS_WINDOW*PULSE_FREQ:])[p], np.array(ydata[i][-ANALYSYS_WINDOW*PULSE_FREQ:])[p])
        rr = np.diff(np.array(xdata[i][-ANALYSYS_WINDOW*PULSE_FREQ:])[p])*1000
        N = len(rr)
        M = np.mean(rr)
        SDNN = (sum([(i-M)**2 for i in rr])/(N-1))**0.5
        CV = SDNN/M
        RMSSD = (sum([(rr[i]-rr[i+1])**2 for i in range(N-1)])/(N-1))**0.5
        pNN50 = sum([(1 if abs(rr[i]-rr[i+1])>50 else 0) for i in range(N-1)])/(N-1)
        axes[3][3].text(10,6,f'SDNN: {SDNN:.2f}')
        axes[3][3].text(10,4,f'CV: {CV:.2f}')
        axes[3][3].text(10,2,f'RMSSD: {RMSSD:.2f}')
        axes[3][3].text(10,0,f'pNN50: {pNN50:.2f}')
    if len(xdata[6]) > PULSE_FREQ*REFERENT_WINDOW:
        if not referent['PULSE']:
            referent['PULSE'] = {'SDNN':SDNN, 'CV':CV, 'RMSSD':RMSSD, 'pNN50':pNN50}
        axes[3][3].text(0,6,f"SDNN: {referent['PULSE']['SDNN']:.2f}")
        axes[3][3].text(0,4,f"CV: {referent['PULSE']['CV']:.2f}")
        axes[3][3].text(0,2,f"RMSSD: {referent['PULSE']['RMSSD']:.2f}")
        axes[3][3].text(0,0,f"pNN50: {referent['PULSE']['pNN50']:.2f}")

try:
    ani = FuncAnimation(fig, update, cache_frame_data=False, interval=200)
    plt.show()
except KeyboardInterrupt as e:
    del ani
    device.StopAcquisition()
    device.Disconnect()
