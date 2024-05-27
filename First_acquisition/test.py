import pyxdf
import mne
import numpy as np
streams, header = pyxdf.load_xdf("parn\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")

stream_channel = 1

raw_data = streams[stream_channel]["time_series"].T #From Steam variable this query is EEG data
(raw_data.shape)

channels = ['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8'] #Set your target EEG channel name
info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*len(channels),
    sfreq= 250 #OpenBCI Frequency acquistion
)
# Create MNE rawarray
raw_mne = mne.io.RawArray(raw_data, info, verbose=False)
raw_mne_dropped = raw_mne.drop_channels(ch_names=['CH1','CH2','CH3','CH4','CH7','CH8'],on_missing='raise')

marker_channel = 0

event_index = streams[marker_channel]["time_series"].T[0] #Get all event marker of experiment 
event_timestamp = streams[marker_channel]["time_stamps"].T #Timestamp when event marked

event_timestamp = [0]
for i in range(1,len(streams[marker_channel]["time_series"])):
    event_timestamp.append( int(streams[marker_channel]["time_stamps"][i])-int(streams[marker_channel]["time_stamps"][0]))
event_timestamp
event_timestamp = np.array(event_timestamp)
event_timestamp = event_timestamp.T

events = np.column_stack((np.array(event_timestamp, dtype = int),
                        np.zeros(len(event_timestamp), dtype = int),
                        np.array(event_index, dtype = int)))
raw_mne_dropped[0]

events_id = { # Set up your event name
    'reset' : -1,
   'baseline' : 0,
   'forward': 1,
   'left': 2,
   'right': 3,
   'backward': 4,
   'break': 5
}


mne_epochs = mne.Epochs(raw_mne_dropped, events, 
        tmin= 0.0,     # init timestamp of epoch (0 means trigger timestamp same as event start)
        tmax= 15.0,    # final timestamp (10 means set epoch duration 10 second)
        event_id =events_id,
     #    reject= dict(eeg = 40e-6),
     #    flat= dict(eeg = 1e-6),
        preload = True,
        baseline = (0.0,0.5),
        event_repeated='drop'
        
        
    )

dropped_log = mne_epochs.plot_drop_log
mne_df = mne_epochs.to_data_frame()

shift = [6,7,8,5]
n = 9
i = 0
L = [n]

while (n <= 136):
    n += shift[i]
    L.append(n)
    i = (i+1)%4
L = L[:-1]

for i in mne_df['epoch'].unique():

    print(i,mne_df[mne_df['epoch'] == i]['condition'].mode()[0])

Idle_epoch = np.array(L)-1
epoch = np.sort(np.concatenate((Idle_epoch,np.array(L))))
Idle_epoch = np.append(Idle_epoch[::5], Idle_epoch[-1])

mne_df = mne_df[mne_df['epoch'].isin(epoch)]

mne_df.loc[mne_df['epoch'].isin(Idle_epoch), 'condition'] = 'idle'



from INTERNET import INTERNET

interNET = INTERNET('model42')
X = mne_df[['CH5','CH6']].values

Z = []
for step in range(2*15*250,3*15*250):
    Z.append(interNET.predict(X[step:step+250])[0])

print(Z)