"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from INTERNET import INTERNET
import time

interNET = INTERNET(sample_rate = 250,control_freq = 10 ,folder_path='model42')

def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    outlet = StreamOutlet(info)
    markernames = ['0','1', '2', '3', '4']

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    mytime = 0
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        # print(timestamp, sample[4:6])
        if(timestamp > mytime):
            result = interNET.predict(sample[4:6])
            outlet.push_sample([markernames[result]])
            mytime += 1/250
        print (result)

if __name__ == '__main__':
    main()
