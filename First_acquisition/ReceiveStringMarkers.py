from pylsl import StreamInlet, resolve_stream


def main():

    streams = resolve_stream('type', 'Markers')

    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample[0], timestamp))

if __name__ == '__main__':
    main()
