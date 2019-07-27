import os
import platform
import math
import pyaudio
import numpy as np
from time import sleep

class sound_gen:
    def __init__(self, tone_start=300, tone_stop=2250, bitrate=180000, length=1.0):
        self.tone_start = tone_start
        self.tone_stop = tone_stop
        self.bitrate = bitrate
        self.length = length
        self.numfr = int(bitrate*length)

    def generate(self, data):
        source_clr = data[:, :-1]*255
        source_clr = source_clr.astype('uint8')
        data[:,:-1] = data[:,:-1]*(self.tone_stop-self.tone_start)
        data[:,:-1] = data[:,:-1]+self.tone_start
        data[:,:-1] = data[:,:-1].astype(int)

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(1), channels=1, rate=self.bitrate, output=True)

        for n, colors in enumerate(data):
            weight = colors[-1]
            for color in colors[:-1]:
                stream.start_stream()
                wdata = ''
                for x in range(int(self.numfr*weight)):
                    wdata += chr(int(128 + 127*math.sin(x/((self.bitrate/color)/math.pi))))
                    
                if platform.system()=='Linux': os.system('clear')
                else: os.system('cls')
                print(
                    'Total: {}/{} \
                    \nColor:  red={} \n\tgreen={} \n\tblue={} \
                    \nPriority {}  \
                    \nFrequences: {}'
                    .format(n+1, len(data),
                            source_clr[n,0], 
                            source_clr[n,1], 
                            source_clr[n,2],
                            data[n, -1], 
                            colors[:-1].astype('uint16'))
                    )
                stream.write(wdata)
                stream.stop_stream()
            sleep(2)
            
        stream.stop_stream()
        stream.close()
        p.terminate()