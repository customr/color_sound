import popcolor
import soundgen

import argparse
import numpy as np
import matplotlib.pyplot as plt


def _main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='images', 
		help="The path to where an images are stored.")
	parser.add_argument('--save_folder', type=str, default='result', 
		help="The path to where an images will be saved.")
	parser.add_argument('--start_freq', type=int, default=300, 
		help="Minimum frequence of sound.")
	parser.add_argument('--stop_freq', type=int, default=2250, 
		help="Maximum frequence of sound.")
	parser.add_argument('--bitrate', type=int, default=180000, 
		help="Bitrate of sound.")
	parser.add_argument('--length', type=float, default=1.0, 
		help="Playback length.")
	parser.add_argument('--image', type=str, 
		help="Image to play")
	args = vars(parser.parse_args())

	sound = soundgen.sound_gen(tone_start=args['start_freq'], 
							   tone_stop=args['stop_freq'], 
							   bitrate=args['bitrate'], 
							   length=args['length'])

	colors = popcolor.vegnn(path=args['path']+'/*', 
    						save_folder=args['save_folder'])

	data = colors[args['image']]/255
	priority = np.linspace(2, 1, data.shape[0]).reshape(-1, 1)
	data = np.hstack((data, priority))
	sound.generate(data)    


if __name__=='__main__':
	_main()