import torch
import torch.optim as optim

import time
import argparse
import math
import logging

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import collections

from loader import get_loader, get_data_path
from DSACPose import DSACPose

def setup_logging(name, filename=None):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    if filename is None:
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=filename)
    logger = logging.getLogger(name)
    return logger

def train(args, logger):
	data_loader = get_loader(args.dataset)
	data_path = get_data_path(args.dataset)
	t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_cols, args.img_rows))

	n_classes = t_loader.n_classes
	nw = args.batch_size if args.batch_size > 1 else 0
	trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=nw, shuffle=True)

	# Setup Model
	net = DSACPose(torch.zeros((3)), args.tiny)
	net = net.cuda()
	net.train()

	optimizer = optim.Adam(net.parameters(), lr=args.learningrate)
	iteration = 0
	epochs = int(args.iterations / len(t_loader))

	train_log = open('log_pose_%s_%s.txt' %(args.dataset, args.session), 'w', 1)
	
	training_start = time.time()
	

	for epoch in range(epochs):
		print("=== Epoch: %7d ======================================" % epoch)
		
		for i, (image, lidar, label) in enumerate(trainloader):
			start_time = time.time()

			image = Variable(image.cuda())
			if type(label) == list:
				var_labels = []
				for ii in range(len(label)):
					var_labels.append(Variable(label[ii].cuda()))		
			else:
				var_labels = Variable(label.cuda())

			lidar = (lidar.cuda())
	
	print ("fin")

"""
			# predict scene coordinates
			scene_coordinates = net(image)
			print(image)

			# RGB-D mode
			loss = dsacstar.backward_rgbd(
				scene_coordinates.cpu(), 
				camera_coordinates,
				scene_coordinate_gradients,
				pose, 
				opt.hypotheses, 
				opt.threshold,
				opt.weightrot,
				opt.weighttrans,
				opt.softclamp,
				opt.inlieralpha,
				opt.maxpixelerror,
				random.randint(0,1000000))

			torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
		    optimizer.step()
		    optimizer.zero_grad()
		
		    end_time = time.time()-start_time

			print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, loss, end_time), flush=True)

		    train_log.write('%d %f\n' % (iteration, loss))
		    iteration = iteration + 1

		print('Saving snapshot of the network to %s.' % opt.network_out)
	    torch.save(network.state_dict(), opt.network_out)
	
	print('Done without errors. Time: %.1f minutes.' % ((time.time() - training_start) / 60))
    train_log.close()
"""	

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--arch', nargs='?', type=str, default='DSACPose', 
                        help='')
	parser.add_argument('--dataset', nargs='?', type=str, default='kitti_road', 
                        help='Dataset to use [\'kitti_road\']')
	parser.add_argument('--img_rows', nargs='?', type=int, default=244, 
                        help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=804, 
                        help='Width of the input image')
	
	parser.add_argument('--iterations', '-it', type=int, default=100000, 
	                    help='number of training iterations, i.e. network parameter updates')
	parser.add_argument('--batch_size', nargs='?', type=int, default=4, 
                        help='Batch Size')
	
	parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	                    help='number of hypotheses, i.e. number of RANSAC iterations')
	parser.add_argument('--threshold', '-t', type=float, default=10, 
	                    help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')
						
	parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	                    help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')
	parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	                    help='learning rate')
	
	parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
                  	    help='weight of rotation part of pose loss')
	parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
	                    help='weight of translation part of pose loss')
	parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	                    help='robust square root loss after this threshold')
	parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	                    help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')
	parser.add_argument('--mode', '-m', type=int, default=1, choices=[1,2],
	                    help='test mode: 1 = RGB, 2 = RGB-D')
	parser.add_argument('--tiny', '-tiny', action='store_true',
	                    help='Train a model with massively reduced capacity for a low memory footprint.')
	parser.add_argument('--session', '-sid', default='',
	                    help='custom session name appended to output files. Useful to separate different runs of the program')

	args = parser.parse_args()

	logger = setup_logging(__name__, filename='./' + args.arch + '.out')
	train(args, logger)
	
