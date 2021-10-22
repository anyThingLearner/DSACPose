import torch
import torch.nn as nn
import torch.nn.functional as F

class DSACPose(nn.Module):
	OUTPUT_SUBSAMPLE = 8
	def __init__(self, mean, tiny):
		super(DSACPose, self).__init__()

        # output_dim = 9 # rotation matrix ?
		self.conv1 = nn.Conv2d(4, 32, 3, 1, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
		self.conv4 = nn.Conv2d(128, (256,128)[tiny], 3, 2, 1)

		self.res1_conv1 = nn.Conv2d((256,128)[tiny], (256,128)[tiny], 3, 1, 1)
		self.res1_conv2 = nn.Conv2d((256,128)[tiny], (256,128)[tiny], 1, 1, 0)
		self.res1_conv3 = nn.Conv2d((256,128)[tiny], (256,128)[tiny], 3, 1, 1)

		self.res2_conv1 = nn.Conv2d((256,128)[tiny], (512,128)[tiny], 3, 1, 1)
		self.res2_conv2 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)
		self.res2_conv3 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 3, 1, 1)

		if not tiny:
			self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

		self.res3_conv1 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)
		self.res3_conv2 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)
		self.res3_conv3 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)

		self.fc1 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)
		self.fc2 = nn.Conv2d((512,128)[tiny], (512,128)[tiny], 1, 1, 0)
		self.fc3 = nn.Conv2d((512,128)[tiny], 3, 1, 1, 0)

		# learned scene coordinates relative to a mean coordinate (e.g. center of the scene)
		self.register_buffer('mean', torch.tensor(mean.size()).cuda())
		self.mean = mean.clone()
		self.tiny = tiny
		
	def forward(self, input):
		'''
        Forward pass.
        
        input : 4D data tensor (BxCxHxW)
        '''
		x = input
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		res = F.relu(self.conv4(x))

		x = F.relu(self.res1_conv1(res))
		x = F.relu(self.res1_conv2(x))
		x = F.relu(self.res1_conv3(x))
		
		res = res + x

		x = F.relu(self.res2_conv1(res))
		x = F.relu(self.res2_conv2(x))
		x = F.relu(self.res2_conv3(x))

		if not self.tiny:
			res = self.res2_skip(res)

		res = res + x

		x = F.relu(self.res3_conv1(res))
		x = F.relu(self.res3_conv2(x))
		x = F.relu(self.res3_conv3(x))

		res = res + x		

		sc = F.relu(self.fc1(res))
		sc = F.relu(self.fc2(sc))
		sc = self.fc3(sc)
		
		sc[:, 0] += self.mean[0]
		sc[:, 1] += self.mean[1]
		sc[:, 2] += self.mean[2]

		return sc


if __name__ == '__main__':
    import torch
    net = DSACPose(torch.zeros(3),1)
    print(net)

    input = torch.randn(4, 4, 244, 804)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.shape)