# libraries
from __future__ import print_function
import numpy as np

import sys
import time
import os
import matplotlib.pyplot as plt

BASE_PATH = '/gandiva-store/user1/vgg_exp/' if os.environ.get('GANDIVA_USER') else './'
if os.environ.get('GANDIVA_USER',None):
    log_path = BASE_PATH + 'logs/'
    jobid = os.environ.get('GANDIVA_JOB_ID',"JOB")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_file = open(log_path+'log_'+str(int(time.time()))+'_'+jobid,"w")
    sys.stdout = log_file
    sys.stderr = log_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.autograd import Variable

import BayesianLayers
from compression import compute_compression_rate, compute_reduced_weights
# from utils import visualize_pixel_importance, generate_gif, visualise_weights

vgg_net = models.vgg16_bn(pretrained=True)
pretrained_layers = list(vgg_net.children())[0]
indices = [0,3,7,10,14,17,20,24,27,30,34,37,40]

for i in range(len(indices)):
    print ("LAYER",i)
    print (pretrained_layers[indices[i]].weight.data.max())
    print (pretrained_layers[indices[i]].weight.data.min())
    print (pretrained_layers[indices[i]].bias.data.max())
    print (pretrained_layers[indices[i]].bias.data.min())

N = 60000.  # number of data points in the training set

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    dataset_path = BASE_PATH + FLAGS.dataset + '_data'
    ds = datasets.MNIST if FLAGS.dataset == 'mnist' else datasets.CIFAR10
    # print (dataset_path)
    train_loader = torch.utils.data.DataLoader(
        ds(dataset_path, train=True, download=True,
           transform=transforms.Compose([
               transforms.ToTensor()
           ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ds(dataset_path, train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    # for later analysis we take some sample digits
    unit_shape = (1, 28, 28) if FLAGS.dataset == 'mnist' else (1, 32, 32, 3)
    mask = 255. * (np.ones(unit_shape))
    examples = train_loader.sampler.data_source.train_data[0:5]
    images = np.vstack([mask, examples])

    # build a simple MLP
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # activation
            self.relu = nn.ReLU()
            # layers
            input_channels = 1 if FLAGS.dataset == 'mnist' else 3

            # CONV : 3*3*64 x 2
            self.conv1 = BayesianLayers.Conv2dGroupNJ(
                input_channels, out_channels=64,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0001,
                init_weight=pretrained_layers[indices[0]].weight.data,
                init_bias=pretrained_layers[indices[0]].bias.data)
            # temp = pretrained_layers[indices[0]].weight.data

            self.conv2 = BayesianLayers.Conv2dGroupNJ(
                64, 64, 3, cuda=FLAGS.cuda,clip_var=0.0001,
                init_weight=pretrained_layers[indices[1]].weight.data,
                init_bias=pretrained_layers[indices[1]].bias.data)

            # CONV : 3*3*128 x 2
            self.conv3 = BayesianLayers.Conv2dGroupNJ(
                64, out_channels=128,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0001,
                init_weight=pretrained_layers[indices[2]].weight.data,
                init_bias=pretrained_layers[indices[2]].bias.data)

            self.conv4 = BayesianLayers.Conv2dGroupNJ(
                128, out_channels=128,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0001,
                init_weight=pretrained_layers[indices[3]].weight.data,
                init_bias=pretrained_layers[indices[3]].bias.data)

            # CONV : 3*3*256 x 2
            self.conv5 = BayesianLayers.Conv2dGroupNJ(
                128, out_channels=256,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0004,
                init_weight=pretrained_layers[indices[4]].weight.data,
                init_bias=pretrained_layers[indices[4]].bias.data)

            self.conv6 = BayesianLayers.Conv2dGroupNJ(
                256, out_channels=256,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0004,
                init_weight=pretrained_layers[indices[5]].weight.data,
                init_bias=pretrained_layers[indices[5]].bias.data)

            self.conv7 = BayesianLayers.Conv2dGroupNJ(
                256, out_channels=256,
                kernel_size=3, cuda=FLAGS.cuda,clip_var=0.0004,
                init_weight=pretrained_layers[indices[6]].weight.data,
                init_bias=pretrained_layers[indices[6]].bias.data)

            # CONV : 3*3*512 x 8
            self.conv8 = BayesianLayers.Conv2dGroupNJ(
                256, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[7]].weight.data,
                init_bias=pretrained_layers[indices[7]].bias.data)

            self.conv9 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[8]].weight.data,
                init_bias=pretrained_layers[indices[8]].bias.data
                )

            self.conv10 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[9]].weight.data,
                init_bias=pretrained_layers[indices[9]].bias.data)

            self.conv11 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[10]].weight.data,
                init_bias=pretrained_layers[indices[10]].bias.data)

            self.conv12 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[11]].weight.data,
                init_bias=pretrained_layers[indices[11]].bias.data)

            self.conv13 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda,
                init_weight=pretrained_layers[indices[12]].weight.data,
                init_bias=pretrained_layers[indices[12]].bias.data)

            self.conv14 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda)

            self.conv15 = BayesianLayers.Conv2dGroupNJ(
                512, out_channels=512,
                kernel_size=3, cuda=FLAGS.cuda)

            # num_units_fc1 = 256 if FLAGS.dataset == 'mnist' else 400
            num_units_fc1 = 512

            self.fc1 = BayesianLayers.LinearGroupNJ(
                num_units_fc1, 4096, cuda=FLAGS.cuda)
            self.fc2 = BayesianLayers.LinearGroupNJ(4096, 1000, cuda=FLAGS.cuda)
            self.fc3 = BayesianLayers.LinearGroupNJ(1000, 10, cuda=FLAGS.cuda)

            # self.fc3 = BayesianLayers.LinearGroupNJ(84, 10, cuda=FLAGS.cuda)
            # layers including kl_divergence
            self.kl_list = [self.conv1, self.conv2,
                            self.conv3,self.conv4,
                            self.conv5,self.conv6,self.conv7,
                            self.conv8,self.conv9,self.conv10,self.conv11,
                            self.conv12,self.conv13,self.conv14,self.conv15,
                            self.fc1,self.fc2,self.fc3]
            self.max_vals = {}
            self.min_vals = {}
            self.weightmu_max_vals = {}
            self.weightmu_min_vals = {}
            for i in range(16):
                self.max_vals[i] = []
                self.min_vals[i] = []
                self.weightmu_max_vals[i] = []
                self.weightmu_min_vals[i] = []

        def add_val(self,out,i,layer):
            # return
            self.max_vals[i].append(float(out.max().cpu().data.numpy()))
            self.min_vals[i].append(float(out.min().cpu().data.numpy()))
            self.weightmu_max_vals[i].append(layer.weight_mu.max().cpu().data.numpy())
            self.weightmu_min_vals[i].append(layer.weight_mu.min().cpu().data.numpy())

        def forward(self, x):
            # print (x.shape)
            # print ("INPUT",x.shape,x.max(),x.min())
            # self.add_val(x,13,self.conv1)
            out = F.relu(self.conv1(F.pad(x, (1,1,1,1), mode='replicate')))
            self.add_val(out,0,self.conv1)
            # print (out.shape)
            # print ("LAYER1",out.shape,out.max(),out.min())

            out = F.relu(self.conv2(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,1,self.conv2)
            # print ("LAYER2",out.shape,out.max(),out.min())

            # print (out.shape)
            out = F.max_pool2d(out, 2)
            # print (out.shape)

            out = F.relu(self.conv3(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,2,self.conv3)
            # print (out.shape)
            out = F.relu(self.conv4(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,3,self.conv4)
            # print (out.shape)
            out = F.max_pool2d(out, 2)
            # print (out.shape)

            out = F.relu(self.conv5(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,4,self.conv5)
            # print (out.shape)
            out = F.relu(self.conv6(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,5,self.conv6)
            # print (out.shape)
            out = F.relu(self.conv7(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,6,self.conv7)
            # print (out.shape)
            out = F.max_pool2d(out, 2)
            out = F.sigmoid(out)

            out = F.relu(self.conv8(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,7,self.conv8)
            # print (out.shape)
            out = F.relu(self.conv9(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,8,self.conv9)
            # print (out.shape)
            out = F.relu(self.conv10(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,9,self.conv10)
            # print (out.shape)
            out = F.max_pool2d(out, 2)
            out = F.sigmoid(out)

            out = F.relu(self.conv11(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,10,self.conv11)
            # print (out.shape)
            out = F.relu(self.conv12(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,11,self.conv12)
            # print (out.shape)
            out = F.relu(self.conv13(F.pad(out, (1,1,1,1), mode='replicate')))
            self.add_val(out,12,self.conv13)
            # print (out.shape)
            out = F.max_pool2d(out, 2)

            # out = F.relu(self.conv12(out))
            # print (out.shape)
            # out = F.relu(self.conv13(out))
            # print (out.shape)
            # out = F.relu(self.conv14(out))
            # out = F.relu(self.conv15(out))
            # out = F.max_pool2d(out, 2)

            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            self.add_val(out,13,self.fc1)

            # print ("OUTPUT",out.shape,out.max(),out.min())
            out = F.relu(self.fc2(out))
            self.add_val(out,14,self.fc2)

            out = F.relu(self.fc3(out))
            self.add_val(out,15,self.fc3)
            # out = F.relu(self.fc2(out))
            # out = self.fc2(out)
            return out

        def get_masks(self, thresholds):
            weight_masks = []
            bias_masks = []
            conv_mask = None
            lin_mask = None
            for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
                # compute dropout mask
                if layer.get_type() == 'conv':
                    if conv_mask is None:
                        mask = [True] * layer.in_channels
                    else:
                        mask = np.copy(conv_mask)

                    # print ("CONV:", np.array(mask).shape)
                    log_alpha = layers[i].get_log_dropout_rates(
                    ).cpu().data.numpy()
                    conv_mask = log_alpha < thresholds[i]
                    # print ("CONV-MASK:", conv_mask.shape)
                    # print (layer.bias_mu.shape)

                    # print(np.sum(mask), np.sum(conv_mask))

                    weight_mask = np.expand_dims(
                        mask, axis=0) * np.expand_dims(conv_mask, axis=1)
                    weight_mask = weight_mask[:, :, None, None]
                    bias_mask = conv_mask
                else:
                    if lin_mask is None:
                        mask = conv_mask.repeat(
                            layer.in_features / conv_mask.shape[0])
                    else:
                        mask = np.copy(lin_mask)
                    # print ("LIN:", mask.shape)
                    try:
                        log_alpha = layers[i +
                                           1].get_log_dropout_rates().cpu().data.numpy()
                        lin_mask = log_alpha < thresholds[i + 1]
                    except:
                        # must be the last mask
                        lin_mask = np.ones(10)
                    # print ("LIN-MASK:", lin_mask.shape)
                    # print (layer.bias_mu.shape)
                    # print(np.sum(mask), np.sum(lin_mask))

                    weight_mask = np.expand_dims(
                        mask, axis=0) * np.expand_dims(lin_mask, axis=1)
                    bias_mask = lin_mask

                weight_masks.append(weight_mask.astype(np.float))
                bias_masks.append(bias_mask.astype(np.float))
            return weight_masks, bias_masks

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD

        def plot_vals(self):
            if not os.path.exists(BASE_PATH + 'thresh_plots'):
                os.mkdir(BASE_PATH + 'thresh_plots')
            for lr in range(len(self.max_vals)):
                plt.figure(lr,figsize=(16,8))
                plt.subplot(221)
                plt.xlabel('Iterations')
                plt.ylabel('Max Output')
                plt.plot(self.max_vals[lr])
                plt.subplot(222)
                plt.xlabel('Iterations')
                plt.ylabel('Min Output')
                plt.plot(self.min_vals[lr])
                plt.subplot(223)
                plt.xlabel('Iterations')
                plt.ylabel('Max WeightMu')
                plt.plot(self.weightmu_max_vals[lr])
                plt.subplot(224)
                plt.xlabel('Iterations')
                plt.ylabel('Min WeightMu')
                plt.plot(self.weightmu_min_vals[lr])
                # plt.figure(1)
                plt.savefig(BASE_PATH + 'thresh_plots/maxval_layer%d' %(lr))

            # for lr,max_val in enumerate(self.max_vals):
            #     plt.subplot(212)
            #     plt.plot(t, 2*s1)
            #     plt.plot(np.array(self.max_vals[max_val]))
            #     plt.xlabel('Interations')
            #     plt.ylabel('Max val')
            #
            #     # plt.show()
            #
            # for lr,min_val in enumerate(self.min_vals):
            #     plt.plot(self.min_vals[min_val])
            #     plt.xlabel('Interations')
            #     plt.ylabel('Min val')
            #     plt.savefig(BASE_PATH + 'thresh_plots/minval_layer%d' %(lr))
            #     # plt.show()
            # print (self.max_vals)
            # print (self.min_vals)


    # init model
    model = Net()
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(),lr=0.005,nesterov=False)

    # we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
    discrimination_loss = nn.functional.cross_entropy

    def objective(output, target, kl_divergence):
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + kl_divergence / N
        if FLAGS.cuda:
            variational_bound = variational_bound.cuda()
        return variational_bound

    def train(epoch):
        # model.training = True
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = objective(output, target, model.kl_divergence())
            loss.backward()
            optimizer.step()
            if batch_idx == 60:
                break
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.item()))

    def test():
        model.eval()
        # model.training = False
        test_loss = 0
        correct = 0
        counter = {}
        with torch.no_grad():
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                # print ("output",output.shape,output.max(),output.min())
                test_loss += discrimination_loss(output,
                                                 target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                # print ("pred",pred.shape,pred.max(),pred.min())

                counter[pred] = counter.get(pred,0) + 1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100.0 * float(correct) / len(test_loader.dataset)))
        # print (counter)

    # train the model and save some visualisations on the way
    layers = model.kl_list
    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch)
        test()
    log_alphas = [l.get_log_dropout_rates() for l in layers]

    thresh_path = BASE_PATH + 'thresh_plots'
    if not os.path.exists(thresh_path):
        os.mkdir(thresh_path)

    for lr,lar in enumerate(log_alphas):
        arr = lar.cpu().data.numpy()
        fname = thresh_path + '/vgg_%d_epochs_layer%d.txt' % (FLAGS.epochs, lr)
        np.savetxt(fname,arr)


    # model.plot_vals()
    # exit()
        # visualise_weights(weight_mus, log_alphas, epoch=epoch)
        # log_alpha = model.conv1.get_log_dropout_rates().cpu().data.numpy()
        # visualize_pixel_importance(images,
        #                            log_alpha=log_alpha,
        #                            epoch=str(epoch))

    # generate_gif(save='pixel', epochs=FLAGS.epochs)
    # generate_gif(save='weight2_e'# libraries
    # generate_gif(save='weight3_e', epochs=FLAGS.epochs)

    # compute compression rate and new model accuracy
    # thresholds = FLAGS.thresholds
    # threshold_vals = [[FLAGS.cv1, FLAGS.cv2, FLAGS.fc1, FLAGS.fc2],
    #                   ]
    thresholds = [-1] * 15 + [-2.8, -3., -5.]
    sig, exp = compute_compression_rate(layers, model.get_masks(thresholds))
    print ("Thresholds:",thresholds)
    print("Test error after with reduced bit precision:")

    weights,biases = compute_reduced_weights(layers, model.get_masks(thresholds), sig, exp)
    for layer, weight,bias in zip(layers, weights,biases):
        if FLAGS.cuda:
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
            layer.post_bias_mu = torch.Tensor(bias).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)
    for layer in layers:
        layer.deterministic = True
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--cv1', type=float, default=-0.6)
    parser.add_argument('--cv2', type=float, default=-0.45)
    parser.add_argument('--fc1', type=float, default=-2.8)
    parser.add_argument('--fc2', type=float, default=-3.0)
    parser.add_argument('--fc3', type=float, default=-5.0)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--thresholds', type=float,
                        nargs='*', default=[-1]*15)

    FLAGS = parser.parse_args()

    FLAGS.cuda = torch.cuda.is_available()
    print (FLAGS.cuda)

    main()
