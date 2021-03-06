# libraries
from __future__ import print_function
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import BayesianLayers
from compression import compute_compression_rate, compute_reduced_weights
# from utils import visualize_pixel_importance, generate_gif, visualise_weights
# from utils import
from settings import BASE_PATH
import struct
N = 60000.  # number of data points in the training set

torch.manual_seed(5)


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def hexify(fname, arr):
    hexvals = []
    for x in arr:
        hexvals.append(float_to_hex(x))
    with open(fname, 'w') as f:
        f.write('\n'.join(hexvals))


def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    dataset_path = BASE_PATH + FLAGS.dataset + '_data'
    ds = datasets.MNIST if FLAGS.dataset == 'mnist' else datasets.CIFAR10
    # print (dataset_path)
    train_loader = torch.utils.data.DataLoader(
        ds(dataset_path, train=True, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(), lambda x: 2 * (x - 0.5),
           ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ds(dataset_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(), lambda x: 2 * (x - 0.5),
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
            self.conv1 = BayesianLayers.Conv2dGroupNJ(
                input_channels, out_channels=20,
                kernel_size=5, stride=1, cuda=FLAGS.cuda)
            self.batch_norm1 = nn.BatchNorm2d(20)
            self.conv2 = BayesianLayers.Conv2dGroupNJ(
                20, out_channels=50,
                kernel_size=5, stride=1, cuda=FLAGS.cuda)
            self.batch_norm2 = nn.BatchNorm2d(50)
            num_units_fc1 = 256 if FLAGS.dataset == 'mnist' else 400
            num_units_fc1 = 800
            self.fc1 = BayesianLayers.LinearGroupNJ(
                num_units_fc1, 500, clip_var=0.04, cuda=FLAGS.cuda)
            # self.batch_norm3 = nn.BatchNorm2d(128)
            self.fc2 = BayesianLayers.LinearGroupNJ(500, 10, cuda=FLAGS.cuda)
            # self.batch_norm1 = nn.BatchNorm2d(20)
            # layers including kl_divergence
            self.kl_list = [self.conv1, self.conv2, self.fc1, self.fc2]

        def forward(self, x):
            # out = self.relu(self.batch_norm1(self.conv1(x)))
            out = self.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            # out = self.relu(self.batch_norm2(self.conv2(out)))
            out = self.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = self.fc2(out)
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

        def get_masks_old(self, thresholds):
            weight_masks = []
            mask = None
            thresholds = [FLAGS.cv1, FLAGS.cv2, FLAGS.fc1, FLAGS.fc2]
            for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
                # compute dropout mask
                if layer.get_type() == 'linear':
                    if mask is None:
                        log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                        mask = log_alpha < threshold
                    else:
                        mask = np.copy(next_mask)

                    try:
                        log_alpha = layers[i +
                                           1].get_log_dropout_rates().cpu().data.numpy()
                        next_mask = log_alpha < thresholds[i + 1]
                    except:
                        # must be the last mask
                        next_mask = np.ones(10)

                    weight_mask = np.expand_dims(
                        mask, axis=0) * np.expand_dims(next_mask, axis=1)
                else:
                    in_ch = layer.in_channels
                    out_ch = layer.out_channels
                    ks = layer.kernel_size[0]

                    log_alpha = layer.get_log_dropout_rates()
                    msk = (log_alpha < threshold).type(torch.FloatTensor)

                    temp = torch.ones(out_ch, in_ch, ks, ks)

                    for k in range(len(msk)):
                        temp[k] = msk[k].expand(in_ch, ks, ks) * temp[k]

                    weight_mask = temp.cpu().data.numpy()

                weight_masks.append(weight_mask.astype(np.float))
            return weight_masks

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD

    # init model
    model = Net()
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters())

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
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.item()))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += discrimination_loss(output,
                                                 target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100.0 * float(correct) / len(test_loader.dataset)))

    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch)
        test()
        # visualizations
        # weight_mus = [model.conv1.weight_mu, model.conv2.weight_mu,
        #               model.fc1.weight_mu]
        log_alphas = [model.conv1.get_log_dropout_rates(), model.conv2.get_log_dropout_rates(),
                      model.fc1.get_log_dropout_rates(), model.fc2.get_log_dropout_rates()]
        #visualise_weights(weight_mus, log_alphas, epoch=epoch)
        # log_alpha = model.fc1.get_log_dropout_rates().cpu().data.numpy()
        # visualize_pixel_importance(images, log_alpha=log_alpha, epoch=str(epoch))

    # generate_gif(save='pixel', epochs=FLAGS.epochs)
    #generate_gif(save='weight2_e', epochs=FLAGS.epochs)
    #generate_gif(save='weight3_e', epochs=FLAGS.epochs)

    lr = 1
    for lar in log_alphas:
        plt.hist(lar.cpu().data.numpy(), bins=30)
        plt.xlabel('Threshold')
        plt.ylabel('# Groups')
        if not os.path.exists(BASE_PATH + 'thresh_plots'):
            os.mkdir(BASE_PATH + 'thresh_plots')
        plt.savefig('thresh_plots/lenet_%d_epochs_layer%d' %
                    (FLAGS.epochs, lr))
        # plt.show()
        plt.close()
        lr += 1

    # compute compression rate and new model accuracy
    layers = [model.conv1, model.conv2, model.fc1, model.fc2]
    thresholds = FLAGS.thresholds
    masks = model.get_masks(thresholds)
    compute_compression_rate(layers, masks)

    print("Test error after with reduced bit precision:")
    weights, biases = compute_reduced_weights(layers, masks, FLAGS.prune)
    if not os.path.exists(BASE_PATH + 'vals'):
        os.mkdir(BASE_PATH + 'vals')
    for i, (layer, weight, bias) in enumerate(zip(layers, weights, biases)):
        new_weight = weight.astype(np.float16).reshape(-1)
        new_bias = bias.astype(np.float16).reshape(-1)

        fname = 'lr' + str(i) + '_ep' + str(FLAGS.epochs) + '_'
        np.savetxt(BASE_PATH + 'vals/' + fname + 'wt.txt', new_weight)
        # hexify(BASE_PATH + 'vals/' + fname + 'wt_hx.txt', new_weight)
        np.savetxt(BASE_PATH + 'vals/' + fname + 'bs.txt', new_bias)
        # hexify(BASE_PATH + 'vals/' + fname + 'bs_hx.txt', new_bias)

        if FLAGS.cuda:
            layer.post_weight_mu = torch.Tensor(weight).cuda()
            layer.post_bias_mu = (torch.Tensor(bias).cuda())
        else:
            layer.post_weight_mu = torch.Tensor(weight)
            layer.post_bias_mu = torch.Tensor(bias)
    for layer in layers:
        layer.deterministic = True
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--cv1', type=float, default=-6.9)
    parser.add_argument('--cv2', type=float, default=-6.5)
    parser.add_argument('--fc1', type=float, default=-4.8)
    parser.add_argument('--fc2', type=float, default=-4)

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--prune', type=int, default=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--thresholds', type=float,
                        nargs='*', default=[-6.9, -6.5, -4.8, -4.])

    FLAGS = parser.parse_args()

    FLAGS.cuda = torch.cuda.is_available()
    print (FLAGS.cuda)

    main()
