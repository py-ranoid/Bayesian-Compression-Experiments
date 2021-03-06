#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities


Karen Ullrich, Oct 2017
"""

import os
import numpy as np
import imageio
import struct
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
cmap = sns.diverging_palette(240, 10, sep=100, as_cmap=True)
from settings import BASE_PATH
# -------------------------------------------------------
# VISUALISATION TOOLS
# -------------------------------------------------------


def visualize_pixel_importance(imgs, log_alpha, epoch="pixel_importance"):
    num_imgs = len(imgs)

    f, ax = plt.subplots(1, num_imgs)
    plt.title("Epoch:" + epoch)
    for i, img in enumerate(imgs):
        print (log_alpha.shape)
        print (img.shape)
        img = (img / 255.) - 0.5
        mask = log_alpha.reshape(img.shape)
        mask = 1 - np.clip(np.exp(mask), 0.0, 1)
        ax[i].imshow(img * mask, cmap=cmap,
                     interpolation='none', vmin=-0.5, vmax=0.5)
        ax[i].grid(False)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
    plt.savefig(BASE_PATH + "plots/pixel" +
                epoch + ".png", bbox_inches='tight')
    plt.close()


def visualise_weights(weight_mus, log_alphas, epoch):
    num_layers = len(weight_mus)

    for i in range(2, num_layers):
        f, ax = plt.subplots(1, 1)
        weight_mu = np.transpose(weight_mus[i].cpu().data.numpy())
        # alpha
        log_alpha_fc1 = log_alphas[i].unsqueeze(1).cpu().data.numpy()
        log_alpha_fc1 = log_alpha_fc1 < -3
        log_alpha_fc2 = log_alphas[i + 1].unsqueeze(0).cpu().data.numpy()
        log_alpha_fc2 = log_alpha_fc2 < -3
        mask = log_alpha_fc1 + log_alpha_fc2
        # weight
        c = np.max(np.abs(weight_mu))
        s = ax.imshow(weight_mu * mask, cmap='seismic',
                      interpolation='none', vmin=-c, vmax=c)
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])
        s.set_clim([-c * 0.5, c * 0.5])
        f.colorbar(s)
        plt.title("Epoch:" + str(epoch))
        plt.savefig(BASE_PATH + "plots/weight" + str(i) + '_e' +
                    str(epoch) + ".png", bbox_inches='tight')
        plt.close()


def generate_gif(save='tmp', epochs=10):
    images = []
    filenames = [BASE_PATH + "plots/" + save + "%d.png" %
                 (epoch + 1) for epoch in np.arange(epochs)]
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
        imageio.mimsave(BASE_PATH + 'figures/' + save +
                        '.gif', images, duration=.5)


def header_gen(files, fname_save):
    header = ''
    num_total = 0
    for fname in files:
        print (fname)
        layer = re.findall('.*lr([0-9])*_.*(wt|bs)', fname)[0]
        with open(fname, 'r') as f:
            contents = f.read().strip()
        nums = [str(np.float32(i)) for i in contents.split('\n')]
        num_total += len(nums)
        le = preprocessing.LabelEncoder()
        le.fit(nums)
        indices = [str(i) for i in le.transform(nums)]
        lookup = [str(i) for i in le.classes_]

        tablename = 'lookup_w_' + \
            layer[0] if layer[1] == 'wt' else 'lookup_b_' + layer[0]
        varname = 'vals_w_' + \
            layer[0] if layer[1] == 'wt' else 'vals_b_' + layer[0]

        table = '{' + ','.join(lookup) + '}'
        array = '{' + ','.join(indices) + '}'

        declaration = 'const float32_t ' + tablename + ' = ' + table
        declaration += '\nconst short ' + varname + ' = ' + array
        header += '\n\n' + declaration
    print (num_total)
    print (fname_save)
    with open(fname_save, 'w') as f:
        f.write(header)
