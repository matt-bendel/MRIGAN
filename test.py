import pickle

import numpy as np
import matplotlib.pyplot as plt

with open(f'/home/bendel.8/Git_Repos/MRIGAN/saved_metrics/loss_kspace_2.pkl', 'rb') as f:
    loss_dict = pickle.load(f)

plt.figure()
plt.plot(np.arange(51), loss_dict['g_loss'])
plt.plot(np.arange(51), loss_dict['d_loss'])
plt.plot(np.arange(51), loss_dict['d_acc'])
plt.savefig('loss.png')
