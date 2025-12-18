import os
import torch

available_devices = []

if os.environ.get("COLAB_TPU_ADDR"):
    available_devices.append(torch.device('xla'))
if torch.cuda.is_available():
    available_devices.append(torch.device('cuda'))
available_devices.append(torch.device('cpu'))

best_device = available_devices[0]