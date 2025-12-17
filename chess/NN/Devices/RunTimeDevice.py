import os
import torch

available_devices = []

if os.environ.get("COLAB_TPU_ADDR"):
    available_devices.append(torch.device('TPU'))
if torch.cuda.is_available():
    available_devices.append(torch.device('GPU'))
available_devices.append(torch.device('CPU'))

best_device = available_devices[0]