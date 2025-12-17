import os
import torch

available_devices = []

if os.environ.get("COLAB_TPU_ADDR"):
    available_devices.append(torch.device('tpu'))
if torch.cuda.is_available():
    available_devices.append(torch.device('gpu'))
available_devices.append(torch.device('cpu'))

best_device = available_devices[0]