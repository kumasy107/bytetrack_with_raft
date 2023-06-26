import torch
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
# GeForce GTX 1080 Ti
print(torch.cuda.get_device_name(torch.device('cuda:0')))
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_name('cuda:0'))
