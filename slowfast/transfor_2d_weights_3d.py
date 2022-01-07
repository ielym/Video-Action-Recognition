import torch

from fastvision.videoRecognition.models import resnet50_3d

resnet50_weights_path = r'P:\PythonWorkSpace\zoos\resnet50.pth'
resnet50_weights = torch.load(resnet50_weights_path)

model = resnet50_3d(in_channels=3, num_classes=1000, including_top=True)



model_keys = {}
for k, v in model.state_dict().items():
    if not 'temporal' in k:
        model_keys[k] = v

for ori_key, model_key in zip(resnet50_weights.keys(), model_keys.keys()):
    ori_value = resnet50_weights[ori_key]
    if 'spatial' in model_key and model_keys[model_key].size()!=ori_value.size():
        transfor_value = ori_value.unsqueeze(2)
        model_keys[model_key] = transfor_value

model.load_state_dict(model_keys, strict=False)
torch.save(model.state_dict(), "resnet50_3d.pth", _use_new_zipfile_serialization=False)

# print(len(resnet50_weights.keys()), len(model_keys.keys()))