import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip

from PIL import Image
import numpy as np
import evaluate

##################################################################
# Pre-Trained Model Parameters
##################################################################
MODEL_TAG = 'ViT-B/16'
TRAINED_ON_DATA_TAG = 'gcc_2m_first_40k'
TRAINED_ON_EPOCH_TAG = 15
IMG_RES = 224
BATCH_SIZE = 256
patch_size = IMG_RES // int(MODEL_TAG.split('/')[1])
##################################################################
def get_prompt_engineered_text_set(cls):
    set = [
        f'itap of a ({cls}).',
        f'a bad photo of the ({cls}).',
        f'a origami({cls}).',
        f'a photo of the large ({cls}).',
        f'a ({cls}) in a video game.',
        f'art of the ({cls}).',
        f'a photo of the small ({cls}).',
    ]
    return set

def _transform_seg(n_px):
    return Compose([
        Resize(n_px, InterpolationMode.NEAREST),
        CenterCrop(n_px)
    ])

def _bilinear_upscale(n_px): return Compose([Resize(n_px, InterpolationMode.NEAREST)])

metric = evaluate.load('mean_iou')
##################################################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"
##################################################################
print('Loading Model ...')
model, preprocess = clip.load(MODEL_TAG,device=device,jit=False) # must set jit=False for training
preprocess_seg = _transform_seg(model.visual.input_resolution)
postprocess_seg = _bilinear_upscale(model.visual.input_resolution)
checkpoint = torch.load(f'model_checkpoint/clip_pacl_{TRAINED_ON_DATA_TAG}_epoch{TRAINED_ON_EPOCH_TAG}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Model is loaded on {device}.')
##################################################################
# Dataset Class
class PascalContext59Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.seg_dir = os.path.join(data_dir, 'seg')
        self.img_dir = os.path.join(data_dir, 'img')

        self.seg_list = os.listdir(self.seg_dir)
        self.seg_list.sort()

    def __len__(self):
        return len(self.seg_list)

    def __getitem__(self, idx):
        seg_name = self.seg_list[idx]
        img_name = seg_name.replace('.png', '.jpg')
        
        img = Image.open(os.path.join(self.img_dir, img_name))
        seg = Image.open(os.path.join(self.seg_dir, seg_name))
        img = preprocess(img)
        seg = np.array(seg)
        seg = torch.from_numpy(seg).unsqueeze(0)
        seg = preprocess_seg(seg)
        return img, seg
    
dataset = PascalContext59Dataset('data/pc-59')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
CLASSES = []
with open('data/pc-59/59_labels.txt', 'r') as f:
    for line in f:
        if line.strip() != '': CLASSES.append(line.split(':')[1].strip())
##################################################################
# Eval
##################################################################
model.eval()
preds = []
targets = []

progress = 0
total = len(dataset)

# pre-calculate text features
per_class_features = []
for cls_id, cls in enumerate(CLASSES):
    text_set = get_prompt_engineered_text_set(cls)
    text_features = []
    for text in text_set:   
        text_input = clip.tokenize(text).to(device)
        text_features.append(model.encode_text(text_input))
    mean_feature = torch.mean(torch.stack(text_features), dim=0)
    per_class_features.append(mean_feature)
mean_text_features = torch.vstack(per_class_features)

with torch.no_grad():
    for img, seg in dataloader:
        targets.append(seg.squeeze(1))
        img_features = model.encode_image(img.to(device))
        
        patch_level_similarity = img_features @ mean_text_features.T
        normalized_patch_level_similarity = F.softmax(patch_level_similarity, dim=2)
        normalized_patch_level_similarity = normalized_patch_level_similarity.reshape(img.shape[0], patch_size, patch_size, -1)
        pred = normalized_patch_level_similarity.argmax(dim=3)
        pred = postprocess_seg(pred)
        preds.append(pred)
        progress += BATCH_SIZE
        print(f'Progress: {progress}/{total}')

# save & print
os.makedirs('eval_result', exist_ok=True)
preds = torch.vstack(preds)
torch.save(preds, f'eval_result/pc_59_pred.pt')
targets = torch.vstack(targets)
torch.save(targets, f'eval_result/pc_59_target.pt')
metric_score = metric.compute(predictions=preds, references=targets, num_labels=len(CLASSES), ignore_index=0)
print(f'Mean IoU {TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}: {metric_score}')
with open(f'eval_result/pc_59_{TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}.txt', 'w') as f:
    f.write(f'Mean IoU {TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}: {metric_score}')