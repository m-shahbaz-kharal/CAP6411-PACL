import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip

from PIL import Image
import numpy as np
import evaluate

from datasets import load_dataset
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
class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset('sezer12138/ADE20k_Segementation', split='val')
        self.dataset = self.dataset.select_columns(['image', 'annotated'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        seg = self.dataset[idx]['annotated']
        img = preprocess(img)
        seg = np.array(seg)
        seg = torch.from_numpy(seg).unsqueeze(0)
        seg = preprocess_seg(seg)
        return img, seg
    
dataset = ADE20KDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
id2label_dict = {"0": "wall", "1": "building, edifice", "2": "sky", "3": "floor, flooring", "4": "tree", "5": "ceiling", "6": "road, route", "7": "bed", "8": "windowpane, window", "9": "grass", "10": "cabinet", "11": "sidewalk, pavement", "12": "person, individual, someone, somebody, mortal, soul", "13": "earth, ground", "14": "door, double door", "15": "table", "16": "mountain, mount", "17": "plant, flora, plant life", "18": "curtain, drape, drapery, mantle, pall", "19": "chair", "20": "car, auto, automobile, machine, motorcar", "21": "water", "22": "painting, picture", "23": "sofa, couch, lounge", "24": "shelf", "25": "house", "26": "sea", "27": "mirror", "28": "rug, carpet, carpeting", "29": "field", "30": "armchair", "31": "seat", "32": "fence, fencing", "33": "desk", "34": "rock, stone", "35": "wardrobe, closet, press", "36": "lamp", "37": "bathtub, bathing tub, bath, tub", "38": "railing, rail", "39": "cushion", "40": "base, pedestal, stand", "41": "box", "42": "column, pillar", "43": "signboard, sign", "44": "chest of drawers, chest, bureau, dresser", "45": "counter", "46": "sand", "47": "sink", "48": "skyscraper", "49": "fireplace, hearth, open fireplace", "50": "refrigerator, icebox", "51": "grandstand, covered stand", "52": "path", "53": "stairs, steps", "54": "runway", "55": "case, display case, showcase, vitrine", "56": "pool table, billiard table, snooker table", "57": "pillow", "58": "screen door, screen", "59": "stairway, staircase", "60": "river", "61": "bridge, span", "62": "bookcase", "63": "blind, screen", "64": "coffee table, cocktail table", "65": "toilet, can, commode, crapper, pot, potty, stool, throne", "66": "flower", "67": "book", "68": "hill", "69": "bench", "70": "countertop", "71": "stove, kitchen stove, range, kitchen range, cooking stove", "72": "palm, palm tree", "73": "kitchen island", "74": "computer, computing machine, computing device, data processor, electronic computer, information processing system", "75": "swivel chair", "76": "boat", "77": "bar", "78": "arcade machine", "79": "hovel, hut, hutch, shack, shanty", "80": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "81": "towel", "82": "light, light source", "83": "truck, motortruck", "84": "tower", "85": "chandelier, pendant, pendent", "86": "awning, sunshade, sunblind", "87": "streetlight, street lamp", "88": "booth, cubicle, stall, kiosk", "89": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "90": "airplane, aeroplane, plane", "91": "dirt track", "92": "apparel, wearing apparel, dress, clothes", "93": "pole", "94": "land, ground, soil", "95": "bannister, banister, balustrade, balusters, handrail", "96": "escalator, moving staircase, moving stairway", "97": "ottoman, pouf, pouffe, puff, hassock", "98": "bottle", "99": "buffet, counter, sideboard", "100": "poster, posting, placard, notice, bill, card", "101": "stage", "102": "van", "103": "ship", "104": "fountain", "105": "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "106": "canopy", "107": "washer, automatic washer, washing machine", "108": "plaything, toy", "109": "swimming pool, swimming bath, natatorium", "110": "stool", "111": "barrel, cask", "112": "basket, handbasket", "113": "waterfall, falls", "114": "tent, collapsible shelter", "115": "bag", "116": "minibike, motorbike", "117": "cradle", "118": "oven", "119": "ball", "120": "food, solid food", "121": "step, stair", "122": "tank, storage tank", "123": "trade name, brand name, brand, marque", "124": "microwave, microwave oven", "125": "pot, flowerpot", "126": "animal, animate being, beast, brute, creature, fauna", "127": "bicycle, bike, wheel, cycle", "128": "lake", "129": "dishwasher, dish washer, dishwashing machine", "130": "screen, silver screen, projection screen", "131": "blanket, cover", "132": "sculpture", "133": "hood, exhaust hood", "134": "sconce", "135": "vase", "136": "traffic light, traffic signal, stoplight", "137": "tray", "138": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "139": "fan", "140": "pier, wharf, wharfage, dock", "141": "crt screen", "142": "plate", "143": "monitor, monitoring device", "144": "bulletin board, notice board", "145": "shower", "146": "radiator", "147": "glass, drinking glass", "148": "clock", "149": "flag"}
CLASSES = []
for i in range(150): CLASSES.append(id2label_dict[str(i)])
##################################################################
# Eval
##################################################################
model.eval()
preds = []
targets = []

progress = 0
total = len(dataset)

# pre-calculate text features
with torch.no_grad():
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
torch.save(preds, f'eval_result/ade_20k_pred.pt')
targets = torch.vstack(targets)
torch.save(targets, f'eval_result/ade_20k_target.pt')
metric_score = metric.compute(predictions=preds, references=targets, num_labels=len(CLASSES), ignore_index=0)
print(f'Mean IoU {TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}: {metric_score}')
with open(f'eval_result/ade_20k_{TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}.txt', 'w') as f:
    f.write(f'Mean IoU {TRAINED_ON_DATA_TAG}_{TRAINED_ON_EPOCH_TAG}: {metric_score}')