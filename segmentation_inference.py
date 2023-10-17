import torch
import torch.nn.functional as F
import clip

from PIL import Image

##################################################################
# Parameters
##################################################################
CLASSES = ['background', 'dog', 'football', 'person', 'cat']
IMAGE = 'test_image_dog_person_cat.jpg'
IMAGE = f'./input_images/{IMAGE}'
MODEL_TAG = 'ViT-B/16'
DATA_TAG = 'gcc_2m_first_40k'
EPOCH = 'epoch15'
IMG_RES = 224
SEG_THRESHOLD = 0.5
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

################################################################## 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Loading Model ...')
model, preprocess = clip.load(MODEL_TAG,device=device,jit=False) # must set jit=False for training
checkpoint = torch.load(f'model_checkpoint/clip_pacl_{DATA_TAG}_{EPOCH}.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print(f'Model is loaded on {device}.')

# if device == "cpu": model.float()
# else : clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

img_input = preprocess(Image.open(IMAGE)).unsqueeze(0).to(device)
image_features = model.encode_image(img_input)

per_class_features = []
for cls in CLASSES:
    text_set = get_prompt_engineered_text_set(cls)
    text_features = []
    for text in text_set:
        text_input = clip.tokenize(text).to(device)
        text_features.append(model.encode_text(text_input))
    mean_text_features = torch.mean(torch.stack(text_features), dim=0)
    per_class_features.append(mean_text_features)

per_class_features = torch.vstack(per_class_features)

patch_size = IMG_RES // int(MODEL_TAG.split('/')[1])

with torch.no_grad():
    patch_level_similarity = (image_features @ per_class_features.T).reshape(patch_size, patch_size, -1)
    normalized_patch_level_similarity = F.softmax(patch_level_similarity, dim=2)
    # create a picture with the patch colors
    seg = normalized_patch_level_similarity.argmax(dim=2).cpu().numpy()
    seg = (seg * 255 / 3).astype('uint8')
    seg_img = Image.fromarray(seg)
    seg_img = seg_img.resize((IMG_RES,IMG_RES))
    seg_img.save(f'./seg_outputs/segmentation.png')