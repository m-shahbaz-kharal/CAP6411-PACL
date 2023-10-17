import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip

from datasets import load_dataset
import time

import wandb


BATCH_SIZE = 64
EPOCH = 10


MODEL_TAG = 'ViT-B/16'

# DATA_TAG = '2m_first_10k'
DATA_TAG = '2m_first_100k'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Loading Model ...')
model, preprocess = clip.load(MODEL_TAG,device=device,jit=False) # must set jit=False for training
print(f'Model is loaded on {device}.')

# dataset = DiffusionDBDataset()
print(f'Creating DataLoader ...')
dataset = load_dataset('poloclub/diffusiondb', DATA_TAG, data_dir='./data')['train']
dataset = dataset.select_columns(['image','prompt'])
def hf_transform(item):
    item['image'] = [preprocess(img) for img in item['image']]
    item['prompt'] = clip.tokenize(item['prompt'], truncate=True)
    return item
dataset.set_transform(hf_transform)
train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE)
print('DataLoader is created.')

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters():
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu": model.float()
else : clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.visual.joint_space_embedder.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

best_loss = 100000

wandb.init(project='clip-patch-level-alignment', config={'batch_size':BATCH_SIZE,'epoch':EPOCH,'model_tag':MODEL_TAG,'data_tag':DATA_TAG})
print(f"Starting Training ...")
for epoch in range(EPOCH):
    epoch_tick = time.time()
    for i, batch in enumerate(train_dataloader) :
        optimizer.zero_grad()

        images, texts = batch['image'], batch['prompt']

        images= images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2

        print(f"[Epoch: {epoch}] [Batch: {i}]: Loss: {total_loss.item()}")
        wandb.log({'Loss': total_loss.item()})

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss},
                f"model_checkpoint/clip_pacl_diffusiondb_{DATA_TAG}_epoch_{epoch}.pt")

        total_loss.backward()
        optimizer.step()

        # if device == "cpu":
        #     optimizer.step()
        # else : 
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     clip.model.convert_weights(model)

    # time in hh:mm:ss
    epoch_tock = time.time()
    epoch_time = epoch_tock - epoch_tick
    print(f"Epoch Took: {epoch_time//3600}:{(epoch_time%3600)//60}:{epoch_time%60}")

wandb.finish()
    