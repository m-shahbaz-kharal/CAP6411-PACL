import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip

from datasets import concatenate_datasets, Dataset
import wandb

BATCH_SIZE = 64
EPOCH = 20

print('Loading model ...')
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) # Must set jit=False for training
checkpoint = torch.load('model_checkpoint/clip_pacl_gcc_2m_first_40k_epoch9.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print('Loading dataset ...')
dataset = concatenate_datasets([Dataset.from_file(file) for file in sorted(glob.glob('data/gcc/labeled_first40k/*.arrow'))])
dataset = dataset.select_columns(['image', 'caption'])
dataset = dataset.with_format('torch')
last_good_img = None
last_good_caption = None
def preprocess_data(item):
    global last_good_img, last_good_caption
    img_list = []
    caption_list = []
    for img, caption in zip(item['image'], item['caption']):
        if img is not None and caption is not None:  
            last_good_img = preprocess(img)
            last_good_caption = clip.tokenize(caption, truncate=True).squeeze(0)
            img_list.append(last_good_img)
            caption_list.append(last_good_caption)
        else:
            img_list.append(last_good_img)
            caption_list.append(last_good_caption)
    item['image'] = img_list
    item['caption'] = caption_list
    return item
dataset.set_transform(preprocess_data)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None: p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

model.eval()
model.visual.joint_space_embedder.train()
optimizer = optim.Adam(model.visual.joint_space_embedder.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# add your own code to track the training progress.
wandb.init(project="clip-patch-level-alignment")
best_loss = 1000
for epoch in range(EPOCH):
    epoch += 10
    for b_number, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images, texts = batch['image'], batch['caption']
        images= images.to(device)
        texts = texts.to(device)
        
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        
        wandb.log({"Loss": total_loss.item()})
        print(f'Epoch {epoch+1} | Batch {b_number+1} | Loss {total_loss.item()}')
        
        total_loss.backward()
        if device == "cpu": optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'joint_space_embedder_state_dict': model.visual.joint_space_embedder.state_dict(), # save only the joint_space_embedder
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
        }, f"model_checkpoint/clip_pacl_gcc_2m_first_40k_epoch{epoch}.pt") #just change to your preferred folder/filename
