from datasets import load_dataset
import urllib.request

# conceptual captions
print ("Downloading Conceptual Captions dataset ...")
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
from datasets.utils.file_utils import get_datasets_user_agent
import PIL

USER_AGENT = get_datasets_user_agent()
def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


num_threads = 16
dataset = load_dataset('conceptual_captions', 'labeled', split='train')
dataset = dataset.shard(num_shards=50, index=0)
dataset = dataset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads})
dataset.save_to_disk(f'gcc/labeled_first40k')

# pv-20
print ("Downloading Pascal VOC 2012 dataset ...")
urllib.request.urlretrieve("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", filename="pv-20-images.tar")
os.system("tar -xvf pv-20-images.tar")

# pc-59
print ("Downloading Pascal Context 2010 dataset ...")
urllib.request.urlretrieve("http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar", filename="pc-59-images.tar")
os.system("tar -xvf pc-59-images.tar")
urllib.request.urlretrieve("https://cs.stanford.edu/~roozbeh/pascal-context/59_context_labels.tar.gz", filename="pc-59-labels.tar.gz")
os.system("tar -xvzf pc-59-labels.tar.gz")
urllib.request.urlretrieve("https://cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt", filename="59-labels.txt")

# cs-171
print ("Downloading COCO Seg dataset ...")
urllib.request.urlretrieve("http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip", filename="cs-171-images.zip")
os.system("unzip cs-171-images.zip")
urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", filename="cs-171-images-val.zip")
os.system("unzip cs-171-images-val.zip")
urllib.request.urlretrieve("https://github.com/nightrome/cocostuff/blob/master/labels.txt", filename="cs-171-labels.txt")

# ade-20k
print ("Downloading ADE20K dataset ...")
ade20k = load_dataset('sezer12138/ADE20k_Segementation', split='val')
urllib.request.urlretrieve("https://huggingface.co/datasets/sezer12138/ADE20k_Segementation/blob/main/object_id2label.json", filename="ade-20k-labels.json")

print('Downloading Complete.')
print("Please create the following directory structure under data folder:")
print('''
    cs-171
    ├── img
    │   ├── 000000000139.jpg
    ├── |── ...
    ├── seg
    │   ├── 000000000139.png
    ├── |── ...
    ├── ids2label.txt
    
    pc-59
    ├── img
    │   ├── 2007_000032.jpg
    |── |── ...
    ├── seg
    │   ├── 2007_000032.png
    |── |── ...
    ├── 59_labels.txt
      
    pv-20
    ├── img
    │   ├── 2007_000032.jpg
    |── |── ...
    ├── seg
    │   ├── 2007_000032.png
    |── |── ...
    ├── 20_labels.txt
    |── val.txt -> rename to val_ids.txt
      
    ade-20k
    ├── object_id2label.json
''')

print('Then run the following command:')
print('python eval_on_<dataset_name>.py \t example: python eval_on_pc_59.py')