import os
from torchvision.transforms import CenterCrop, Resize
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

def preprocess(paths):
    base_path, target_path, file = paths
    downsample_size = 32
    img = Image.open(os.path.join(base_path, file))
    crop = CenterCrop(img.width)
    scale = Resize(downsample_size)
    img = scale(crop(img))
    img.save(os.path.join(target_path, file.replace('jpg', 'png')))

if __name__ == '__main__':
    base_path = 'data/celeba/img_align_celeba'
    target_path = 'data/celeba/preprocessed'
    os.makedirs(target_path, exist_ok=True)
    files = os.listdir(base_path)
    paths = [(base_path, target_path, f) for f in files]
    pool = Pool()
    processes = pool.imap_unordered(preprocess, paths)
    iterator = tqdm(processes, total=len(paths))
    for _ in iterator:
        pass
    pool.close()
    pool.join()