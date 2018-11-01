import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib
from torchvision import datasets, transforms

# Download datasets for experiments
parser = argparse.ArgumentParser(description='Download datasets for VSC experiments')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['mnist', 'fashion', 'celebA'],
help='name of dataset to download [mnist, fashion, celebA]')


def download_MNIST( root ):

	#Download MNIST dataset
	datasets.MNIST(root = root, train=True, download=True, transform=transforms.ToTensor())


def download_FASHION ( root ):

	#Download Fashion-MNIST dataset
	datasets.FashionMNIST(root = root, train=True, download=True, transform=transforms.ToTensor())


def download_CELEBA( root ):
  
  #Download CelebA dataset 
  filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
  save_path = os.path.join(root, filename)

  if os.path.exists(save_path):
    print('[*] {} already exists'.format(save_path))
  else:
    download_file_from_google_drive(drive_id, save_path)

  zip_dir = ''
  with zipfile.ZipFile(save_path) as zf:
    zip_dir = zf.namelist()[0]
    zf.extractall(root)
  os.remove(save_path)
  os.rename(os.path.join(root, zip_dir), os.path.join(root, data_dir))


def download(url, dirpath):
  filename = url.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  u = urllib.request.urlopen(url)
  f = open(filepath, 'wb')
  filesize = int(u.headers["Content-Length"])
  print("Downloading: %s Bytes: %s" % (filename, filesize))

  downloaded = 0
  block_sz = 8192
  status_width = 70
  while True:
    buf = u.read(block_sz)
    if not buf:
      print('')
      break
    else:
      print('', end='\r')
    downloaded += len(buf)
    f.write(buf)
    status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
    print(status, end='')
    sys.stdout.flush()
  f.close()
  return filepath

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)


if __name__ == '__main__':

	# Create project folder organization
	os.makedirs('data/mnist', exist_ok=True)
	os.makedirs('data/fashion-mnist', exist_ok=True)
	os.makedirs('data/celeba', exist_ok=True)
	os.makedirs('notebooks', exist_ok=True)
	os.makedirs('src', exist_ok=True)

	# Read datasets to download
	args = parser.parse_args()

	if 'mnist' in args.datasets:
		print('Downloading MNIST dataset...')
		download_MNIST('data/mnist')
	if 'fashion' in args.datasets:
		print('Downloading Fashion-MNIST dataset...')
		download_FASHION('data/fashion-mnist')
	if 'celebA' in args.datasets:
		print('Downloading CelebA dataset...')
		download_CELEBA('data/celeba')

