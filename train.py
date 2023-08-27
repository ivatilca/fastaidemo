
import os 
import torch
import numpy as np
import fastai
from fastai import *
from fastai.vision import *


print("PyTorch version %s" % torch.__version__)
print("fastai version: %s" % fastai.__version__)
print("CUDA supported: %s" % torch.cuda.is_available())
print("CUDNN enabled: %s" % torch.backends.cudnn.enabled)

def download_data():
    """Download and extract the training data."""
    import urllib
    from zipfile import ZipFile
    from pathlib import Path
    # download data
    data_file = './Babies.zip'
    download_url = 'http://p0f.310.myftpupload.com/wp-content/uploads/2020/02/Babies.zip'
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, 'r') as zip:
        print('extracting files...')
        zip.extractall()
        print('finished extracting')
        data_dir = Path('.' + '/Babies')
        
    # delete zip file
    os.remove(data_file)
    return data_dir

path = download_data()

data = ImageDataBunch.from_folder(path,valid_pct=0.05,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy)

learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5,3e-4), pct_start=0.05)

saved_model_path = learn.save('Babies', return_path = True)
learn.export()
saved_model_pkl = str(learn.path) + '/export.pkl'

from azureml.core import Run
run = Run.get_context()

def reduce_list(all_values):
    return [np.max(all_values[i:i+10]) for i in range(0,len(all_values)-1,10)]

losses_values = [tensor.item() for tensor in learn.recorder.losses] 
accuracy_value = np.float(accuracy(*learn.TTA()))

run.log('training_acc', accuracy_value)
run.log('pytorch', torch.__version__)
run.log('fastai', fastai.__version__)
run.log('base_model', 'resnet50')
#run.log_list('Learning_rate', reduce_list(learn.recorder.lrs))
run.log_list('Loss', reduce_list(losses_values))

from shutil import copyfile
copyfile(saved_model_pkl, './outputs/Babies.pkl')