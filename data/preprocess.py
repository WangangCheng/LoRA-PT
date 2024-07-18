import pickle
import os
import numpy as np
import nibabel as nib

#modalities = ('flair', 't1ce', 't1', 't2')
#modalities = ('t1')

# train
train_set = {
        'root': 'your_path/Datasets/EADC-LPBA40',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': 'your_path/Datasets/EADC-LPBA40',
        'flist': 'valid.txt',
        'has_label': True
        }


test_set = {
        'root': 'your_path/Datasets/EADC-LPBA40',
        'flist': 'test.txt',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data



def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'mask.nii.gz'), dtype='uint8', order='C')

        #images = np.array(nib_load(path + 'img.nii'),dtype='float32',order = 'C')
        #images = np.expand_dims(images, axis=-1)
    #print(nib_load(path + modal + '.nii')
    images =  np.array(nib_load(path  + 'img.nii.gz'), dtype='float32', order='C' )
    images = np.expand_dims(images, axis=-1)
    #images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  #[240,240,155,4]
    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    #x = images[..., 0]  #
    #y = x[mask]

    # 0.8885
    #x[mask] -= y.mean()
    #x[mask] /= y.std()

    #images[..., 0] = x
    for k in range(1):
    
        x = images[..., k]  #
        y = x[mask]
    
         # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()
    
        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    #subjects = open(file_list, encoding='utf-16', errors='ignore').read().splitlines()
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:

        process_f32b0(path, has_label)



if __name__ == '__main__':
    doit(train_set)
    doit(valid_set)
    doit(test_set)
    

