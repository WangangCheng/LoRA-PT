# LoRA-PT
LoRA-PT: Low-Rank Adapting UNETR for Hippocampus Segmentation Using Principal Tensor Singular Values and Vectors
[https://export.arxiv.org/abs/2407.11292](https://export.arxiv.org/abs/2407.11292)
![LoRA-PT](images/LoRA-PT.png)

# Contact
if you have any question about our project, please feel free to contact us by email at wgcheng0@gmail.com

# Environment install
Clone this repository and navigate to the root directory of the project.
```python
git clone https://github.com/WangangCheng/LoRA-PT.git
cd LoRA-PT
pip install -r requirements.txt
```
Note: Our cuda version is 12.4, You can install pytorch from this [link](https://pytorch.org/).

# Data downloading
1. You can download the pre-trained weights at this [Link](https://drive.google.com/file/d/1Jtkw2epEYVknGOSGKG0xnQpg4sacEs8j/view?usp=sharing), You can download the pre-trained weights from this link and put them in the `checkpoint/UNETR2024-05-23`.

2. Download our processed EADC data from Baidu Netdisk [Link](https://pan.baidu.com/s/1IRGgkp4BCqcgnv6Ftg0ZNQ?pwd=1111), Or you can also get the source data on the official website [EADC](http://adni.loni.usc.edu/).

3. Download the LPBA40 dataset You can get it from this [Link](https://www.loni.usc.edu/research/atlas_downloads)

4. Download the HFH dataset You can gei it from this [Link](http://www.radiologyresearch.org/HippocampusSegmentationDatabase/)

## Data storage
The data storage location should follow the following method
- Datasets/
  - EADC/
    - inputs/
      - sub1/
        - sub1_img.nii.gz
        - sub1_mask.nii.gz
      - ...
      - sub135/
        - sub135_img.nii.gz
        - sub135_mask.nii.gz 
    - train.txt
    - valid.txt
    - test.txt

Note:The label is not matched the image in th fllowing subjects:002_S_0938 (sub8), 007_S_1304 (sub35), 016_S_4121 (sub65), 029_S_4279 (sub85), 136_S_0429 (sub134).

# Preprocessing
```python
cd preprocess
python preprocess.py
