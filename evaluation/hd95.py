import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd

def compute_hausdorff_distance(y_true, y_pred, spacing):
    '''
    Function to compute the Hausdorff distance for multiple labels.
    Parameters:
        y_true (np.array): ground truth array
        y_pred (np.array): predicted array
        spacing (tuple): voxel spacing
    Returns:
        dict: Hausdorff distance for each label
    '''
    labels = [1]
    hausdorff = {}
    for label in labels:
        true = (y_true == label).astype(np.uint8)
        pred = (y_pred == label).astype(np.uint8)
        
        true_img = sitk.GetImageFromArray(true)
        pred_img = sitk.GetImageFromArray(pred)
        
        # Set the spacing for the images
        true_img.SetSpacing(spacing)
        pred_img.SetSpacing(spacing)
        
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(true_img, pred_img)
        hausdorff[label] = hausdorff_filter.GetHausdorffDistance()
    
    return hausdorff

hausdorff_results = []
with open("your_path/LoRA-PT/Datasets/EADC/test.txt", "r") as file:
    subjects = [line.strip() for line in file.readlines()]

# Iterate over the range of sub directories
for sub in subjects:
    # Path for input files
    mask_path = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_mask.nii.gz"
    pred_path = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_post.nii"

    y_true_nii = nib.load(mask_path)
    y_pred_nii = nib.load(pred_path)

    #spacing = y_true_nii.header.get_zooms()
    #spacing = list(y_true_nii.header.get_zooms())
    spacing = [float(val) for val in y_true_nii.header.get_zooms()]

    y_true = y_true_nii.get_fdata()
    y_pred = y_pred_nii.get_fdata()

    hausdorff_distance = compute_hausdorff_distance(y_true, y_pred, spacing)
    
    # Append the results to our list
    hausdorff_results.append({"Sub": sub, 
                              "Hausdorff Distance Label 1": hausdorff_distance[1]})

# Convert the results to a DataFrame and write it to an Excel file
df_hausdorff = pd.DataFrame(hausdorff_results)
df_hausdorff.to_excel("./hausdorff_distances.xlsx", index=False)
