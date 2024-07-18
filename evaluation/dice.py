import nibabel as nib
import SimpleITK as sitk
import numpy as np
from skimage.morphology import remove_small_objects
import pandas as pd

# 1. Read subject IDs from the file
with open("your_path/LoRA-PT/Datasets/EADC/test.txt", "r") as file:
    subjects = [line.strip() for line in file.readlines()]


def remove_small_objs(img_data):
    mask = (img_data == 1)
    cleaned_mask = remove_small_objects(mask, min_size=1000)
    img_data[cleaned_mask == False] = 0
    return img_data

# 2. Modify the loop to iterate over the subjects list
for sub in subjects:
    # Paths
    input_path = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_mask.nii.gz"
    input_file = f"your_path/LoRA-PT/output/submission/UNETR2024-07-12/{sub}.nii"
    output_file = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_pred.nii"

    # Read metadata with SimpleITK
    origin_img = sitk.ReadImage(input_path)
    origin = origin_img.GetOrigin()
    spacing = origin_img.GetSpacing()
    direction = origin_img.GetDirection()

    # Read the predicted nii.gz file
    w = sitk.ReadImage(input_file)
    w.SetOrigin(origin) 
    w.SetSpacing(spacing) 
    w.SetDirection(direction) 

    # Save to the appropriate location after processing
    sitk.WriteImage(w, output_file)

    # Now, remove small objects from the processed image
    # Read image data with nibabel
    img_data = sitk.GetArrayFromImage(w)

    # Remove small objects
    cleaned_data = remove_small_objs(img_data)

    # Convert cleaned data back to SimpleITK Image
    cleaned_img = sitk.GetImageFromArray(cleaned_data.astype(np.uint8))
    cleaned_img.SetOrigin(origin)
    cleaned_img.SetSpacing(spacing)
    cleaned_img.SetDirection(direction)

    # Save the cleaned image to the post path
    post_output_file = f"your_path/LoRA-PT/Datasets/EADC-LPBA40/{sub}/{sub}_post.nii"
    sitk.WriteImage(cleaned_img, post_output_file)

def dice_coefficient(y_true, y_pred):
    ''' 
    Function to calculate the Dice coefficient for multiple labels.
    Parameters:
        y_true (np.array): ground truth array
        y_pred (np.array): predicted array
    Returns:
        dict: Dice coefficient for each label and average
    '''

    # List of unique labels
    labels = [1]

    dice = {}
    for label in labels:
        true = (y_true == label)
        pred = (y_pred == label)
        intersection = np.logical_and(true, pred)

        # Check if the denominator is 0
        denominator = true.sum() + pred.sum()
        if denominator == 0:
            dice[label] = np.nan  # Assign NaN if denominator is 0
        else:
            dice[label] = (2. * intersection.sum()) / denominator
    
    return dice

dice_results = []

# Modify the second loop to iterate over the subjects list
for sub in subjects:
    # Path for input files
    mask_path = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_mask.nii.gz"
    pred_path = f"your_path/LoRA-PT/Datasets/EADC/{sub}/{sub}_post.nii"

    y_true = nib.load(mask_path)
    y_pred = nib.load(pred_path)

    y_true = y_true.get_fdata()
    y_pred = y_pred.get_fdata()

    dice_score = dice_coefficient(y_true, y_pred)
    
    # Append the results to our list
    dice_results.append({"Sub": sub, "Dice Score Label 1": dice_score[1]})

# Convert the results to a DataFrame and write it to an Excel file
df = pd.DataFrame(dice_results)
df.to_excel("./dice_scores.xlsx", index=False)

