# DiffGuard

Source code of the paper "Privacy enhancing and generalizable deep learning with synthetic data for mediastinal neoplasm diagnosis".

## Environment

**Hardware requirement**

DiffGuard runs on a computer with NVIDIA GPUs. At least 48 GB GPU memory is needed.

**Software requirement**

A Python 3.6+ environment is needed with the packages in the  `requirements.txt` installed.

## Project structure

The main code is put in the `code` folder. The U-Net, nnU-Net and TransUNet models trained on DiffGuard-generated images are put in the `model/segmentation` folder, and the membership inference attack models are put in the `model/membership inference attack` folder. Due to privacy issues, models trained on the real data cannot be released.

## Prepare dataset

This involves the real training data, internal test data and external test data. Prepare real data and a JSON file that describes information about the dataset. The JSON file should contain a list of JSON objects describing the samples with the following keys:

- id: an unique identifier for the sample.
- label: the class label of the sample, should be 'normal', 'thymoma', 'benign cysts', 'germ cell tumor', 'neurogenic tumor' or 'thymic carcinoma'.
- image_path: path to the image, which is a numpy array stored in a `.npy` file or an image stored in `.png` file.
- mask_path: path to the annotation mask, which is a numpy array stored in a `.npy` file.

Save the JSON file into the `data` folder (e.g., `CE_internal_train.json` for contrast-enhanced CT training set and `PL_internal_train.json` for plain CT training set).

## Train DiffGuard models

Set the parameter `--train_datasets`  in `options/DiffGuard_options.py` to the JSON dataset file path.  Choose the settings in `setting/DiffGuard.py`, modify the value of `setting_name` parameter to `DiffGuard` in `main_bash.py` and run the script.

## Synthesize data

When training of DiffGuard model finishes, it can be used to synthesize data by running  `run_DiffGuard.py`. Before running the script, choose the correct setting of `opt.modality` , `opt.data_dir` , `generate_normal_control` and `synthesis_num`. The synthetic samples are saved in `opt.data_dir`.

To use the synthetic data, its JSON dataset file should also be created in the standard format. In the `synthetic data` folder, we provide 1,000 examples of DiffGuard-generated samples, whose corrsponding JSON dataset files are put in the `data` folder.

## Train models

The code supports U-Net and TransUNet. For nnU-Net, we use the official implementation at [MIC-DKFZ/nnUNet (github.com)](https://github.com/MIC-DKFZ/nnUNet).

TransUNet is trained on the basis of the pretrained model `R50+ViT-B_16.npz` on ImageNet-21k, which should be downloaded at [here](https://console.cloud.google.com/storage/vit_models/) and saved into the `pretrained` folder.

Set the parameter `--train_datasets` in `options/seg_options.py` to the JSON dataset file path. Choose the settings in `setting/segmentation.py`, modify the value of `setting_name` parameter to `segmentation` in `main_bash.py` and run the script.

## Predict with models

To predict with the models, choose the settings in `test_segmentation.py` and run the script. By default, the model outputs are saved into folders such as `vis_transunet` and `vis_unet`.

## Membership inference attack

### 1. Train shadow models

Train shadow models on some additional data for the segmentation task. We use the DiffGuard model trained on internal test data to generate synthetic data (e.g., internal_test_generated_50_per_cls.json), and randomly split them into two datasets (e.g., shadow_model_private_data.json, shadow_model_public_data.json). The JSON files should additionally contain a key called `MIA_label` where 1 indicates private data and 0 indicates public data.

### 2. Save predictions of the shadow models

Run `test_shadow_model.py` to save the shadow models' predictions of the training datasets (shadow_model_private_data.json, shadow_model_public_data.json).

### 3. Train attack models

Train a classification model (ResNet-50 by default) to launch the attack. Choose the settings in `setting/membership_inference_attack.py`, modify the value of `setting_name` parameter to `membership_inference_attack` in `main_bash.py` and run the script.

### 4. Test attack models

Before testing the attack models on the real datasets, save the predictions of the victim models (customly trained segmentation models) on the real datasets. Then, choose the settings in  `test_attack.py` and run the script.

## Reference

+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
