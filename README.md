# luna16
LUNA16 Lung Nodule Analysis - NWI-IMC037 Final Project 

Respository containing code for our final project of the computer aided medical diagnosis course, which yielded an entry in the LUNA16 competition. 
Unfortunately this project is ill documented, as it was created very quickly and pragmatically as we hurried to meet deadlines for the university course this was the main project of. Running it will likely require minor edits left and right (mostly to do with filepaths), unless you place the data in the same folder as we did. 

A rough list of requirements:

**Python 2.7** with `theano, lasagne (with CUDNN configured), tqdm, pandas, numpy, scipy, scikit-image, scikit-learn, opencv2`.

## UNET for dense prediction

### Preprocessing
Convert data to 1x1x1mm_512x512 slices. A requirement is also a set of segmentations of the lungs (can be found on the LUNA16 website). Place your data in folder `data/original_lungs`:


Use script `src/data_processing/create_same_spacing_data_NODULE.py`, this may take a long time (up to 24 hours) depending on your machine.

Then, download `imagename_zspacing.csv` from [here](https://gzuidhof.stackstorage.com/s/qsqz9dERe7atYU5) and put it in the data folder.

### Unet training
```
src/deep/unet/unet_trainer.py <config>
```

Example conifg `config/unet_splits/split01.ini`. Repeat this for each of the splits.

### Unet held out set dense predictions
```
src/deep/predict <model_name> <epoch>
```
When training a model a  folder is created in `/models`. The folder name is the model name here. Manually look up which epoch had the lowest validation set loss and was a checkpoint.

### Unet dense predictions -> candidates:
This part is likely hard to redo using our code, as it was done by manually editing the scripts. This part is however the easiest conceptually and clearly described in our method description on the Luna competition page. To go from dense prediction to the initial set without postprocessing you can look at `src/candidates.py`. You will have to edit line 30.

## False positive reduction (Wide ResNet)

### Preprocessing:
We performed this on a large cluster (using up to 20 nodes simoultaneously, this may take very long on a single machine). First we equalize the spacings tp 0.5mm*0.5mm*0.5mm using `src/data_processing/equalize_spacings.py`.  Then we create the 96x96 patches in all three orientations using `src/data_processing/create_xy_xz_yz_CARTESIUS.py`.
Scripts to perform this using SLURM for job management (will require some editing of paths) can be found in `/scripts/create_patches/` and `/scripts/rescale`.

### Training
`python train.py <config>` in `/src/deep`  
Configurations: `https://github.com/gzuidhof/luna16/blob/master/config/resnet56_X_diag.ini`  with X 0 through 9.

### Predict

```
src/deep/predict_resnet <model_name> <epoch> <which subsets to predict>
```

The prediction CSV can then be found in the model folder. All you have to do now is combine these. You could use `src/ensembleSubmissions.py` for this, which also features some equalization of predictions of the different models.
