# Watermarking-cvpr2026
Watermarking project for the CVPR 2026 Conference: Code and  Supplementary Materials
Introduction.

The watermarking system introduced here is capable of processing both color and grayscale datasets with image sizes of up to 512x512. The system automatically resizes all input images to 512x512 and attempts to select the best processing route for the type of images served. There are 3 stages to the watermarking process:
1.	Autoencoder training
2.	Naïve classifier training
3.	Watermarking pipeline training and watermark applications
This manual outlines the process of all 3 stages and is tailored for Windows 11 and PowerShell environment.
Dataset preparation.

Your dataset must be ready for autoencoder processing and subsequent classification for the naïve and conditional classifiers. The most common image formats, such as .png and .jpeg, are highly recommended. We will use the Animal Faces High Quality (AFHQ) Dataset to work through the data preparation examples. As an example in this manual, we will set E:\AFHQ_LOWMEM as the dataset home directory.
Data Preparation.
All datasets must have at least one training and one validation directory. The number of classes, and the class names MUST match exactly in both folders. The images must be sorted into subfolders, whose names match the image classes. Therefore, for our AFHD dataset with “cat”, “dog” and “wild” classes, the folder structure is as follows:
Training Directory: E:\AFHQ_LOWMEM\Train
Validation Directory E:\AFHQ_LOWMEM\Val
Inside these folders, there are subfolders- classes. For example, a training folder with cat images has the following path: E:\AFHQ_LOWMEM\Train\cat.  Similarly, a validation folder with dogs would be located at  E:\AFHQ_LOWMEM\Val\dog
It is imperative that the files and subfolders are arranged in exactly this way for the pipeline to properly access the files. There can be as many subfolder classes as necessary: the system will adjust to any number of subfolder classes automatically, as long as there are no subfolders with non-image data in any of the /Train or /Val folders.
1.	Autoencoder launch.

The autoencoder is implemented in AE_universal_Nov2025.py and exposes:


The model class UniversalAutoEncoder:

forward_plain(x) for reconstruction.

External watermark API:

embed_external_wm_rgb(...)

embed_external_wm_gray(...)

Encoder outputs with keys "latent" ([B,1024,32,32]) and "s64" ([B,512,64,64]).

A training loop driven by argparse with a small set of CLI options.
The autoencoder is capable of training with more than one dataset at the same time, at the expense of lower accuracy of reconstruction, with an accuracy drop of 3-5%.  You may train your autoencoder on at most 3 datasets at the same time, and all datasets must be color or grayscale. Combining grayscale and color datasets is not permitted and WILL cause a malfunction. The autoencoder script checks 100 random images for color or grayscale channels. If both color and grayscale images are found, the system will exit with an error, because the system can only process one colorness state at a time. 
For the autoencoder to work properly, you must specify image source folders, and the target folder, where the .pth encoder file should be saved.
Please see the set up process below

At the top of the file, adjust dataset roots and output locations:

AFHQ_TRAIN = r" E:\AFHQ_LOWMEM\Train"  #your training dataset
AFHQ_VAL   = r" E:\AFHQ_LOWMEM\Val" # your validation dataset

DATASET2_TRAIN = r" E:\DATASET2\Train"  #your training dataset
DATASET2_VAL   = r" E:\DATASET\Val" # your validation dataset


TRAIN_ROOT_PAIRS = [
    (AFHQ_TRAIN, AFHQ_VAL),
    (DATASET2_TRAIN, DATASET2_VAL,),
    (None, None), ]
add DATASET2_TRAIN an DATASET2_VAL parameters for an additional dataset you would like to train on more than one dataset at the same time. You can use a maximum of 3 datasets of the same color-ness

SAVE_ROOT = Path(r"E:\Universal_ae_ver2_AFHQ") # destination folder for the .pth file
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = SAVE_ROOT
LOG_DIR  = SAVE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


Meaning:
•	AFHQ_TRAIN, AFHQ_VAL :: train/val roots for your dataset.
•	TRAIN_ROOT_PAIRS :: up to three (train, val) pairs. Unused are left as (None, None).
•	SAVE_ROOT :: base folder for all AE artifacts, home directory of the autoencoder.
•	CKPT_DIR :: directory where checkpoints (*.pth) will be stored.
•	LOG_DIR :: directory with CSV logs of PSNR/SSIM metrics.
If additional datasets are added in the future, they should appear as extra entries in TRAIN_ROOT_PAIRS

Once you set the paths, you are ready to launch the autoencoder. You do not need to pass any command line parameters to lauch the autoencoder with the defaults. The defaults are set as follows:
Batch size: 4
Epochs: 10
Image size: 512
Workers: 4
Learning rate: 1e-4
Use only primary GPU if more than one are available.

However, you may run the encoder from command line with the following parameters:

Key arguments:

--train_dir :: root for training images (ImageFolder).
--val_dir :: root for validation images.
--save_dir :: directory for checkpoints and metrics.
--img_size :: image resolution (set to 512).
--epochs :: number of epochs (e.g. 35).
--batch_size :: batch size (e.g. 32).
--lr :: learning rate (default 1e-4).
--aug :: augmentation strength: none | light | medium | strong.
--devices :: optional CUDA_VISIBLE_DEVICES string.
--num_workers :: dataloader workers.
--log_every :: logging frequency.
--amp :: AMP mode: none | fp16 | bf16.
--classes_txt :: optional text file with a fixed class list to enforce consistent ordering.

The typical command to start an autoencoder with custom parameters would be:

python AE_universal_Nov2025.py ^
  --epochs 10 ^
  --bs 4 ^
  --size 512 ^
  --workers 4 ^
  --lr 1e-4

Make sure the selected GPU index and batch size fit your hardware.

Once the autoencoder starts, follow the instructions on the screen and wait for the training completion. Once the training completes, you will find an appropriate .pth file with the _best postfix in the filename for the model. Please do not forget to change the destination directory each time you change your dataset in order to avoid overwriting previously trained models. 

2.	Naïve classifier pre-training

The C1 training pipeline is implemented in C1_agnostic_trainer_CM.py

It trains a ResNet-34 classifier with:
•	Grayscale input (internally converting RGB → Y).
•	Optional mixup and label smoothing.
•	Per-epoch train/val confusion matrices saved to CSV.
•	Dual export:
o	1-channel checkpoint for grayscale input.
o	3-channel “RGB-wrapped” checkpoint that accepts normalized RGB in [-1,1].
C1 is later used by the watermark trainer as a frozen guard model (C1_CKPT).
Key arguments:
•	--train_dir :: root for training images (ImageFolder).
•	--val_dir :: root for validation images.
•	--save_dir :: directory for checkpoints and metrics.
•	--img_size :: image resolution (set to 512).
•	--epochs :: number of epochs (e.g. 35).
•	--batch_size :: batch size (e.g. 32).
•	--lr :: learning rate (default 1e-4).
•	--aug :: augmentation strength: none | light | medium | strong.
•	--devices :: optional CUDA_VISIBLE_DEVICES string.
•	--num_workers :: dataloader workers.
•	--log_every :: logging frequency.
•	--amp :: AMP mode: none | fp16 | bf16.
•	--classes_txt :: optional text file with a fixed class list to enforce consistent ordering.
Example command matching the AFHQ setup:

python C1_agnostic_trainer_RGB_BW.py ^
  --train_dir E:\AFHQ_LOWMEM\Train^
  --val_dir   E:\AFHQ_LOWMEM\Val ^
  --save_dir  E:\ AFHQ_LOWMEM \C1_RS34 ^
  --img_size  512 ^
  --epochs    35 ^
  --batch_size 32 ^
  --lr 1e-4 ^
  --aug medium ^
  --amp fp16


The script automatically scans the dataset to detect whether it is RGB or grayscale and then converts everything to a tensor normalized to [-1,1].

Training may take a substantial amount of time. Upon completion, you will obtain the following:
Inside --save_dir you will obtain:
•	C1_CKPT = r"E:\AFHQ_LOWMEM\C1_RS34\best_raw.pth"
•	best_raw.pth :: 1-channel C1 (ResNet34Gray).
Used by the grayscale watermark trainer via the C1_CKPT path.
•	best_RGBwrapped.pth :: RGB-wrapped C1 (3-channel entrypoint), useful for RGB pipelines.
•	last_state.pth :: last-epoch training state (model, optimizer, scheduler).
•	metrics.csv :: global epoch metrics (train/val loss, accuracy, macro F1).
•	confmat_train_epoch_XX.csv, confmat_val_epoch_XX.csv :: per-epoch confusion matrices

3.	Lauching the watermarking pipeline

Once you have the autoencoder and naïve classifier .pth files saved, and you know paths to your dataset, it is time to set up the watermarking pipeline. 
The pipeline is located in the 2026-11-11_Trainer_AGNOSTIC_NH33_GRAY-RGB_CM.py file. 
The variables in the code must be set as follows:
•	C1_CKPT :: path to C1 checkpoint (best_raw.pth) produced in Step 2.
•	TRAIN_DIR, VAL_DIR :: train/val roots for the grayscale dataset (same structure as in Step 2).
•	OUT_ROOT :: root folder for all watermark outputs (collages, logs, class file).
•	AE_PTH :: path to AE checkpoint (universal_ae_best.pth) from Step 1.
•	AE_IMPORT :: module and class name to import the AE:
o	from AE_universal_Nov2025 import UniversalAutoEncoder
•	AE_PY_PATH :: optional extra directory to append to sys.path if the AE file is not on the default module path.
Additional relevant constants (usually left at defaults):
•	IMAGE_SIZE = 512 :: must match AE and C1.
•	BATCH_SIZE, NUM_WORKERS, EPOCHS, ACC_STEPS_C2, MIXED_PRECISION.

Once the variables have been set, you can run the file script file directly. There are no command-line parameters to set or pass.
The process of watermarking begins as follows:

The loader in make_loaders():
•	Resizes images to IMAGE_SIZE × IMAGE_SIZE using bilinear interpolation.
•	Forces grayscale: transforms.Grayscale(num_output_channels=1).
•	Converts to tensor and normalizes to [-1,1] with (mean=[0.5], std=[0.5]).
Naïve classifier is wrapped by GuardC1, which:
•	Loads the checkpoint from C1_CKPT.
•	Verifies that:
o	The class list inside the checkpoint matches the dataset folder structure.
o	The Naive head dimensionality is consistent with the number of classes.
•	Automatically adapts between 1-channel and 3-channel checkpoints.
The AE is imported and loaded via:
•	AEClass = import_ae_class() (using AE_IMPORT / AE_PY_PATH).
•	self.ae = AEClass().to(DEVICE) followed by state dict loading from AE_PTH.
The watermark trainer relies on AE methods:
•	forward_plain(x01)
•	embed_external_wm_gray(x01, wm_lat, wm_skip, alpha_lat, alpha_skip, roi_lat_32, roi_skip_64)
•	enc(x01) returning a dict with keys "latent" and "s64".

The training loop performs, per batch:
1.	Generator step (step_generator)
o	Extracts AE features (latent, s64) for grayscale inputs in [0,1].
o	Builds ROI masks using:
	C1 Grad-CAM (S) at 512×512.
	Low-texture masks.
	Top-k selection and border ring.
o	Generates latent and skip-64 watermark patterns, normalized within ROIs.
o	Calls embed_external_wm_gray(...) to generate watermarked images.
o	Optimizes for:
	Minimal leakage outside ROI (leak + high-frequency loss).
	Stability of C1 predictions (JS + cosine losses).
	Low total variation and sparsity of ROIs.
2.	C2 step (step_c2)
o	Uses the current generator to produce watermarked and clean images.
o	Trains C2 with:
	Classification loss on watermarked images, biased to reward high confidence with watermark.
	Auxiliary loss terms that encourage:
	Larger logit margin on watermarked vs clean images.
	Balanced but higher accuracy on watermarked images.
	Explicit exploitation of the watermark logit.
3.	Controller update (controller_update)
o	Adapts global ε and latent/skip ratio r_skip based on:
	PSNR between base AE reconstruction and watermarked images.
	Overlap of ROIs with C1 CAM.
	C1 drop in accuracy.
	EMA of C2’s accuracy gain Δ.
Collages with diagnostics are periodically written to OUT_ROOT for qualitative inspection.
Training stops when the EMA of Δ reaches the target TARGET_DELTA (default ≈ 0.65, adjustable).

Under OUT_ROOT you will find:
•	classes.txt — one class name per line, matching C1 and the dataset.
•	wm_epoch_summary.csv — per-epoch summary with:
o	C1 accuracy (clean and watermarked).
o	C2 accuracy (clean and watermarked).
o	Accuracy deltas and soft probability deltas.
•	Collages with watermarked images for your inspection.
Each collage shows:
•	Original image.
•	AE reconstruction.
•	Watermarked image.
•	Seismic difference map.
•	Latent and skip64 contribution maps.

After these steps you obtain a complete watermarking system: a universal AE with external injection surfaces, a clean classifier C1, and a watermark-aware classifier C2 trained jointly with a constrained generator, as well as the watermarked dataset for your visual inspection.




