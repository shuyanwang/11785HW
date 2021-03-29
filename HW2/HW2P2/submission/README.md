# Writeup for HW2P2

Author: Zongyue Zhao (zongyuez@andrew.cmu.edu)

# Classification:

## Run:
Please run `python hw2_classification.py --train --flip --erase --perspective`. The rest hyperparameters are set to
default in my script.

## Architecture:
The best performance is achieved from a variant of EfficientNet B4. The key differences between my implementation and
the public available architecture include significant simplifications that adapt HW2, data structures, dropout, etc.
Please see the notes in `models.py` for additional references.

## Loss Function and Optimizer:
I used CrossEntropy and the Adam Optimizer.

## Hyperparameters and Data Augmentation:
I resized the images to 380x380, used perspective transformation, random horizontal flipping, and random erase. Although
I tried normalization and random rotation, they did not work well for me.
Batch size: 16; dropout: 0.4; learning rate: 1e-3.

# Verification:

## Run:
Please run `python hw2_verification_center.py --train --flip --erase --perspective`. The rest hyperparameters are set to
default in my script.

## Architecture
The best performance is achieved from a variant of Resnet34. I used 3x3 kernel and stride = 1 in the first conv layer,
in order to adapt to the 64x64 image. Before I did this, I tried various efforts in this task (some of them can be found
in the `deprecated` folder), including using the same EfficientNetB4 (resized to 380x380); pair-wise metric loss;
triplet loss; b-way loss; etc. None of them worked as good as the 3x3 kernel-ed ResNet 34.

## Loss Function and Optimizer:

I used CrossEntropy & center loss for best performance. In this task, SGD performed better than Adam.
Like I mentioned, I tried b-way loss, pairwise loss, triplet loss, but they did not work well during my tuning.

## Hyperparameters:
No resize, normalization or rotation. Like the classification task, I used perspective transformation, random horizontal
flipping, and random erase. Learning rate: 5e-2; the ratio between crossEntropy and center loss: 0.01 (the center loss
is divided by 100 before adding to the CE loss).
