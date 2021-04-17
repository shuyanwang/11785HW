# Writeup for HW3P2

Author: Zongyue Zhao (zongyuez@andrew.cmu.edu)

## Run:
Please run `python hw3.py --train --bi`. The rest hyperparameters are set to
default in my script.

## Architecture:
The best performance is achieved from the following architecture:
- A wide variant of ResNet: the first convolutional layer is with stride 1 and kernel size 3x3, the output dimension is
  512; I used 3 sets of layers, expanding the channel dimension from 512 to 2048.
  
- A 1D bidirectional rnn operating on the time dimension. The hidden size is 1024, with a dropout probability of 0.5.
  
- A final dense layer.

## Loss Function and Optimizer:
I used CTCLoss and the Adam Optimizer.

## Decoder
I used `ctcdecoder` (beam search).

## Other Hyperparameters
The learning rate is 2e-3, the weight decay is 5e-5; batch size is 32, I also used a scheduler to cut the lr in half
every 5 epochs.

## Other efforts
I tested different model architectures, optimizers, and various hyperparameters. The models I tested can be found in
`models.py`
