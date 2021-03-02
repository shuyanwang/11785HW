# HW1P2:
Author: Zongyue Zhao (zongyuez@andrew.cmu.edu); Kaggle Username: Andromeda15; Kaggle Leaderboard Name: Zongyue Zhao.

## Test

To verify the result, please run `main.py` from the root dir. The best-performance results was obtained under the 
following hyperparameters: batch size `B=8192`, learning rate `lr=1e-3`, context `k=30`, dropout = 0.5.
I trained the network for 70 epochs before testing.

## Code Structure

- `main.py` is the main entrance to conduct tests.

- `utils/base.py` aims to provide a template for all HWP2 assignments.

- `models.py` contains all network architectures I used.

- `learninghw1.py` implements the dataset for this assignment, as well as certain abstract methods in `utils/base.py`,

- `utils/check_data.py` is just what I used to check numpy array dimensions in the debugging mode.

- `utils/count_classes.py` calculated the label counts, however I did not use the counted results to balance the loss.

## Network Architecture

`models.py` stores all network architectures I used during this assignment. All of them are based on MLP 
(fully-connected perceptron layers.) In different models, I tried tricks like skipping layers by concatenation,
skip layers by addition (perceptron-based residual); batch-normalization, dropout, etc.

I used the ADAM optimizer and the CrossEntropy loss throughout this assignment.

## Hyperparameters Tuned

I tuned the context size K, the batch size, the dropout ratio, the number of channels of each perceptron layer,
learning rate, and early-stopping epoch numbers. I also tried using double over float for the tensor dtypes.

## Data Loading Scheme

I built my own dataset class (inherited the `torch.utils.data.Dataset` class) in `learninghw1.py`. More specifically, I 
first load the data into numpy arrays, then pad each utterance at the beginning and at the end.
Then, I built a look-up table to convert (utterance id, frame id) to a single index for the map-based torch dataset.

When retrieving data, I called the `torch.flatten` function to convert the 2D features (Context * Channels) to a
1D vector for the first layer of the MLP.

For the submitted results, I did not balance differences among classes. This is because doing a weighted sampling turns
out to be memory-inefficient.


