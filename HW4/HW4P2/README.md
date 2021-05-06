# Writeup for HW4P2

Author: Zongyue Zhao (zongyuez@andrew.cmu.edu)

## Verify the Results:
Please run `utils/make_vocab.py` to parse the labels. Then, please run `python hw4.py --train --clip` until epoch 53.
Then run `python hw4.py --train --clip --load --forcing (0.6,0.6,10) --epoch 53
--load Model1_b32lr0.0005s100decay0Adamdrop0.4le3he256hd512emb256att128forcing(0.9,0.8,20)clip` until epoch 65,
and finally run `python hw4.py --train --clip --load --forcing (0.5,0.5,10) --epoch 65
--load Model1_b32lr0.0005s100decay0Adamdrop0.4le3he256hd512emb256att128forcing(0.6,0.6,10)clip` until epoch 76.
All other hyperparameters are set to default in the scripts.

## Architecture:
The best performance is achieved from the following architecture:

- The listen-attend-spell model (with key-value dot attention) (template provided by course bootcamp)
- Locked dropout (from HW4P1) is applied between each two layers in the encoder.
- Weight tying (also from HW4P1) is applied in the decoder.

## Loss Function and Optimizer:
I used CrossEntropy (masked) and the Adam Optimizer.

## String Decoding
I used greedy decoding.

## Other Hyperparameters
The learning rate is 5e-4, no weight decay; batch size is 32. The teacher forcing rate is described above.
No Gumble noise is used, nor is the beam search.

## Effort not used for the best result
I tested different model architectures, some additional techniques introduced in HW4P1, e.g., drop connect,
and various hyperparameters. The models I tested can be found in `models.py`. Furthermore, I found that removing all
packing, i.e., use padded sequence throughout the network, actually runs faster with my GPUs.
