import sys
#sys.path.append('hw1')
#Append the correct path to hw1.py, this might throw an error
sys.path.append('hw1p1/main/handout/hw1')
import hw1
import activation
import loss
import linear
import batchnorm
import numpy as np

#Don't change this.
seeds = [4,1,2]

pos = 0

''' This function generates the weights given in the pdf. You are welcome to define a function of your own, but make sure that the weight values
	are the same.
'''
def create_weight(input_size,output_size):
  global pos

  np.random.seed(seeds[pos])

  dims = (input_size,output_size)

  mat = np.random.randint(-3,7, size=dims)

  if pos < len(seeds):
  	pos+=1
  else:
  	pos = 0

  return mat

def bias_init(x):
	return np.zeros((1, x))

inputs = np.array(([4,3],[5,6],[7,8]))
outputs = np.array(([240,4],[320,2],[69,5]))

#Call the MLP class as you have defined in hw1.py. Look up how to use classes imported from another file if you are confused.
#Pass all the necessary arguments as defined in hw1.py. Refer the toyproblem pdf for specific values.

#set learning rate = 0.008

'''
To Verify the values that the weight matrices take as forward propagation happens look at what values can be accessed as defined in the 
__init__ method of linear.py. Do the same for verifying the gradients. You can do this for each layer if you carefully look for the variable
holding the information about the layers of the neural network, perhaps a list maybe?
'''

