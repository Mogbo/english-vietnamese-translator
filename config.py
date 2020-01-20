""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the hyperparameters for the model.

See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

DATA_PATH = 'processed'
OUTPUT_FILE = 'test.txt' # This file is written in processed_path
PROCESSED_PATH = 'processed'
CPT_PATH = 'model'

#CPT_PATH = '/scratch/ojuba.e/EECE7398/HW3/viet-exp/checkpoints-HS_512_NL4'
#OUTPUT_FILE = 'viet_translate-HS_512_NL4.txt'


THRESHOLD = 2 # Not needed, only for building dictionary

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63), (71, 73), (105, 105)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3 # default = 3
HIDDEN_SIZE = 256 # default = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512

DEC_VOCAB = 7710
ENC_VOCAB = 17192


