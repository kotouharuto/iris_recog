"""
Configuration parameters for Iris Recognition.
"""

# Image Normalization
NORM_HEIGHT = 64
NORM_WIDTH = 512

# Preprocessing / Detection
PUPIL_MIN_AREA = 50
PUPIL_THRESH_START = 10
PUPIL_THRESH_END = 80
PUPIL_THRESH_STEP = 5

# Gabor Filter Parameters
GABOR_KSIZE = 31
GABOR_SIGMA = 3.0
GABOR_THETA = 0
GABOR_LAMBD = 8.0
GABOR_GAMMA = 0.5

# Matching
SHIFT_RANGE = 8