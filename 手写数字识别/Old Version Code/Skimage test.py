
import matplotlib.pyplot as plt
from skimage import measure,io,morphology,transform,util
from skimage import filters,color,data
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
import matplotlib.patches as mpatches
import numpy as np


img = io.imread('./data/x_train_png/3.png')
test = io.imread('./pic/standerpic/numbers_1_10.png')



threshold = filters.threshold_yen(test)
test = (test > threshold)
test = transform.resize(test,(28,28))
print(test)
io.imshow(test)
io.show()




