from skimage import io,color,measure,morphology,filters,transform,util,exposure
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

begin_time = time.clock()
file_path = './pic/20181031_160155.jpg'
image = io.imread(file_path)

def PreProcess(image):
    '''图像预处理，包括亮度提升，腐蚀膨胀，'''
    image = color.rgb2gray(image)
    cols, rows = image.shape
    area = cols * rows

    image = exposure.adjust_gamma(image, 0.7)

    threshold = filters.threshold_yen(image)
    Binary_image = (image < threshold)

    Binary_image = morphology.erosion(Binary_image, morphology.square(3))
    Binary_image = morphology.dilation(Binary_image, morphology.square(5))
    Binary_image = morphology.remove_small_objects(Binary_image, min_size=int(area * 0.00005))
    return Binary_image

def ColumnCut(image):
    col_stats = []
    cols, rows = image.shape
    for i in range(cols):
        temp = 0
        for j in range(rows):
            if image[i][j] == 1:
                temp += 1
        col_stats.append(temp)

    # row 切割点计算
    edge_begin = []
    edge_end = []
    threshold = max(col_stats) * 0.005
    for i in range(cols - 1):
        if col_stats[i + 1] == 0 and col_stats[i] > threshold:
            edge_end.append(i)
        if col_stats[i + 1] > threshold and col_stats[i] == 0:
            edge_begin.append(i)

    # 图像纵向切割
    col_image = []
    if len(edge_begin)!=len(edge_end):
        print("Error, column did't match, try to fix.")
        if len(edge_begin) > len(edge_end):
            edge_end.append(cols-1)
        else:
            edge_begin.insert(0,0)
    print(len(edge_begin), len(edge_end))
    for i in range(len(edge_begin)):
        width = (edge_end[i] - edge_begin[i]) * 0.05
        width = int(width)
        col_image.append(image[edge_begin[i] - width:edge_end[i] + width, 0:rows])
    return col_image

def FindNumbers(col_image):
    col_labels = []
    for c in range(len(col_image)):
        labels, nums = measure.label(col_image[c],8,background=0,return_num=True)
        col_labels.append(labels)
        print('判断第'+str(c+1)+'行有'+str(nums)+'个数字。')
    every_num = []
    for c in range(len(col_image)):
        temp_col = []
        connection = measure.regionprops(col_labels[c])
        count = 0
        location = []
        for region in connection:
            minr, minc, maxr, maxc = region.bbox
            location.append((minc, count))
            temp = col_image[c]
            temp = temp[minr:maxr, minc:maxc]
            border = int(max(maxc - minc, maxr - minr) * 0.2)
            row = maxr - minr
            col = maxc - minc
            if row < col:
                temp = util.numpy_pad(temp, ((col - row // 2, col - row // 2), (border, border)),
                                      'constant', constant_values=0)
            else:
                temp = transform.rescale(temp, (1, 1.5))
                temp = util.numpy_pad(temp, (
                (border, border), (int((row - col) * 0.5) // 2 + border, int((row - col) * 0.5) // 2 + border)),
                                      'constant', constant_values=0)
            temp = transform.resize(temp, (224, 224))
            temp = morphology.dilation(temp, morphology.square(3))
            temp = transform.resize(temp,(28,28))
            temp_col.append(temp)
            count += 1
        location.sort()
        index = []
        for i in range(len(location)):
            index.append(location[i][1])
        temp_col = np.array(temp_col)
        temp_col = temp_col[index]
        every_num.append(temp_col)
    return every_num

def SaveImage(every_image):
    for i in range(len(every_num)):
        for j in range(len(every_num[i])):
            io.imsave('./pic/standerpic/numbers_' + str(i) + '_' + str(j) + '.png', every_num[i][j])


def PredictNumbers(every_num,model):
    answer = []
    sess = tf.Session()
    for r in range(len(every_num)):
        every_num[r] = np.reshape(every_num[r], (len(every_num[r]), 28, 28, 1))
        temp = np.array(every_num[r],dtype=np.float)
        temp_ans = tf.keras.backend.argmax(model.predict(temp))
        answer.append(sess.run(temp_ans))
    answer_list = np.array(answer).tolist()
    for i in answer_list:
        print(i)



image = PreProcess(image)
col_image = ColumnCut(image)
every_num = FindNumbers(col_image)
print(np.shape(every_num[0]))
SaveImage(every_num)
model = tf.keras.models.load_model('./model/Last_Version.h5')
PredictNumbers(every_num,model)


