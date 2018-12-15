from skimage import io,color,measure,morphology,filters,transform,util,exposure
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import sys
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Image_Process():
    def __init__(self):
        pass

    def PreProcess(image):
        '''图像预处理，包括亮度提升，图像二值化，移除小的像素块，腐蚀膨胀，'''
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
        '''通过统计找到每一行数字的分割点，进行切割'''
        cols, rows = image.shape
        col_stats = np.sum(image,axis=1)

        #画出统计图
        plt.plot(col_stats)
        plt.xlabel('Photo Width(pixels)')
        plt.ylabel('Intensity(pixels)')
        plt.savefig('./pic/process/col_stats.svg')

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
            print("Error, column did't match, try to fix this problem.")
            print('In this situation, reconition might not work well.')
            if len(edge_begin) > len(edge_end):
                edge_end.append(cols-1)
            else:
                edge_begin.insert(0,0)
        for i in range(len(edge_begin)):
            width = (edge_end[i] - edge_begin[i]) * 0.05
            width = int(width)
            col_image.append(image[edge_begin[i] - width:edge_end[i] + width, 0:rows])
        print('This image has '+str(len(col_image))+' columns.')
        return col_image

    def FindNumbers(col_image):
        '''对每一行图片进行处理，切割出每一行中含有的数字'''
        col_labels = []
        for c in range(len(col_image)):
            labels, nums = measure.label(col_image[c],8,background=0,return_num=True)
            col_labels.append(labels)
            print('We judge line '+str(c+1)+' has '+str(nums)+' numbers.')
        every_num = []
        for c in range(len(col_image)):
            temp_col = []
            connection = measure.regionprops(col_labels[c])
            count = 0
            location = []

            #保存处理过程的图片
            label_image = col_labels[c]
            image_label_overlay = color.label2rgb(label_image, image=col_image[c])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

            for region in measure.regionprops(label_image):
                # take regions with large enough areas
                if region.area >= 100:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig('./pic/process/col_'+str(c)+'_selectnumber.svg')
            #

            #正式处理分割
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

    def SaveImage(every_num):
        #储存标准化处理后的文件
        c = len(every_num)
        r = max([len(every_num[l]) for l in range(len(every_num))])
        fig, axs = plt.subplots(c, r)
        for i in range(len(every_num)):
            for j in range(len(every_num[i])):
                axs[i, j].imshow(every_num[i][j], cmap='gray')
                axs[i, j].axis('off')
                io.imsave('./pic/standerpic/numbers_' + str(i) + '_' + str(j) + '.png', every_num[i][j])
        fig.savefig('./pic/process/Stander_Process.png')
        plt.close()

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

class Processing():
    def __init__(self,filepath='./pic/test/20181031_160155.jpg'):
        warnings.filterwarnings('ignore')
        original = sys.stdout
        sys.stdout = Logger("./log/steam.txt")

        file_path = filepath
        image = io.imread(file_path)

        begin_time = time.clock()

        image = Image_Process.PreProcess(image)
        col_image = Image_Process.ColumnCut(image)
        every_num = Image_Process.FindNumbers(col_image)
        Image_Process.SaveImage(every_num)
        model = tf.keras.models.load_model('./model/Last_Version.h5')
        Image_Process.PredictNumbers(every_num,model)

        end_time = time.clock()

        print('Time Cost: %f seconds.'%(end_time-begin_time))
        sys.stdout = original

# temp = Processing()