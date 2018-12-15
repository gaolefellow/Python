from skimage import io,color,measure,morphology,filters,transform,util
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import time
import cv2

begin_time = time.clock()

FilePath = './pic/qq_pic_merged_1538276983645.jpg'


image = io.imread(FilePath)
image = color.rgb2gray(image)

cols, rows = image.shape
image_area = cols*rows

threshold = filters.threshold_yen(image)
Binary_image = (image < threshold )

Binary_image = morphology.dilation(Binary_image,morphology.square(5))
Binary_image = morphology.remove_small_objects(Binary_image,min_size=int(image_area*0.00005))

plt.imshow(Binary_image,cmap='gray')
plt.show()

row_stats=[]
for i in range(cols):
    temp = 0
    for j in range(rows):
        if Binary_image[i][j] == 1:
            temp+=1
    row_stats.append(temp)


#col切割点计算
edge_begin=[]
edge_end=[]
threshold = max(row_stats)*0.001
for i in range(cols-1):
    if row_stats[i+1] == 0 and row_stats[i] > threshold:
        edge_end.append(i)
    if row_stats[i+1] > threshold and row_stats[i] == 0:
        edge_begin.append(i)

#图像纵向切割
row_image = []
print(len(edge_begin),len(edge_end))
for i in range(len(edge_begin)):
    width = (edge_end[i]-edge_begin[i])*0.05
    width = int(width)
    row_image.append(Binary_image[edge_begin[i]-width:edge_end[i]+width,0:rows])

#对每行进行联通区域计算。
row_labels = []
for i in range(len(row_image)):
    label, nums = measure.label(row_image[i],8,background=0,return_num=True)
    row_labels.append(label)
    print(nums)

#对每行的数字进行分割
every_num = []
for r in range(len(row_image)):
    temp_row = []
    connection = measure.regionprops(row_labels[r])
    count = 0
    location = [] #用于建立从左到右的顺序
    for region in connection:
        minr, minc, maxr, maxc = region.bbox
        location.append((minc,count))
        temp = row_image[r]
        temp = temp[minr:maxr,minc:maxc]
        border = int(max(maxc - minc, maxr - minr) * 0.2)
        row = maxr-minr
        col = maxc-minc
        if row < col:
            temp = util.numpy_pad(temp,((col-row//2,col-row//2),(border,border)),
                                  'constant',constant_values=0)
        else:
            temp = transform.rescale(temp,(1,1.5))
            temp = util.numpy_pad(temp, ((border,border), (int((row-col)*0.5)//2+border,int((row-col)*0.5)//2+border)),
                                  'constant', constant_values=0)
        temp = transform.resize(temp,(112,112))
        temp = morphology.dilation(temp,morphology.square(3))
        temp_row.append(temp)
        count += 1

    location.sort()
    index = []
    for i in range(len(location)):
        index.append(location[i][1])
    temp_row = np.array(temp_row)
    temp_row =temp_row[index]
    every_num.append(temp_row)

for i in range(len(every_num)):
    for j in range(len(every_num[i])):
        io.imsave('./pic/standerpic/numbers_' + str(i) +'_'+ str(j) + '.png', every_num[i][j])
print(np.shape(every_num))

model = tf.keras.models.load_model('./model/VGG_self.h5')
answer=[]
sess = tf.Session()

for r in range(len(every_num)):
    every_num[r] = np.reshape(every_num[r],(len(every_num[r]),112,112,1))
    temp_ans = tf.keras.backend.argmax(model.predict(every_num[r]))
    answer.append(sess.run(temp_ans))

answer_list = np.array(answer).tolist()
for i in answer_list:
    print(i)

#作图检验测试区
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(row_image[1],cmap='gray')

for region in measure.regionprops(row_labels[1]):
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
end_time = time.clock()
print(end_time-begin_time)
plt.show()