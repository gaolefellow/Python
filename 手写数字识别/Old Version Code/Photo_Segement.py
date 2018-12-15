import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

all_img = cv2.imread('./pic/phone1.jpg',0)
cols,rows = all_img.shape
print(cols,rows)

#二值化
thr, all_img = cv2.threshold(all_img,80,255,cv2.THRESH_BINARY)

# #腐蚀操作
kernel = np.ones((3,3),np.uint8)
all_img = cv2.erode(all_img, kernel, iterations=2)

#纵向分布图
row_stats=[]
for i in range(cols):
    temp = 0
    for j in range(rows):
        if all_img[i][j] == 0:
            temp+=1
    row_stats.append(temp)

# x = np.arange(cols)
# plt.bar(x,row_stats)
# plt.show()

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
    width = (edge_end[i]-edge_begin[i])*0.3
    width = int(width)
    row_image.append(all_img[edge_begin[i]-width:edge_end[i]+width,0:rows])



col_stats=[]
for r in range(len(row_image)):
    eachcol_stats=[]
    for i in range(rows):
        temp=0
        for j in range(len(row_image[r])):
            if row_image[r][j][i]==0:
                temp+=1
        eachcol_stats.append(temp)
    col_stats.append(eachcol_stats)

row_cut_begin=[]
row_cut_end=[]
for r in range(len(row_image)):
    each_row_edge_begin=[]
    each_row_edge_end=[]
    for i in range(rows-1):
        if col_stats[r][i + 1] == 0 and col_stats[r][i] > threshold:
            each_row_edge_end.append(i)
        if col_stats[r][i + 1] > threshold and col_stats[r][i] == 0:
            each_row_edge_begin.append(i)
    row_cut_begin.append(each_row_edge_begin)
    row_cut_end.append(each_row_edge_end)

#单个数字切割
each_image=[]
print(len(row_image),len(row_cut_begin),len(row_cut_end))
for r in range(len(row_image)):
    temp_row_images=[]
    for i in range(len(row_cut_begin[r])):
        width = (row_cut_end[r][i] - row_cut_begin[r][i]) * 0.1
        width = int(width)
        temp_row_images.append(row_image[r][0:,row_cut_begin[r][i]-width:row_cut_end[r][i]+width])
    each_image.append(temp_row_images)

square,short = np.shape(each_image[0][0])
print(square,short)

#补齐成为正方形
kernel = np.ones((3,3),np.uint8)
for i in range(len(each_image)):
    for j in range(len(each_image[i])):
        square,short = np.shape(each_image[i][j])
        temp = cv2.copyMakeBorder(each_image[i][j], 0, 0, (square - short) // 2, (square - short) // 2,
                                  cv2.BORDER_CONSTANT, value=255)
        cut =int(square*0.15)
        temp = temp[cut:square-cut,cut:square-cut]
        temp = cv2.erode(temp,kernel,iterations=4)
        each_image[i][j] = cv2.resize(temp,(112,112),cv2.INTER_LINEAR)
        each_image[i][j] = np.ones((np.shape(each_image[i][j])))*255-each_image[i][j]
# temp = cv2.copyMakeBorder(each_image[i][j],0,0,(square-short)//2,(square-short)//2,cv2.BORDER_CONSTANT,value=255)



for i in range(len(each_image)):
    for j in range(len(each_image[i])):
        cv2.imwrite('./pic/standerpic/number'+str(i)+'_'+str(j)+'.png',each_image[i][j])


model = tf.keras.models.load_model('./model/CNN5.h5')
answer=[]
sess = tf.Session()

for i in range(len(each_image)):
    row_answer=[]
    for j in range(len(each_image[i])):
        each_image[i][j] = each_image[i][j]/255
        each_image[i][j] = np.expand_dims(each_image[i][j], axis=0)
        each_image[i][j] = np.expand_dims(each_image[i][j], axis=3)
        temp_ans = tf.keras.backend.argmax(model.predict(each_image[i][j]))
        row_answer.append(sess.run(temp_ans))
    answer.append(row_answer)

answer_list = np.array(answer).tolist()
for i in answer_list:
    print(i)





cv2.imwrite('./pic/phone2_change.png',all_img)


