#！/usr/bin/env python
#!-*-coding:utf-8-*-
#!@Time    :2018/10/31 17:23
#!@Author  :GL

import tkinter
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from Process import Image_Process,Processing


root = tkinter.Tk()
root.title('应用程序窗口')  # 窗口标题
root.resizable(False, False)  # 固定窗口大小
windowWidth = 850  # 获得当前窗口宽
windowHeight = 700  # 获得当前窗口高
screenWidth, screenHeight = root.maxsize()  # 获得屏幕宽和高
geometryParam = '%dx%d+%d+%d' % (
windowWidth, windowHeight, (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
root.geometry(geometryParam)  # 设置窗口大小及偏移坐标
root.wm_attributes('-topmost', 1)  # 窗口置顶
lb = Label(root,text = '')

frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH,expand=1)

filename=""
# def xz():
#     filename = tkinter.filedialog.askopenfilename()
#     if filename != '':
#         #lb.config(text = "您选择的文件是："+filename);
#     else:
#         lb.config(text = "您没有选择任何文件");

global file_path
global output
output = tkinter.StringVar()
output.set('结果展示区')

# 结果展示区
label_text = tkinter.Label(root, textvariable=output, width=700,height=13);
label_text.pack();


def printcoords():
    File = filedialog.askopenfilename(parent=root, initialdir="./pic", title='Choose an image.')
    filename = ImageTk.PhotoImage(Image.open(File))
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(0, 0, anchor='nw', image=filename)
    global file_path
    file_path = File


def youself_fun():
    global file_path,output
    Processing(file_path)
    f = open('./log/steam.txt','r')
    temp = f.read()
    output.set(temp)
    f.close()



Button(root, text='Choose Picture', compound='left',command=printcoords,width= 15,height = 2).pack(side='left',padx=150)
Button(root, text='Run', command=youself_fun,compound='right',width= 15,height = 2).pack(side='right',padx=150)
root.mainloop()
