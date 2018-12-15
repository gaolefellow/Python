
import tkinter as tk
from tkinter import ttk

from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from Process import Image_Process,Processing

class HelloView(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.output = tk.StringVar()
        self.output.set('Welcome! Click Select to choose a photo.')
        self.width,self.height = parent.maxsize()

        self.img_display = tk.Canvas(self,bg='white',width=self.width,height=self.height*0.4)
        label_text = tk.Label(self, textvariable=self.output, wraplength=600,font=(None, 16))
        botton_Slect_Pic = Button(self, text='Select', compound='left', command=self.printcoords, width=15, height=2)
        botton_Run = Button(self, text='Run', command=self.Reconition, compound='right', width=15, height=2)
        botton_Show = Button(self, text='Show', command=self.ShowResult, compound='right', width=15, height=2)

        self.img_display.grid(row=0,column=0,columnspan=5,sticky=tk.W+tk.E)
        botton_Slect_Pic.grid(row=1,column=0)
        botton_Run.grid(row=1, column=2)
        botton_Show.grid(row=1,column=1)
        label_text.grid(row=2, column=0, columnspan=5)
        self.columnconfigure(1, weight=1)

    def resize(self,w, h, w_box, h_box, pil_image):
        f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return pil_image.resize((width, height), Image.ANTIALIAS)

    def printcoords(self):
        File = filedialog.askopenfilename(parent=self,initialdir="./pic", title='Choose an image.')
        pil_image = Image.open(File)
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, self.width, self.height*0.4, pil_image)
        filename = ImageTk.PhotoImage(pil_image_resized)
        self.img_display.image = filename  # <--- keep reference of your image
        self.img_display.create_image(self.width*0.5, self.height*0.2, image=filename)
        global file_path
        file_path = File
        self.output.set('Click Run to predict the numbers!')

    def Reconition(self):
        global file_path
        self.output.set('Processing, please wait a moment.')
        Processing(file_path)
        self.output.set('Finished! Click Show to review the result.')


    def ShowResult(self):
        f = open('./log/steam.txt', 'r')
        temp = f.read()
        f.close()
        self.output.set(temp)





class MyApplication(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("手写数字识别")
        self.geometry("800x600")
        self.resizable(width=True, height=True)

        HelloView(self).grid(sticky=(tk.E + tk.W + tk.N + tk.S))
        self.columnconfigure(0, weight=1)
if __name__ == '__main__':
    app = MyApplication()
    app.mainloop()