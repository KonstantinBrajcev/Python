#!/usr/bin/python
# -*- coding: utf-8 -*-

from tkinter import Tk, Frame, Entry, BOTH
from tkinter.ttk import Button, Style
import tkinter as tk


class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background='white')
        self.parent = parent
        self.parent.title("Образец")
        self.style = Style()
        self.style.theme_use("default")
        self.pack(fill=BOTH, expand=1)
        self.centerWindow()
        self.initUI()

    def centerWindow(self):
        w = 300
        h = 150
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def initUI(self):
        quitButton = Button(self, text="Выход", command=self.quit)
        entry = tk.Entry(fg="black", bg="white", width=150)
        label = tk.Label(text="Имя")

        quitButton.pack()
        label.pack()
        entry.pack()


def main():
    root = Tk()
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()
