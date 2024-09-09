import tkinter as tk

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

window = tk.Tk()
window.title("Draw a Digit")
canvas = tk.Canvas(window, width=200, height=200, bg="white")
canvas.bind("<B1-Motion>", paint)
canvas.pack()
window.mainloop()
