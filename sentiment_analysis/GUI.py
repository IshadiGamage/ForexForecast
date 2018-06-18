from tkinter import *

# create the window
root = Tk()

# modify root window

root.title("Sentiment Analysys")
root.geometry("300x200")

app = Frame(root)
app.grid()
button1 = Button(app,text = "Analyse Tweet")
button1.grid()

# kick off the event loop
root.mainloop()