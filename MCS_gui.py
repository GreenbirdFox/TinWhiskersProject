from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Treeview
from tkinter import ttk
from Main_Monte_Carlo import main_MCS, draw_img, export_Pairs_Data
from PIL import Image
import time
import os

root = Tk()
root.title("Monte Carlo Simulator")
root.geometry("810x1000")
root.resizable = (True, True)
start = time.time()

# Create A Main Frame
main_frame = Frame(root, width=810, height=1000)
main_frame.place(x=0, y=0)

# Create A Canvas
my_canvas = Canvas(main_frame, width=810, height=1000)
my_canvas.place(x=0, y=0)

# Add A Scrollbar To the Canvas
y_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
y_scrollbar.place(x=795, y=0, height=1000)

# Configure the Canvas
my_canvas.configure(yscrollcommand=y_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=(0, 0, 0, 1300)))

# Create ANOTHER Frame INSIDE the Canvas
second_frame = Frame(my_canvas, width=810, height=1000)
second_frame.place(x=0, y=0)

# Add that New frame to a window in the Canvas
my_canvas.create_window((0, 0), window=second_frame, anchor='nw')

fName = None


def run():
    root.lPWB = float(root.var1.get())
    root.wPWB = float(root.var2.get())
    root.tkNum = int(root.var3.get())
    root.simNum = int(root.var4.get())
    root.lMu = float(root.var5.get())
    root.lSigma = float(root.var6.get())
    root.tMu = float(root.var7.get())
    root.tSigma = float(root.var8.get())

    # Get data from main_MCS
    treeview_data, results = main_MCS(root.lPWB, root.wPWB, root.tkNum, root.simNum,
                                      root.lMu, root.lSigma, root.tMu, root.tSigma, fName)

    # Create striped row tags
    my_tree.tag_configure('oddrow', background="white")
    my_tree.tag_configure('evenrow', background="lightblue")

    count = 0
    for record in treeview_data:
        if count % 2 == 0:
            my_tree.insert(parent='', index='end', iid=str(count), text="",
                           values=(record[0], record[1], record[2], record[3], record[4]), tags='evenrow')
        else:
            my_tree.insert(parent='', index='end', iid=str(count), text="",
                           values=(record[0], record[1], record[2], record[3], record[4]), tags='oddrow')
        count += 1

    # Store data
    trial, num_run_bridging, total_num_wsk, total_num_bridged = results[0], results[1], results[2], results[3]
    P_bridging_events = round(results[1] / results[0], 4) * 100
    P_bridging = round(results[3] / results[2], 4) * 100

    # Treeview for summary
    # First Part
    tree_frame_s1 = Frame(second_frame)
    tree_frame_s1.place(x=35, y=820)
    # Create Treeview
    my_tree_s1 = Treeview(tree_frame_s1, height=1, yscrollcommand=tree_scroll.set, selectmode="extended",
                          show=["headings"])
    # Pack to the screen
    my_tree_s1.pack()
    # Configure the scrollbar
    tree_scroll.config(command=my_tree_s1.yview)
    # Define Columns
    my_tree_s1['columns'] = ("# of Trials with 1 or more Bridging Whiskers",
                             "# of Trials",
                             "P of 1 or more Bridging Events (%)",)
    # Formate Columns
    my_tree_s1.column("# of Trials with 1 or more Bridging Whiskers", anchor=CENTER, width=285)
    my_tree_s1.column("# of Trials", anchor=CENTER, width=200)
    my_tree_s1.column("P of 1 or more Bridging Events (%)", anchor=CENTER, width=235)

    # Create Headings
    my_tree_s1.heading("# of Trials with 1 or more Bridging Whiskers",
                       text="# of Trials with 1 or more Bridging Whiskers", anchor=CENTER)
    my_tree_s1.heading("# of Trials", text="# of Trials", anchor=CENTER)
    my_tree_s1.heading("P of 1 or more Bridging Events (%)",
                       text="P of 1 or more Bridging Events (%)", anchor=CENTER)

    my_tree_s1.tag_configure('oddrow', background="lightblue")
    my_tree_s1.insert(parent='', index='end', iid=str(0), text="",
                      values=(num_run_bridging, trial, P_bridging_events), tags='oddrow')

    # Second Part
    tree_frame_s2 = Frame(second_frame)
    tree_frame_s2.place(x=35, y=880)
    # Create Treeview
    my_tree_s2 = Treeview(tree_frame_s2, height=1, yscrollcommand=tree_scroll.set, selectmode="extended",
                          show=["headings"])
    # Pack to the screen
    my_tree_s2.pack()
    # Configure the scrollbar
    tree_scroll.config(command=my_tree_s2.yview)
    # Define Columns
    my_tree_s2['columns'] = ("Total # of Bridging Whiskers",
                             "Total # of Falling Whiskers",
                             "P of Bridging after {} Trials (%)".format(trial))
    # Formate Columns
    my_tree_s2.column("Total # of Bridging Whiskers", anchor=CENTER, width=285)
    my_tree_s2.column("Total # of Falling Whiskers", anchor=CENTER, width=200)
    my_tree_s2.column("P of Bridging after {} Trials (%)".format(trial), anchor=CENTER, width=235)

    # Create Headings
    my_tree_s2.heading("Total # of Bridging Whiskers",
                       text="Total # of Bridging Whiskers", anchor=CENTER)
    my_tree_s2.heading("Total # of Falling Whiskers", text="Total # of Falling Whiskers", anchor=CENTER)
    my_tree_s2.heading("P of Bridging after {} Trials (%)".format(trial),
                       text="P of Bridging after {} Trials (%)".format(trial), anchor=CENTER)

    my_tree_s2.tag_configure('oddrow', background="lightblue")
    my_tree_s2.insert(parent='', index='end', iid=str(0), text="",
                      values=(total_num_bridged, total_num_wsk, P_bridging), tags='oddrow')


# Button 'Browse' Function
def browse_files():
    global fName
    fName = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                       filetypes=(('Png Files', '*.png'), ('Jpg Files', '*.jpg'), ('All Files', '*.*')))
    label_file_explorer.configure(text="File Opened")
    print("File Opened: " + fName)


# Button 'Reset' Function
def reset_values():
    root.var1.set("")
    root.var2.set("")
    root.var3.set("")
    root.var4.set("")
    root.var5.set("")
    root.var6.set("")
    root.var7.set("")
    root.var8.set("")
    root.var9.set("")


# Button 'Show' Function
def show_img():
    root.imgN = float(root.var9.get())
    draw_img(root.imgN)


# Button 'Export" Function
def export_data():
    No_Bridging_Whisker = export_Pairs_Data()
    if No_Bridging_Whisker:
        label_file_saved.configure(text="No Bridging Whisker")
    else:
        label_file_saved.configure(text="File Saved to: " + os.getcwd())


# Open the first figure
def open_f1():
    root.simNum = int(root.var4.get())
    fig1_path = 'Percentage of 1 or More Whiskers Bridging after {} simulations.png'.format(root.simNum)
    img = Image.open(fig1_path)
    img.show()


# Open the second figure
def open_f2():
    root.simNum = int(root.var4.get())
    fig2_path = 'Cumulative Probability after {} simulations.png'.format(root.simNum)
    img = Image.open(fig2_path)
    img.show()


# Open the third figure
def open_f3():
    root.simNum = int(root.var4.get())
    fig2_path = 'Conductor Pairs Bridging Frequency after {} simulations.png'.format(root.simNum)
    img = Image.open(fig2_path)
    img.show()


# Open the fourth figure
def open_f4():
    root.simNum = int(root.var4.get())
    fig2_path = 'Conductor Bridging Frequency after {} simulations.png'.format(root.simNum)
    img = Image.open(fig2_path)
    img.show()


# PWB_L
root.var1 = StringVar()
# PWB_W
root.var2 = StringVar()
# tkNum
root.var3 = StringVar()
# simNum
root.var4 = StringVar()
# lMu
root.var5 = StringVar()
# lSigma
root.var6 = StringVar()
# tMu
root.var7 = StringVar()
# tSigma
root.var8 = StringVar()
# imgN
root.var9 = StringVar()

typeface = 'DejaVu Sans Mono'
Label(second_frame, text="Tin Whiskers Risk Assessment Tool", font=(typeface, 18, 'bold')).place(anchor=CENTER, x=400,
                                                                                                 y=30)

# Create board for inputs and outputs
input_frame = Frame(second_frame, highlightbackground="grey", highlightthickness=1, width=760, height=360)
Label(second_frame, text="INPUTS", font=(typeface, 15, 'bold')).place(x=25, y=40)
input_frame.place(x=20, y=55)

output_frame = Frame(second_frame, highlightbackground="grey", highlightthickness=1, width=760, height=500)
Label(second_frame, text="OUTPUTS", font=(typeface, 15, 'bold')).place(x=25, y=420)
output_frame.place(x=20, y=435)

# Step 1 --------------------
Label(second_frame, text="Step 1: Importing Layout of PWB: ", font=(typeface, 15, 'bold')).place(x=35, y=70)
Button(second_frame, text='Browse', command=browse_files, font=(typeface, 10)).place(x=390, y=70)
Label(second_frame, text="(format: png or jpg)", font=(typeface, 12)).place(x=480, y=70)
label_file_explorer = Label(second_frame, text="", font=(typeface, 12))
label_file_explorer.place(x=650, y=70)

# Step 2 --------------------
Label(second_frame, text="Step 2: Entering PWB Dimensions", font=(typeface, 15, 'bold')).place(x=35, y=115)
Label(second_frame, text="Length (mm): ", font=(typeface, 12)).place(x=50, y=160)
e1 = Entry(second_frame, textvariable=root.var1)
# e1.insert(0, "10.08")
e1.insert(0, "")
e1.place(x=230, y=160, width=120)

Label(second_frame, text="Width (mm): ", font=(typeface, 12)).place(x=430, y=160)
e2 = Entry(second_frame, textvariable=root.var2)
# e2.insert(0, "5.73")
e2.insert(0, "")
e2.place(x=600, y=160, width=120)

# Step 3 --------------------
Label(second_frame, text="Step 3: Entering Simulation Constants", font=(typeface, 15, 'bold')).place(x=35, y=205)
Label(second_frame, text="Number of Tin Whiskers: ", font=(typeface, 12)).place(x=50, y=250)
e3 = Entry(second_frame, textvariable=root.var3)
e3.place(x=230, y=250, width=120)

Label(second_frame, text="Number of Simulations: ", font=(typeface, 12)).place(x=430, y=250)
e4 = Entry(second_frame, textvariable=root.var4)
e4.place(x=600, y=250, width=120)

# Step 4 --------------------
Label(second_frame, text="Step 4: LOGNORMAL Whisker Distribution Parameters", font=(typeface, 15, 'bold')).place(x=35,
                                                                                                                 y=295)
Label(second_frame, text="Mu", font=(typeface, 12)).place(x=280, y=325)
Label(second_frame, text="Sigma", font=(typeface, 12)).place(x=430, y=325)

Label(second_frame, text="Length (microns): ", font=(typeface, 12)).place(x=50, y=355)
Label(second_frame, text="Thickness (microns): ", font=(typeface, 12)).place(x=50, y=385)

e5 = Entry(second_frame, textvariable=root.var5)
e5.insert(0, "5.0093")  # L_mu
e5.place(x=230, y=355, width=120)
e6 = Entry(second_frame, textvariable=root.var6)
e6.insert(0, "1.1519")  # L_sigma
e6.place(x=400, y=355, width=120)
e7 = Entry(second_frame, textvariable=root.var7)
e7.insert(0, "1.1685")  # T_mu
e7.place(x=230, y=385, width=120)
e8 = Entry(second_frame, textvariable=root.var8)
e8.insert(0, "0.6728")  # T_sigma
e8.place(x=400, y=385, width=120)

# Results for Each Trial --------------------
Label(second_frame, text="Results for Each Run", font=(typeface, 15, 'bold')).place(x=35, y=450)
Label(second_frame, text="Check Image for Run #:", font=(typeface, 12)).place(x=50, y=635)
e9 = Entry(second_frame, textvariable=root.var9)
e9.insert(0, "1")  # imgN
e9.place(x=230, y=635, width=60)
Button(second_frame, text='Show', command=show_img, font=(typeface, 10)).place(x=320, y=635)

Label(second_frame, text="Conductor Pairs Data in Excel:", font=(typeface, 12)).place(x=430, y=635)
Button(second_frame, text='Export', command=export_data, font=(typeface, 10)).place(x=680, y=635)
label_file_saved = Label(second_frame, text="", font=(typeface, 8))
label_file_saved.place(x=430, y=660)

# Add some style
style = ttk.Style()
# Pick a theme
style.theme_use("default")
# Configure treeview colors
style.configure("Treeview.Heading", foreground='black')
style.configure("Treeview",
                background="#D3D3D3",
                foreground="black",
                rowheight=20,
                fieldbackground="#D3D3D3"
                )

# Change selected color
style.map('Treeview', background=[('selected', 'blue')])

# Create Treeview Frame
tree_frame = Frame(second_frame)
tree_frame.place(x=35, y=480)

# Treeview Scrollbar
tree_scroll = Scrollbar(tree_frame)
tree_scroll.pack(side=RIGHT, fill=Y)

# Create Treeview
my_tree = Treeview(tree_frame, height=6, yscrollcommand=tree_scroll.set, selectmode="extended", show=["headings"])
# Pack to the screen
my_tree.pack()

# Configure the scrollbar
tree_scroll.config(command=my_tree.yview)

# Define Columns
my_tree['columns'] = ("Run No.",
                      "# of Bridging (n)",
                      "# of Detached (N)",
                      "P of Bridging (n/N)",
                      "# of Conductor Pairs Bridged")

# Formate Columns
my_tree.column("Run No.", anchor=CENTER, width=80)
my_tree.column("# of Bridging (n)", anchor=CENTER, width=150)
my_tree.column("# of Detached (N)", anchor=CENTER, width=150)
my_tree.column("P of Bridging (n/N)", anchor=CENTER, width=150)
my_tree.column("# of Conductor Pairs Bridged", anchor=CENTER, width=190)

# Create Headings
my_tree.heading("Run No.", text="Run No.", anchor=CENTER)
my_tree.heading("# of Bridging (n)", text="# of Bridging (n)", anchor=CENTER)
my_tree.heading("# of Detached (N)", text="# of Detached (N)", anchor=CENTER)
my_tree.heading("P of Bridging (n/N)", text="P of Bridging (n/N)", anchor=CENTER)
my_tree.heading("# of Conductor Pairs Bridged", text="# of Conductor Pairs Bridged", anchor=CENTER)

# Plots --------------------
Label(second_frame, text="Plots", font=(typeface, 15, 'bold')).place(x=35, y=665)

# Buttons for Graphs --------------------
Button(second_frame, width=15, wraplength=130,
       text="Percentage of 1 or More Whiskers Bridging in Each Run", command=open_f1).place(x=40, y=705, width=150,
                                                                                            height=60)
Button(second_frame, width=15, wraplength=130,
       text="Cumulative Probability of Whiskers Bridging", command=open_f2).place(x=225, y=705, width=150, height=60)
Button(second_frame, width=15, wraplength=130,
       text="Conductor Pairs Bridging Frequency", command=open_f3).place(x=410, y=705, width=150, height=60)
Button(second_frame, width=15, wraplength=130,
       text="Conductor Bridging Frequency", command=open_f4).place(x=600, y=705, width=150, height=60)

# Result Summary --------------------
Label(second_frame, text="Results Summary", font=(typeface, 15, 'bold')).place(x=35, y=780)

# Run & Quit --------------------
Button(second_frame, width=15, text="Run", command=run).place(x=150, y=950, width=110)
# Button(second_frame, width=15, text="Reset", command=reset_values).place(x=350, y=950, width=110)
Button(second_frame, width=15, text="Quit", command=root.quit).place(x=550, y=950, width=110)

end = time.time()
root.mainloop()
