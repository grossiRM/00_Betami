#import packages
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np
from pandas import Timestamp
import seaborn as sns
from PIL import Image
import glob
from numpy.random import randint
import matplotlib as mpl
import os
import math
import seaborn as sns

#set working directory as "directory"
directory = (r"C:\Users\srubin\OneDrive - INTERA Inc\Lisbon\MODFLOW_Cl_Visuals1")
os.chdir(directory)

df_raw = pd.read_csv(r"l210621g_S6_targets_Cl.csv")
reducer = len(df_raw) -13
df = df_raw[:reducer]

#try and create subfolder for figures, plots
try:
    os.mkdir("Head_Vs_Time_Figures")
except FileExistsError:
    pass

try:  
    os.mkdir("ScatterPlots")
except FileExistsError:
    pass

try:
    os.mkdir("Head_Vs_Time_3x2")
except FileExistsError:
    pass

#set referential dates
refDate = dt(2003, 6, 1)
maxDate = dt(2021,6,1)

df['Date'] = refDate + pd.TimedeltaIndex(df['Time'], unit='D')

#print(df.head(10))
print(df.info())

#define series for graphing: observed, computed, layers, residual
#data to be multiplied by 0.0353147 to convert from mg/ft3 to mg/L
convert_var = 0.0353147

observed = df['Observed'] * convert_var
computed = df['Computed'] * convert_var
residual = df['Residual'] * convert_var

df['Layer'] = df['Layer'].astype(int)
print(df.Layer.value_counts())

#scatter plots1,2
def scatter1():
    sns.scatterplot(data=df, x="Observed", y="Residual",hue="Layer",  style = "Layer", palette='deep')
    plt.title("Observed vs. Residual")
    plt.legend(title="Layers", bbox_to_anchor=(1.0, 1.05))
    plt.tight_layout()
    plt.savefig("ScatterPlots/Observed_vs_Resid_Scatter.png", dpi=300)

def scatter2():
    sns.scatterplot(data=df, x="Observed", y="Computed",hue="Layer",  style = "Layer", palette='deep')
    plt.title("Computed vs. Residual")
    plt.legend(title="Layers", bbox_to_anchor=(1.0, 1.05))
    plt.tight_layout()
    plt.savefig("ScatterPlots/Observed_vs_Computed_Scatter.png", dpi=300)

#call scatter1, scatter2
scatter1()
scatter2()

######
UniqueNames = df.Name.unique()
df_dict = {elem : pd.DataFrame for elem in UniqueNames}

for key in df_dict.keys():
    df_dict[key] = df[:][df.Name == key]

# iterate through dictionary and plot
fig, ax1 = plt.subplots()

for i,val in df_dict.items():
    #plt.figure()
    ax1 = df_dict[i].plot(x='Date',y='Observed',marker = 'o') 
    #ax = "Time", y="Computed", marker = 'o', color="r")
    ax1 = df_dict[i].plot(x="Date", y="Computed",ax=ax1, marker = 'o', color="r")
    plt.xlim([refDate, maxDate])
    plt.ylim([df.Observed.min(), df.Observed.max()])
    ax1.set_yscale('log')  # set y-axis to log-scale
    ax1.set_ylabel("Head")
    #plt.legend()
    #plt.xlabel("Time")
    #plt.ylabel("Head")
    #plt.savefig("...")
    #scatter_directory = 
    ax1.set_title(i)
    plt.savefig(r'Head_Vs_Time_Figures/{}.png'.format(i), dpi=300)
    plt.close()

FOLDER_PATH = "Head_Vs_Time_Figures"
GROUP_BY = 6

def paste_images_in_PDF(page, groups): 
    width, height = int(8.27 * 450), int(11.7 * 375) # A4 at 300dpi
    pdf = Image.new('RGB', (width, height), 'white')
    for imgIndex, img in enumerate(groups):
        #There is a nicer way mathmatical formula way of doing this - but my brain is fried.
        if imgIndex == 0:
            pdf.paste(Image.open(img), box=(0, 0))
        if imgIndex == 1:
            pdf.paste(Image.open(img), box=(int(width/2. +0.5), 0))
        if imgIndex == 2:
            pdf.paste(Image.open(img), box=(0, int(height/3. + 0.5)))
        if imgIndex == 3:
            pdf.paste(Image.open(img), box=(int(width/2.+0.5), int(height/3.+0.5))) 
        if imgIndex == 4:
            pdf.paste(Image.open(img), box=(0, int(height/3.+0.5)*2))
        if imgIndex == 5:
            pdf.paste(Image.open(img), box=(int(width/2.+0.5), int(height/3.+0.5)*2))  
    pdf.save('Head_Vs_Time_3x2/page_{0}.pdf'.format(page+1))

def group_images_generator(images):
    total_images = len(images)
    for i in range(0,math.ceil(total_images/GROUP_BY)):
        yield images[i*GROUP_BY:i*GROUP_BY+GROUP_BY]

def main():
    images = glob.glob("{0}/*.png".format(FOLDER_PATH)) 
    for image in images:
        with open(image, 'rb') as file:
            img = Image.open(file)
    groups = group_images_generator(images)
    [paste_images_in_PDF(page, group) for page, group in enumerate(groups) ]

#run above functions; return png and pdf files to subfolders
if __name__ == "__main__":
    main()
