import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import math
def freqToPower(f, a, fo, q, c):
    return a * 10**-10 * (1 / ((f - fo)**2 + (fo / (2 * q))**2)) + c
cenpafolder = "empty_modemap/"
filenames = ["EXP_MAP_16.0.csv",
  "EXP_MAP_16.5.csv",
  "EXP_MAP_17.0.csv",
  "EXP_MAP_17.5.csv",
  "EXP_MAP_18.0.csv",
  "EXP_MAP_18.5.csv",
  "EXP_MAP_19.0.csv",
]
pzeros = [
  [15000, 17.9, 3000, 0],
  [15000, 17.4, 3000, 0],
  [15000, 16.9, 3000, 0],
  [15000, 16.4, 3000, 0],
  [15000, 16.0, 3000, 0],
  [15000, 15.55, 3000, 0],
  [15000, 15.1, 3000, 0],
]
for index in range(0, len(filenames)): #figure, graphs = plt.subplots(2, sharex = True)

# open csv file
    A = np.genfromtxt(
    cenpafolder + filenames[index], delimiter = ',', skip_header = 18,skip_footer = 1 )# print(A)# parse data
    xdata = A[: , 0]/10**9
    ydata = A[: , 1]# linearize data
    ydata_lin = 10 ** (ydata / 10)
    if index == 0:
        xdata = xdata[150:]
        ydata = ydata[150:]
        ydata_lin = ydata_lin[150:]
    elif index == 1:
        xdata = xdata[120:180]
        ydata = ydata[120:180]
        ydata_lin = ydata_lin[120:180]
    elif index == 2:
        xdata = xdata[110:150]
        ydata = ydata[110:150]
        ydata_lin = ydata_lin[110:150]
    elif index == 3:
        xdata = xdata[75:110]
        ydata = ydata[75:110]
        ydata_lin = ydata_lin[75:110]
    elif index == 4:
        xdata = xdata[50:90]
        ydata = ydata[50:90]
        ydata_lin = ydata_lin[50:90]
    elif index == 5:
        xdata = xdata[18:60]
        ydata = ydata[18:60]
        ydata_lin = ydata_lin[18:60]
    elif index == 6:
        xdata = xdata[0:50]
        ydata = ydata[0:50]
        ydata_lin = ydata_lin[0:50]
    plt.plot(xdata, ydata_lin, 'b-', label = 'data')

    popt, pcov = curve_fit(freqToPower, xdata, ydata_lin, p0 = pzeros[index], maxfev = 100000000)# runs the curve fit
    print(popt[2])
    plt.plot(xdata, (freqToPower(xdata, * popt)), 'r-', label = 'fit')
    plt.xlabel('frequency')
    plt.ylabel('S21(dB)')
    plt.legend()
    plt.show()
