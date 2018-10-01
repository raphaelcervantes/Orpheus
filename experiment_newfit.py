import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd
import math
import cmath


def freqToPower2(f, f0, q, beta, l):
    delta = q * (f - f0) / f0
    #l = math.pi/(2*beta)
    g = complex(0, beta)
    gamma = ((1 - beta + 2j * beta * delta + 4 *
              delta ** 2) / (1 + 4 * delta ** 2))
    #exp = (math.cos(2*g.imag*l)- 1j * math.sin(2*g.imag*l))
    exp = (math.exp(2 * g * l))
    #exp = cmath.cos(2*g.imag*l)-1j*cmath.sin(2*g.imag*l)
    new_gamma = abs(gamma * exp)**2
    return new_gamma * 0.2
# def freqToPower(f,fo,q,b):
#     return ((((1-b)+4*(q*(f-fo)/fo)**2)**2+4*b**2*(q*(f-fo)/fo)**2)/(1+4*(q*(f-fo)/fo)**2)**2)*0.2


resfre_qual = np.ndarray(shape=(7, 3))

qual1 = np.ndarray(shape=(7, 1))
q_unloaded = np.ndarray(shape=(7, 1))
q_external = np.ndarray(shape=(7, 1))
coupling_coefficent_vals = np.ndarray(shape=(7, 1))
resonant_frequencies = np.ndarray(shape=(7, 1))
background_level = np.ndarray(shape=(7, 1))
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
cenpafolder = "Orpheus_empty_lcavitylength/"
filenames = [
    "ORPHEUS_EMPTY_17CM_S22.CSV",
    "ORPHEUS_EMPTY_17.5CM_S22.CSV",
    "ORPHEUS_EMPTY_18CM_S22.CSV",
    "ORPHEUS_EMPTY_18.5CM_S22.CSV",
    "ORPHEUS_EMPTY_19CM_S22.CSV",
    "ORPHEUS_EMPTY_19.5CM_S22.CSV",
    "ORPHEUS_EMPTY_20CM_S22.CSV"]
pzeros = [
    [17.97, 3000, 0.999, 1],
    [17.415, 3000, 0.69, 1],
    [16.94, 3000, 0.77, 1],
    [16.49, 3000, 0.95, 1],
    [16.06, 3000, 0.77, 1],
    [15.649, 3000, 0.87, 1],
    [15.258, 3000, 0.78, 1]
]
for index in range(0, len(filenames)):
    #figure, graphs = plt.subplots(2, sharex=True)

    # open csv file
    A = np.genfromtxt(
        cenpafolder + filenames[index], delimiter=',', skip_header=1,)
    # print(A)
    # parse data
    xdata = A[:, 0] / (10**9)
    ydata = A[:, 1]
    # linearize data
    ydata_lin = 10**(ydata / 10)
    if index == 0:
        xdata = xdata[50:]
        ydata_lin = ydata_lin[50:]
        ydata = ydata[50:]

    minval = np.amin(ydata_lin)
    i = np.where(ydata_lin == minval)
    # print(i)
    firstnum = i[0]
    # print(firstnum)
    initial = firstnum - 40
    final = firstnum + 40
    # print(initial)
    # print(final)
    # minval_data = np.amin(ydata)
    # index = np.where(ydata == minval)
    # #print(i)
    # firstnum_2 = index[0]
    # #print(firstnum)
    # initial_2 = firstnum_2-40
    # final_2 = firstnum_2+40

    data_extracted_x = np.array(xdata[initial[0]:final[0]])
    xlist = data_extracted_x[0:]
    data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
    ylist = data_extracted_y[0:]
    ylist_data = np.array(ydata[initial[0]:final[0]])
    # data_extracted_ydata = np.array(ydata[initial_2[0]:final_2[0]])
    # y_data_list = data_extracted_ydata[0:]
    #graphs[0].plot(xdata , ydata_lin)
    plt.plot(xdata, ydata_lin, 'b-', label='data')  # plot first graph
    #plt.plot(xdata, freqToPower(xdata, *pzeros[index]))
    popt, pcov = curve_fit(freqToPower2, xdata, ydata_lin,
                           p0=pzeros[index], maxfev=100000000)  # run the curve fit
    plt.plot(xdata, (freqToPower2(xdata, *popt)),
             'r-', label='fit')  # plots second graph
    print(popt)
    # plt.show()
    plt.xlabel('frequency')
    plt.ylabel('S11(dB)')
    plt.legend()



    #plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
    # plt.close()
    # show total plot
    plt.show()
