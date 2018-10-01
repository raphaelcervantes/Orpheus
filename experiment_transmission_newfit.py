import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd
import math

def freqToPower2(f,f0,q,beta,l):
    delta = q * (f - f0) / f0
    #l = math.pi/(2*beta)
    g = complex(0,beta)
    gamma = ((1 - beta + 2j * beta * delta + 4 * delta ** 2) / (1 + 4
            * delta ** 2))
    exp  = (math.e**(-2 * g * l))
    new_gamma = abs(1+(gamma * exp))**2
    return new_gamma * 0.2

# def freqToPower2(f,f0,q,beta):
#     delta = q * (f - f0) / f0
#     gamma = (1 - beta + 2j * beta * delta + 4 * delta ** 2) / (1 + 4
#             * delta ** 2)
#     return abs(1 - gamma) ** 2


# def freqToPower(f,fo,q,b):
#     return ((abs(1+((1-b+2j*b*(q*(f-fo)/fo)+4*(q*(f-fo)/fo)**2))/(1+4*(q*(f-fo)/fo)**2)))**2)

qual3 = np.ndarray(shape=(7, 1))
q_unloaded = np.ndarray(shape=(7, 1))
q_external = np.ndarray(shape=(7, 1))
coupling_coefficent_vals = np.ndarray(shape=(7, 1))
resonant_frequencies = np.ndarray(shape=(7, 1))
background_level = np.ndarray(shape=(7, 1))
maintext = np.array(['Q = ', 'g = ', 'F = '])
font = {'color': 'black', 'weight': 'normal', 'size': 12}

cenpafolder = 'Orpheus_empty_lcavitylength/'
filenames = [
    'ORPHEUS_EMPTY_17CM_S21.CSV',
    'ORPHEUS_EMPTY_17.5CM_S21.CSV',
    'ORPHEUS_EMPTY_18CM_S21.CSV',
    'ORPHEUS_EMPTY_18.5CM_S21.CSV',
    'ORPHEUS_EMPTY_19CM_S21.CSV',
    'ORPHEUS_EMPTY_19.5CM_S21.CSV',
    'ORPHEUS_EMPTY_20CM_S21.CSV',
    ]
pzeros = [
    [17.956, 3000, 0.37, 1.5],
    [17.42, 3000, 1,0.9],
    [16.956, 2000, 1,1.8],
    [16.474, 2000, 1,1.5],
    [16.075, 2000, 1,2.2],
    [15.657, 2000, 1,2],
    [15.275, 2000, 1,1],
    ]
for index in range(0, len(filenames)):
    A = np.genfromtxt(cenpafolder + filenames[index], delimiter=',',skip_header=1)#open csv file
    xdata = A[:, 0] / 10 ** 9
    ydata = A[:, 1]
    ydata_lin = 10 ** (ydata / 10)
    if index == 0:
        xdata = xdata[50:]
        ydata_lin = ydata_lin[50:]
        ydata = ydata[50:]

    maxval = np.amax(ydata_lin)
    max_s21 = np.amax(ydata)
    i = np.where(ydata_lin == maxval)
    firstnum = i[0]
    initial = firstnum - 20
    final = firstnum + 20
    data_extracted_x = np.array(xdata[initial[0]:final[0]])
    xlist = data_extracted_x[0:]
    data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
    ylist = data_extracted_y[0:]
    ylist_data = np.array(ydata[initial[0]:final[0]])

    # #plot to the first graph
    # graphs[0].plot(xdata , ydata_lin)

    plt.plot(xdata, ydata_lin, 'b-', label='data')

      # plt.plot(xdata, freqToPower(xdata, *pzeros[index]))

    # #run the curve fit
    popt, pcov = curve_fit(freqToPower2, xdata,ydata_lin,p0=pzeros[index],maxfev=100000000)

    #popt = pzeros[index]

    # popt = np.round(popt,decimals = 4)
    print(popt)
    # #plot to the second graph
    # graphs[1].plot(xdata, freqToPower(xdata,*popt))

    plt.plot(xdata, freqToPower2(xdata, *popt), 'r-', label='fit')
    plt.xlabel('frequency')
    plt.ylabel('S21(dB)')
    plt.legend()
    # plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
    # plt.close()
    # #show total plot

    plt.show()
