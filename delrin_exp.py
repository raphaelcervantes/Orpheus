import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd
import math
# for n = 18 mode


def freqToPower(f, a, fo, q, c):
    return a * 10**-10 * (1 / ((f - fo)**2 + (fo / (2 * q))**2)) + c


qual3 = np.ndarray(shape=(7, 1))
q_unloaded = np.ndarray(shape=(7, 1))
q_external = np.ndarray(shape=(7, 1))
coupling_coefficent_vals = np.ndarray(shape=(7, 1))
resonant_frequencies = np.ndarray(shape=(7, 1))
background_level = np.ndarray(shape=(7, 1))
maintext = np.array(['Q = ', 'g = ', 'F = '])
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

cenpafolder = "delrin_experiment/"

filenames = ["EXP_12.95.csv",
             "EXP_13.76.csv",
             "EXP_14.67.csv",
             ]
pzeros = [
	[10.48, 16.8, 3000, 0],
	[5, 16.06, 3000, 0],
	[4, 15.4, 3000, 0]
	]
for index in range(0, len(filenames)):
	# figure, graphs = plt.subplots(2, sharex=True)

	# open csv file
	A = np.genfromtxt(
	    cenpafolder + filenames[index], delimiter=',', skip_header=18,skip_footer = 1)
	# print(A)
	# parse data
	xdata = A[:, 0]/10**9
	ydata = A[:, 1]
	# linearize data
	ydata_lin = 10**(ydata / 10)
	plt.plot(xdata, ydata, 'b-', label = 'Measurement')

	popt, pcov = curve_fit(freqToPower, xdata, ydata_lin, p0 = pzeros[index], maxfev = 100000000)# runs the curve fit
	plt.plot(xdata, 10*np.log10(freqToPower(xdata, * popt)), 'r-', label = 'Lorentzian fit')
	plt.xlabel('Frequency (GHz)')
	plt.ylabel('S21 (dB)')
	plt.legend()
	popt = np.round(popt, decimals = 3)
	popt[2] = np.around(popt[2], decimals = 1)
	if index == 0:
		plt.text(16.5,-50,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.5,-52,"F = {}GHz".format(popt[1]),fontdict = font)
		#plt.text(17.85,-15,str_q\nstr_f,fontdict = font)
	elif index == 1:
		plt.text(15.6,-45,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.6,-46,"F = {}GHz".format(popt[1]),fontdict = font)
	elif index == 2:
		plt.text(15.6,-60,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.6,-62,"F = {}GHz".format(popt[1]),fontdict = font)
	# plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	# plt.close()
	# show total plot
	plt.show()
