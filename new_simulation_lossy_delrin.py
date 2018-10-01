import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd
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

cenpafolder = "new_simulations/lossy_delrin_s21/"

filenames = ["orpheus_lossyDelrin_q18_17GHz_L12p948cm_s21.csv",
             "orpheus_lossyDelrin_q18_16GHz_L13p757cm_s21.csv",
             "orpheus_lossyDelrin_q18_15GHz_L14p674cm_s21.csv",
             ]
pzeros = [
	[0.012, 17.1, 3000, 0],
	[0.001, 16.116, 480, 0],
	[0.002, 15.325, 480, 0]
	]
for index in range(0, len(filenames)):
	# figure, graphs = plt.subplots(2, sharex=True)

	# open csv file
	A = np.genfromtxt(
	    cenpafolder + filenames[index], delimiter=',', skip_header=1,)
	# print(A)
	# parse data
	xdata = A[:, 0]
	ydata = A[:, 1]
	# linearize data
	ydata_lin = 10**(ydata / 10)
	plt.plot(xdata, ydata_lin, 'b-', label = 'data')
	popt, pcov = curve_fit(freqToPower, xdata, ydata_lin, p0 = pzeros[index], maxfev = 100000000)# runs the curve fit
	print(popt)
	plt.plot(xdata, (freqToPower(xdata, * popt)), 'r-', label = 'fit')
	plt.xlabel('frequency')
	plt.ylabel('S21(linearized)')
	plt.legend()
	popt = np.round(popt, decimals = 4)
	# if index == 0:
	# 	plt.text(17.025,-130,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
	# 	plt.text(17.025,-133,"F = {}".format(popt[1]),fontdict = font)
	# 	#plt.text(17.85,-15,str_q\nstr_f,fontdict = font)
	# elif index == 1:
	# 	plt.text(16.25,-165,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
	# 	plt.text(16.25,-166.5,"F = {}".format(popt[1]),fontdict = font)
	# elif index == 2:
	# 	plt.text(15.3,-180,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
	# 	plt.text(15.3,-182,"F = {}".format(popt[1]),fontdict = font)
	# plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	# plt.close()
	# show total plot
	plt.show()
