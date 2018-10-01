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

cenpafolder = "new_simulations/empty_resonator_s21/"

filenames = ["orpheus_empty_q18_18GHz_L16cm_s21.csv",
             "orpheus_empty_q18_17p5GHz_L16p49cm_s21.csv",
             "orpheus_empty_q18_17GHz_L16p98cm_s21.csv",
             "orpheus_empty_q18_16p5GHz_L17p5cm_s21.csv",
             "orpheus_empty_q18_16GHz_L18p05cm_s21.csv",
             "orpheus_empty_q18_15p5GHz_L18p35cm_s21.csv",
             "orpheus_empty_q18_15GHz_L19p26cm_s21.csv",
             ]
pzeros = [
	[0.506, 18.036, 3000, 0],
	[0.423, 17.504, 3000, 0],
	[0.359, 17.003, 3000, 0],
	[0.550, 16.502, 3000, 0],
	[0.581, 16.003, 3000, 0],
	[0.538, 15.5, 3000, 0],
	[0.175, 15.006, 3000, 0],
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
	plt.plot(xdata, ydata, 'b-', label = 'data')

	popt, pcov = curve_fit(freqToPower, xdata, ydata_lin, p0 = pzeros[index], maxfev = 100000000)# runs the curve fit
	plt.plot(xdata,10*np.log10(freqToPower(xdata, * popt)), 'r-', label = 'fit')
	plt.xlabel('frequency')
	plt.ylabel('S21(dB)')
	plt.legend()
	popt = np.round(popt, decimals = 4)
	if index == 0:
		plt.text(17.925,-130,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.925,-133,"F = {}".format(popt[1]),fontdict = font)
		#plt.text(17.85,-15,str_q\nstr_f,fontdict = font)
	elif index == 1:
		plt.text(17.4,-110,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.4,-115,"F = {}".format(popt[1]),fontdict = font)
	elif index == 2:
		plt.text(16.9,-120,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.9,-125,"F = {}".format(popt[1]),fontdict = font)
	elif index == 3:
		plt.text(16.4,-130,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.4,-133,"F = {}".format(popt[1]),fontdict = font)
	elif index == 4:
		plt.text(15.9,-130,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.9,-133,"F = {}".format(popt[1]),fontdict = font)
	elif index == 5:
		plt.text(15.4,-132,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.4,-135,"F = {}".format(popt[1]),fontdict = font)
	elif index == 6:
		plt.text(14.9,-140,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(14.9,-143,"F = {}".format(popt[1]),fontdict = font)
	# plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	# plt.close()
	# show total plot
	plt.show()
