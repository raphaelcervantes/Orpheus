import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import math
def resonant_frequency(q,m,n,d):
	f0 = (3*10**8)/(4*d/100)
	pi = math.pi
	acos = math.acos
	f = 2*(q+1)*f0 + (f0/pi)*(1+m+n)*acos(1-((2*d)/33.02))
	# 2*(q+1)*((3*10**8)/(4*(d/100)))+(((3*10**8)/(4*(d/100)))/math.pi)*(1+m+n)*math.acos(1-((2*d)/34.59))
	return f
def freqToPower(f, a, fo, q, c):
    return a * 10**-10 * (1 / ((f - fo)**2 + (fo / (2 * q))**2)) + c
maintext = np.array(['Q = ','g = ','F = '])
font = {'color':  'black',
        'weight': 'heavy',
        'size': 10,
        }
# cavity_length = np.linspace(16.0,19.3,num = 34, endpoint = True)
# # print(cavity_length)
# A = np.genfromtxt("empty_modemap/EXP_MAP_16.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# B = np.genfromtxt("empty_modemap/EXP_MAP_16.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# C = np.genfromtxt("empty_modemap/EXP_MAP_16.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# D = np.genfromtxt("empty_modemap/EXP_MAP_16.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# E = np.genfromtxt("empty_modemap/EXP_MAP_16.4.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# F = np.genfromtxt("empty_modemap/EXP_MAP_16.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# G = np.genfromtxt("empty_modemap/EXP_MAP_16.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# H = np.genfromtxt("empty_modemap/EXP_MAP_16.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# I = np.genfromtxt("empty_modemap/EXP_MAP_16.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# J = np.genfromtxt("empty_modemap/EXP_MAP_16.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# K = np.genfromtxt("empty_modemap/EXP_MAP_17.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# L = np.genfromtxt("empty_modemap/EXP_MAP_17.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# M = np.genfromtxt("empty_modemap/EXP_MAP_17.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# N = np.genfromtxt("empty_modemap/EXP_MAP_17.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# O = np.genfromtxt("empty_modemap/EXP_MAP_17.4.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# P = np.genfromtxt("empty_modemap/EXP_MAP_17.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# Q = np.genfromtxt("empty_modemap/EXP_MAP_17.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# R = np.genfromtxt("empty_modemap/EXP_MAP_17.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# S = np.genfromtxt("empty_modemap/EXP_MAP_17.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# T = np.genfromtxt("empty_modemap/EXP_MAP_17.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# U = np.genfromtxt("empty_modemap/EXP_MAP_18.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# V = np.genfromtxt("empty_modemap/EXP_MAP_18.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# W = np.genfromtxt("empty_modemap/EXP_MAP_18.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# X = np.genfromtxt("empty_modemap/EXP_MAP_18.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# Y = np.genfromtxt("empty_modemap/EXP_MAP_18.4.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# Z = np.genfromtxt("empty_modemap/EXP_MAP_18.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# AA = np.genfromtxt("empty_modemap/EXP_MAP_18.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# BB = np.genfromtxt("empty_modemap/EXP_MAP_18.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# CC = np.genfromtxt("empty_modemap/EXP_MAP_18.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# DD = np.genfromtxt("empty_modemap/EXP_MAP_18.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# EE = np.genfromtxt("empty_modemap/EXP_MAP_19.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# FF = np.genfromtxt("empty_modemap/EXP_MAP_19.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# GG = np.genfromtxt("empty_modemap/EXP_MAP_19.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# HH = np.genfromtxt("empty_modemap/EXP_MAP_19.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
# data = np.array([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R])
# ydata = np.array([A[:,0],B[:,0],C[:,0],D[:,0],E[:,0],F[:,0],G[:,0],H[:,0],I[:,0],J[:,0],K[:,0],L[:,0],M[:,0],N[:,0],O[:,0],
#                  P[:,0],Q[:,0],R[:,0],S[:,0],T[:,0],U[:,0],V[:,0],W[:,0],X[:,0],Y[:,0],Z[:,0],AA[:,0],BB[:,0],CC[:,0],DD[:,0],
#                  EE[:,0],FF[:,0],GG[:,0],HH[:,0]])
# ydata = ydata/10**9
# # print(ydata)
# zdata = np.array([A[:,1],B[:,1],C[:,1],D[:,1],E[:,1],F[:,1],G[:,1],H[:,1],I[:,1],J[:,1],K[:,1],L[:,1],M[:,1],N[:,1],O[:,1],
#                  P[:,1],Q[:,1],R[:,1],S[:,1],T[:,1],U[:,1],V[:,1],W[:,1],X[:,1],Y[:,1],Z[:,1],AA[:,1],BB[:,1],CC[:,1],DD[:,1],
#                  EE[:,1],FF[:,1],GG[:,1],HH[:,1]])
# zdata_lin = 10**(zdata / 10)
# # plt.plot(cavity_length,ydata)
# xvals = np.ndarray(shape = ydata.shape)
# # plt.pcolor(X,Y,zdata, cmap ='OrRd' )
# for index in range(0,34):
#     xvals[index] = cavity_length[index]
# plt.pcolormesh(xvals, ydata,zdata , cmap = 'OrRd',vmin = -60)
# plt.xlabel("Cavity Length (cm)",fontsize = 'large')
# plt.ylabel("Frequency (GHz)",fontsize = 'large')
# # print(xvals)
# # plt.title("ModeMap")
# c = plt.colorbar()
# c.ax.set_ylabel("S21 (dB)", labelpad = 10,fontsize = 'large')
# xdata = np.array([16.1, 16.5, 17, 17.5, 18, 18.5, 19, 19.3])
# sim_xdata = np.array([16, 16.5, 17, 17.5, 18, 18.635, 19.26])
# frequency = np.array([18,17.504,17.003,16.512,16.0054,15.507,15.0079])
# predicted_frequency = np.array([17.931,17.499,16.989,16.508,16.054,15.624,15.216,15.0129])
# frequency_q17 = np.ndarray(shape = (7,1))
# frequency_q19 = np.ndarray(shape = (7,1))
# for i in range(0,7):
#     frequency_q17[i] = resonant_frequency(17,0,0,sim_xdata[i])
# frequency_q17 = frequency_q17/10**9
# for i in range(0,7):
#     frequency_q19[i] = resonant_frequency(19,0,0,sim_xdata[i])
# frequency_q19 = frequency_q19/10**9
# fre_q_17_m1_n1 = np.ndarray(shape = (7,1))
# for i in range(0,7):
#     fre_q_17_m1_n1[i] = resonant_frequency(17,1,1,sim_xdata[i])
# fre_q_17_m1_n1 = fre_q_17_m1_n1/10**9
# fre_q_18_m1_n1 = np.ndarray(shape = (7,1))
# for i in range(0,7):
#     fre_q_18_m1_n1[i] = resonant_frequency(18,1,1,sim_xdata[i])
# fre_q_18_m1_n1 = fre_q_18_m1_n1/10**9
# plt.plot(sim_xdata,frequency,'w', marker = 'o', label = 'Simulation q =18')
# plt.plot(xdata,predicted_frequency,'k',marker = 'o', label = 'Predicted q = 18')
# plt.plot(sim_xdata,frequency_q17,'g',marker = 'o',linewidth = 2.5)
# plt.plot(sim_xdata,frequency_q19,'teal',marker = '>',linewidth = 2.5)
# plt.plot(sim_xdata,fre_q_17_m1_n1,'m',marker = 'o')
# plt.plot(sim_xdata,fre_q_18_m1_n1,'b',marker = 'o')
# plt.legend(fontsize = 'small', framealpha = 0.3)
# plt.ylim(15,18)
# plt.text(17.2,17.75,"0,0,19",fontdict = font,color ='teal')
# plt.text(16,16.5,"0,0,17",fontdict = font,color = 'g')
# plt.text(18.2,15.1,"1,1,17",fontsize = 'small',color = 'm',weight = 'heavy')
# plt.text(18.7,15.5,"1,1,18",fontsize = 'small',color = 'b',weight = 'heavy')
# plt.text(16,17.6,"0,0,18",fontsize = 'small', weight = 'heavy')
# # plt.show()
# # plt.savefig("modemap.pdf",bbox_inches = 'tight')
# # plt.close()
# plt.plot(ydata[11],zdata[11])
# plt.xlabel("Frequency (GHz)", fontsize = 'large')
# plt.ylabel("S21 (dB)",fontsize = 'large')
# plt.text(16.85,-10,"q = 18")
# plt.text(15.96,-10,"q = 17")
# plt.text(17.7,-10,"q = 19")
# plt.text(14.9,-7,"Cavity Length = 17cm",fontsize = 'small')
# plt.show()
# plt.savefig("modemap_data.pdf",bbox_inches = 'tight')
# plt.close()
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
