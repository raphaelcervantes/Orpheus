import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

maintext = np.array(['Q = ','g = ','F = '])
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
cavity_length = np.linspace(12.5,15,num = 26, endpoint = True)
# print(cavity_length)
A = np.genfromtxt("delrin_modemap/MODEMAP_12.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
B = np.genfromtxt("delrin_modemap/MODEMAP_12.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
C = np.genfromtxt("delrin_modemap/MODEMAP_12.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
D = np.genfromtxt("delrin_modemap/MODEMAP_12.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
E = np.genfromtxt("delrin_modemap/MODEMAP_12.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
F = np.genfromtxt("delrin_modemap/MODEMAP_13.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
G = np.genfromtxt("delrin_modemap/MODEMAP_13.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
H = np.genfromtxt("delrin_modemap/MODEMAP_13.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
I = np.genfromtxt("delrin_modemap/MODEMAP_13.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
J = np.genfromtxt("delrin_modemap/MODEMAP_13.4.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
K = np.genfromtxt("delrin_modemap/MODEMAP_13.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
L = np.genfromtxt("delrin_modemap/MODEMAP_13.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
M = np.genfromtxt("delrin_modemap/MODEMAP_13.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
N = np.genfromtxt("delrin_modemap/MODEMAP_13.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
O = np.genfromtxt("delrin_modemap/MODEMAP_13.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
P = np.genfromtxt("delrin_modemap/MODEMAP_14.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
Q = np.genfromtxt("delrin_modemap/MODEMAP_14.1.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
R = np.genfromtxt("delrin_modemap/MODEMAP_14.2.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
S = np.genfromtxt("delrin_modemap/MODEMAP_14.3.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
T = np.genfromtxt("delrin_modemap/MODEMAP_14.4.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
U = np.genfromtxt("delrin_modemap/MODEMAP_14.5.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
V = np.genfromtxt("delrin_modemap/MODEMAP_14.6.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
W = np.genfromtxt("delrin_modemap/MODEMAP_14.7.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
X = np.genfromtxt("delrin_modemap/MODEMAP_14.8.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
Y = np.genfromtxt("delrin_modemap/MODEMAP_14.9.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
Z = np.genfromtxt("delrin_modemap/MODEMAP_15.0.csv",delimiter = ',',skip_header = 18,skip_footer = 1)
data = np.array([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R])
ydata = np.array([A[:,0],B[:,0],C[:,0],D[:,0],E[:,0],F[:,0],G[:,0],H[:,0],I[:,0],J[:,0],K[:,0],L[:,0],M[:,0],N[:,0],O[:,0],
                 P[:,0],Q[:,0],R[:,0],S[:,0],T[:,0],U[:,0],V[:,0],W[:,0],X[:,0],Y[:,0],Z[:,0]])
ydata = ydata/10**9
# print(ydata)
zdata = np.array([A[:,1],B[:,1],C[:,1],D[:,1],E[:,1],F[:,1],G[:,1],H[:,1],I[:,1],J[:,1],K[:,1],L[:,1],M[:,1],N[:,1],O[:,1],
                 P[:,1],Q[:,1],R[:,1],S[:,1],T[:,1],U[:,1],V[:,1],W[:,1],X[:,1],Y[:,1],Z[:,1]])
zdata_lin = 10**(zdata / 10)
# plt.plot(cavity_length,ydata)
xvals = np.ndarray(shape = ydata.shape)
# plt.pcolor(X,Y,zdata, cmap ='OrRd' )
for index in range(0,26):
    xvals[index] = cavity_length[index]
plt.pcolormesh(xvals, ydata,zdata , cmap = 'OrRd',vmin = -50, shading = 'gouraud')
plt.xlabel("Cavity Length (cm)")
plt.ylabel("Frequency (GHz)")
# print(xvals)
# plt.title("ModeMap")
c = plt.colorbar()
c.ax.set_ylabel("S21 (dB)", labelpad = 10)
# resonant_frequency = np.array([])
xdata = np.array([12.95,13.757,14.674])
frequency = np.array([17.1,16.15,15.33])
exp_frequency = np.array([16.8315,16.0588,15.394])
plt.plot(xdata,frequency,'b', marker = 'o', label = 'simulation')
plt.plot(xdata,exp_frequency,'black',marker = 'o', label = 'experiment')
plt.legend()
# plt.xlim(12.95,14.674)
plt.show()
# plt.savefig("modemap.pdf",bbox_inches = 'tight')
