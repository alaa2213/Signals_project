
import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
from scipy import pi
from scipy.fftpack import fft
import math
B= 1024
𝑁 = 3*1024




Samples=6
time1 = np.linspace(0 ,3 , 12 * 1024)
#time=np.reshape(time1,(1,np.size(time1)))
freq1 = 261.63

freq2 = 130.81

frequencies_3rd= [130.81,146.83,164.81,174.61,196,220,246.93]
frequencies_4th= [261.63,293.66,329.63,349.23,392,440,493.88]
Tiarray=np.array([0.2,0.2,0.1,0.1,0.2,0.1])
Timearray=np.array([0,0.5,1,1.5,2,2.5])

def shifted_unit_step(t, t0):
    return np.heaviside(t-t0,1)
 
Ti=0.1

x = np.zeros(len(time1))
for j in range(Samples):
      t=j
     # Time delay for each term within the frequency
      x += np.sin(2*np.pi*frequencies_3rd[j]*time1) * (shifted_unit_step(time1,Timearray[j]) - shifted_unit_step(time1,Timearray[j]+Tiarray[j]))
     
     


x2=np.zeros(len(time1))

for i in range(Samples):
      t=i
     # Time delay for each term within the frequency
      x2 += np.sin(2*np.pi*frequencies_4th[i]*time1) * (shifted_unit_step(time1,Timearray[i]) - shifted_unit_step(time1,Timearray[i]+Tiarray[i]))
     
   
y=x2+x
#plt.plot(time, x)
#plt.plot(time, x2)

#sd.play(y, 3 *1024)
noise = np.random.normal (0,5, N) #takes the mean and standard deviation
f = np. linspace(0 , 512 , int(𝑁/2))
fn1,fn2= np. random. randint(0, 512,2)

x_f1 = fft(y)
x_f1 = 2/N * np.abs(x_f1 [0:int(N/2)])

n=np.sin(2*np.pi*fn1*time1)+np.sin(2*np.pi*fn2*time1)
xn=y+n

x_f = fft(xn)
x_f = 2/N * np.abs(x_f [0:int(N/2)])
noiseplace = np.where(x_f>math.ceil(np.max(y)))
ind1=noiseplace[0][0]
ind2=noiseplace[0][1]
place1 = int(f[ind1])
place2 = int(f[ind2])
xFiltered = xn - (np.sin(2*np.pi*place1*time1)+np.sin(2*np.pi*place2*time1))
x_fil = fft(xFiltered)
x_fil = 2/N * np.abs(x_fil [0:int(N/2)])

plt.figure()
plt.subplot(3,1,1)
plt.plot(f,x_f1)
plt.subplot(3,1,2)
plt.plot(f,x_f)
plt.subplot(3,1,3)
#plt.plot(f,x_f)
plt.plot(f, x_fil)
plt.show ()
  
