import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ====================== Read in data =====================
# Necessary:
# zmu[lat,level,time]
# zmt[lat,level,time]
# time (potentially as calendaric date, but not necessary)
# latitude (degree)
# pressure levels



height=press_height(press)
wind=np.mean(zmu[abs(lat)<5,:,:],axis=0)
temp=np.mean(zmt[abs(lat)<5,:,:],axis=0)
startdate=np.floor(min(cal))
enddate=np.ceil(max(cal))

#print(press.shape)
#print(wind.shape)
#print(cal.shape)

plt.figure()

levels=np.arange(-50,55,5)
plt.subplot(211)
plt.yscale('log')
plt.xlim([startdate,enddate])
plt.ylim([max(press),min(press)])
CS1=plt.contourf(cal,press,wind,levels=levels,cmap='seismic',extend='both')
CS2=plt.contour(cal,press,wind,levels=[0],color='b')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())

CB=plt.colorbar(CS1)
CB.set_label('u (m/s)')
plt.title('Equatorial ZMZ wind (QBO)')
plt.xlabel('Time (year)')
plt.ylabel('Pressure (hPa)')

levels=np.arange(-5,5.5,0.5)
temp_a=declim(temp,clim=False)
plt.subplot(212)
plt.yscale('log')
plt.xlim([startdate,enddate])
plt.ylim([max(press),min(press)])
CS1=plt.contourf(cal,press,temp_a,levels=levels,cmap='seismic',extend='both')
CS2=plt.contour(cal,press,wind,levels=[0],color='b')
#CS2=plt.contour(cal,press,temp_a,levels=[0],color='b')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())

CB=plt.colorbar(CS1)
CB.set_label('$\Delta$ T (K)')
plt.title('Equatorial ZM temperature')
plt.xlabel('Time (year)')
plt.ylabel('Pressure (hPa)')


plt.show()


 

