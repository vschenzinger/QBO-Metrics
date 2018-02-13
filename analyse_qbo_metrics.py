import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


# ========================== Load relevant data
# Analysis needs
# Time (in calendaric years), called cal
# Pressure levels (in hPa), called press
# Latitude (in deg), called lat
# zmu[lat,level,time]
# zmT[lat,level,time]



# Useful numbers

nlat=len(lat)
nlev=len(press)
ntime=len(cal)

# =================  Retrieving the QBO metrics 
height=press_height(press)

# Analyse equatorial zmu
wind_eq=np.mean(zmu[abs(lat)<5,:,:],axis=0)
wind_eq_strat=wind_eq[np.logical_and(press>=10,press<=70),:]

spectrum,frequencies=get_spectrum(wind_eq,cal)

amps_prov=get_mean_famp(wind_eq,cal,1.5,3.)
lev_prov=find_values(amps_prov,np.nanmax(amps_prov))
print('Height of maximum QBO amplitude (provisional)')
print(press[lev_prov])

tests=spectrum[lev_prov,:].flatten()
testf=frequencies.flatten()

periods=get_periods(wind_eq[lev_prov,:],cal,ref=True,max_a=0.2,min_d=3.5)
qboamps=get_mean_famp(wind_eq,cal,min(periods),max(periods))
maxamp=np.nanmax(qboamps)
maxlev=find_values(qboamps,np.nanmax(qboamps))
lowamp=0.1*np.nanmax(qboamps)
lowlevs=press_height(get_vals_at(height,get_zeros(qboamps,val=lowamp)),reverse=True)
lowlev=max(lowlevs)
lev_fwhm=get_fwhm(height,qboamps)

windtest=wind_eq[lev_prov,:]
test={'wind':windtest}
save_file(test,'test')

#plt.plot()
#plt.plot(testf.flatten(),tests.flatten())
#plt.plot([min(periods),min(periods)],[0,50])
#plt.plot([max(periods),max(periods)],[0,50])
#plt.show()

#print(testf[np.logical_and(testf>=min(periods),testf<=max(periods))])
#print(tests[np.logical_and(testf>=min(periods),testf<=max(periods))])


zeros=get_zeros(wind_eq[maxlev,:],refine=True,max_a=0.2,min_d=3.5)

erates,wrates=get_descent_rates(wind_eq_strat,press)
erpz,wrpz=get_descent_rates(wind_eq_strat,press,per_zone=True)
match=min(len(erpz),len(wrpz))

amps_e, amps_w=get_amplitudes(wind_eq[maxlev,:],refine=True)

# Surrogate QBO timeseries for the estimation of error of min/max period and Fourier amplitude
nn=100
nc=1000
qbo_surr=surrogate_qbo(wind_eq[maxlev,:],n_cycles=nc)
s_pmin=np.zeros((nn))
s_pmax=np.zeros((nn))
s_famp=np.zeros((nn))

for ii in np.arange(0,nn):
    startind=random.randrange(0,len(qbo_surr)-ntime)
    surr_u=qbo_surr[startind:startind+ntime]
    surr_p=get_periods(surr_u,cal,ref=True)
    surr_f=get_mean_famp(surr_u,cal,min(surr_p),max(surr_p))
    s_pmin[ii]=np.nanmin(surr_p)
    s_pmax[ii]=np.nanmax(surr_p)
    s_famp[ii]=np.nanmax(surr_f)


# Analyse latitudinal structure of zmu at height of maximum QBO amplitude

wind_lat=zmu[:,maxlev,:]
wind_lat=wind_lat.reshape(nlat,ntime)
qboamps_lat=get_mean_famp(wind_lat,cal,min(periods),max(periods))
lat_fwhm=get_fwhm(lat,qboamps_lat)
lat_halfamp=0.5*np.nanmax(qboamps_lat)
lat_halfmax=get_vals_at(lat,get_zeros(qboamps_lat,val=lat_halfamp))

qbo_lat_height=get_mean_famp(zmu,cal,min(periods),max(periods))

# Analyse equatorial zmt
temp_eq=np.mean(zmt[abs(lat)<5,:,:],axis=0)
temp_eq_strat=temp_eq[np.logical_and(press>=10,press<=70),:]

spectrum_T,frequencies=get_spectrum(temp_eq,cal)

qboamps_T=get_mean_famp(temp_eq,cal,min(periods),max(periods))
plt.plot()
plt.yscale('log')
plt.plot(qboamps_T,press)
plt.plot([np.nanmax(qboamps_T),np.nanmax(qboamps_T)],[min(press),max(press)])
plt.plot([0.1*np.nanmax(qboamps_T),0.1*np.nanmax(qboamps_T)],[min(press),max(press)])
plt.show()

maxamp_T=np.nanmax(qboamps_T)
maxlev_T=find_values(qboamps_T,np.nanmax(qboamps_T))
lowamp_T=0.1*np.nanmax(qboamps_T)
lowlevs=press_height(get_vals_at(height,get_zeros(qboamps_T,val=lowamp_T,nans=True)),reverse=True)
lowlev_T=max(lowlevs[np.logical_and(lowlevs>=10,lowlevs<=100)]) # Confine to stratosphere
lev_fwhm_T=get_fwhm(height,qboamps_T)

# Analyse latitudinal structure of zmt at height of maximum QBO amplitude

temp_lat_T=zmt[:,maxlev_T,:]
temp_lat_T=temp_lat_T.reshape(nlat,ntime)
qboamps_lat_T=get_mean_famp(temp_lat_T,cal,min(periods),max(periods))
lat_fwhm_T=get_fwhm(lat[abs(lat)<15],qboamps_lat_T[abs(lat)<15])
lat_halfamp_T=0.5*np.nanmax(qboamps_lat_T)
lat_halfmax_T=get_vals_at(lat,get_zeros(qboamps_lat_T,val=lat_halfamp_T))

lat_halfmax_T=lat_halfmax_T[lat_halfmax_T<15] #Confine to tropical latitudes (polar signal can be quite large)

print(lat_halfmax_T)

qbo_lat_height_T=get_mean_famp(zmt,cal,min(periods),max(periods))
# ================= 6 Characteristic plots

plt.subplot(331)
plt.xscale('log')
plt.yscale('log')
plt.xlim([12,1./4.])
plt.ylim([100,1])
plt.plot([min(periods),min(periods)],[100,1])
plt.plot([max(periods),max(periods)],[100,1])
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
SP=plt.contourf(frequencies,press,spectrum,levels=[1,2,4,8,16,32],cmap='Greys',extend='both')
#SL=plt.contour(frequencies,press,spectrum,levels=[1,2,4,8,16,32],colors='black')
CB=plt.colorbar(SP)
CB.set_label('u (m/s)')
plt.title('Spectrum of equatorial ZMZ wind')
plt.xlabel('Period (year)')
plt.ylabel('Pressure (hPa)')

plt.subplot(332)
plt.yscale('log')
plt.ylim([100,1])
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
QBO_amp=plt.contour(lat,press,qbo_lat_height.T,levels=[1,2,4,8,16,32],colors='black')
QBO_amp_fill=plt.contourf(lat,press,qbo_lat_height.T,levels=[1,2,4,8,16,32],cmap='Greys',extend='both')
plt.title('QBO Fourier amplitude (Wind)')
plt.xlabel('Latitude (deg)')
plt.ylabel('Pressure (hPa)')

plt.subplot(333)
plt.yscale('log')
plt.ylim([100,1])
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.plot(qboamps,press)
plt.plot([maxamp,maxamp],[press[maxlev],press[maxlev]],'or')
plt.plot([lowamp,lowamp],[lowlev,lowlev],'ob')
plt.ylabel('Pressure (hPa)')
plt.xlabel('Amplitude (m/s), (K/10.)')
plt.title('QBO amplitude at the equator')
plt.plot(10.*qboamps_T,press,'--')
plt.plot([10.*maxamp_T,10.*maxamp_T],[press[maxlev_T],press[maxlev_T]],'or')
plt.plot([10.*lowamp_T,10.*lowamp_T],[lowlev_T,lowlev_T],'ob')

plt.subplot(334)
plt.xscale('log')
plt.yscale('log')
plt.xlim([12,1./4.])
plt.ylim([100,1])
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
SP=plt.contourf(frequencies,press,spectrum_T,levels=[0.05,0.1,0.2,0.4,0.8,1.6],cmap='Greys',extend='both')
#SL=plt.contour(frequencies,press,spectrum_T,levels=[0.05,0.1,0.2,0.4,0.8,1.6],colors='black')
CB=plt.colorbar(SP)
CB.set_label('u (m/s)')
plt.title('Spectrum of equatorial ZMZ temperature')
plt.xlabel('Period (year)')
plt.ylabel('Pressure (hPa)')

plt.subplot(335)
plt.yscale('log')
plt.ylim([100,1])
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
QBO_amp_T=plt.contour(lat,press,qbo_lat_height_T.T,levels=[0.05,0.1,0.2,0.4,0.8,1.6],colors='black')
QBO_amp_T_fill=plt.contourf(lat,press,qbo_lat_height_T.T,levels=[0.05,0.1,0.2,0.4,0.8,1.6],cmap='Greys',extend='both')
plt.title('QBO Fourier amplitude (Temperature)')
plt.xlabel('Latitude (deg)')
plt.ylabel('Pressure (hPa)')

plt.subplot(336)
plt.plot(lat,qboamps_lat)
plt.plot(lat_halfmax,[lat_halfamp,lat_halfamp],'ob')
plt.xlabel('Latitude')
plt.ylabel('Amplitude (m/s), (K/10.)')
plt.title('QBO amplitude at height of maximum QBO')
plt.plot(lat,10.*qboamps_lat_T,'--')
plt.plot(lat_halfmax_T,[10.*lat_halfamp_T,10.*lat_halfamp_T],'or')

plt.subplot(337)
plt.plot(cal,wind_eq[maxlev,:].flatten())
plt.plot(get_vals_at(cal,zeros),np.zeros(len(zeros)),'or')
plt.ylabel('ZMZ wind (m/s)')
plt.xlabel('Time (year)')
plt.title('Timeseries of equatorial zmu and cycle division')

plt.subplot(338)
plt.hist(periods*12,bins=np.arange(10,60))
plt.xlabel('Period (months)')
plt.ylabel('# Occurances')
plt.title('Period distribution (height of maximum QBO)')

plt.subplot(339)
plt.plot(wrpz[:match],erpz[:match],'og')
plt.plot([-2,2],[-2,2],'--')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel('Westerly descent rate (km/yr)')
plt.ylabel('Easterly descent rate (km/yr)')
plt.title('Descent rate of subsequent shear zones')


print('============  Characteristic QBO metrics of dataset (wind) ============')
print('============       (And their errors; if applicable)       ============')

print('Height of maximum (hPa)')
print(press[maxlev])

print('Fourier amplitude (m/s)')
print(maxamp)
print(np.std(s_famp))

print('Latitudinal extent (deg)')
print(lat_fwhm)

print('Vertical extent (km)')
print(lev_fwhm/1000.)

print('Lowest level (hPa)')
print(lowlev)

print('Min/Max/Mean period (months)')
print(min(periods)*12.,max(periods)*12.,np.mean(periods)*12.)
print(np.std(s_pmin)*12.,np.std(s_pmax)*12.,np.std(periods)*12.)

print('Mean Easterly/Westerly amplitude (m/s)')
print(np.mean(amps_e),np.mean(amps_w))
print(np.std(amps_e),np.std(amps_w))

print('Mean Easterly/Westerly descent rate (km/year)')
print(np.mean(erates),np.mean(wrates))
print(np.std(erates),np.std(wrates))

print('============  Characteristic QBO metrics of dataset (temperature)  ============')

print('Height of maximum (hPa)')
print(press[maxlev_T])

print('Fourier amplitude (K)')
print(maxamp_T)

print('Latitudinal extent (deg)')
print(lat_fwhm_T)

print('Vertical extent (km)')
print(lev_fwhm_T/1000.)

print('Lowest level (hPa)')
print(lowlev_T)





plt.show()
