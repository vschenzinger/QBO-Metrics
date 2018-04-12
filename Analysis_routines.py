#!/usr/bin/python
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy import stats
import random
import atmos_constants
from atmos_constants import *

# Define routines used for QBO analysis in climate models
# Contains functions for all QBO metrics ()
# Function for preparing a QBO surrogate timeseries
# Bootstrap test function

# Calculate climatological mean of data (time dimension last)
# Only works when data is in full years (i.e. time dimension is multiple of 12)

def clim_mean(data):
    dims=np.shape(data)
    ntime=dims[-1]
    newdims=dims[:-1]+(ntime/12,12)
    datam=np.reshape(data,newdims)
    cmean=np.mean(datam,axis=-2)
    return cmean

# Remove (climatological) mean from data (time index last)
# Returns data without (climatological) mean (i.e. deseasonalized or detrended)
# Only works when data is in full years (i.e. time dimension is multiple of 12)

def declim(data, clim=True):
    if clim:
        dims=np.shape(data)
        ndims=np.size(dims)
        ntime=dims[-1]
        if ntime % 12 != 0:
            print('Error: Time dimension should be multiple of 12')
            return
        else:
            newdims=dims[:-1]+(ntime/12,12)
            print(newdims)
            datam=np.reshape(data,newdims)
            cmean=clim_mean(data)
            declim_data=datam-cmean[np.newaxis,...]
            declim_data=np.reshape(declim_data,dims)
            return declim_data
    else:
        tmean=np.mean(data,axis=-1)
        de_data=data-tmean[...,np.newaxis]
	return de_data

# Find local extremes of function
# Input: function and type of extreme ('max' or 'min')
# Can refine output

def find_local_extremes(func, typ, refine=False, dist=None):
    sigs=[]
    if typ=='min':
        func=-1.*func
    for ii in range(1,len(func)):
        if func[ii]-func[ii-1] > 0:
            sigs=np.append(sigs,1)
        elif func[ii]-func[ii-1] == 0:
            sigs=np.append(sigs,0)
        elif func[ii]-func[ii-1] < 0:
            sigs=np.append(sigs,-1)            
    print(sigs)
    inds=np.arange(0,len(sigs))
    inds=inds[sigs!=0]
    pos=[]
    for ii in range(1,len(inds)):
        if sigs[inds[ii-1]] == 1 and sigs[inds[ii]] == -1:
            pos=np.append(pos,inds[ii])
    pos=pos.astype('int64')
    if refine:
        if not dist:
            dist=20
        for ii in range(1,len(pos)):
            if pos[ii]-pos[ii-1]<dist:
                if func[pos[ii-1]]<func[pos[ii]]:
                    pos[ii-1]=len(func)+1
        pos=pos[pos!=len(func)+1]
    return pos

# Find extreme values in between two zeros
# Input: function and type of extreme ('max' or 'min')

def find_cycle_extremes(func, typ, min_d=None, max_a=None, refine=False, singmax=False):
    func=func.flatten()
    if typ=='min':
        func*=-1.
    if refine:                                                        # Get the zero crossings
        zeropos=get_zeros(func, min_d=min_d, max_a=max_a, refine=True)
    else:
        zeropos=get_zeros(func, refine=False)
    zeropos=zeropos.astype(np.int64)
    cyclepos=zeropos[np.arange(0,len(zeropos))%2==0]
    maxpos=[]
    for ii in range(1,len(cyclepos)):
        locfunc=func[cyclepos[ii-1]:cyclepos[ii]]
        if not singmax:
            maxpos=np.append(maxpos,find_values(locfunc,np.nanmax(locfunc))+cyclepos[ii-1])
        else:
            maxpos=np.append(maxpos,np.nanmin(find_values(locfunc,np.nanmax(locfunc)))+cyclepos[ii-1])
    if 2*ii+1<len(zeropos) and func[zeropos[2*ii+1]]>0: # Last half cycle if uneven number of half cycles
        locfunc=func[zeropos[2*ii]:zeropos[2*ii+1]]
        if not singmax:
            maxpos=np.append(maxpos,find_values(locfunc,np.nanmax(locfunc))+zeropos[2*ii])
        else:
            maxpos=np.append(maxpos,np.nanmin(find_values(locfunc,np.nanmax(locfunc)))+zeropos[2*ii])        
    maxpos=maxpos.astype('int64')
    return maxpos
    
# Find all indices where an array equals a certain value

def find_values(func, val):
    inds=[]
    for ii in range(0,len(func)):
        if func[ii] == val:
            inds=np.append(inds,ii)
    inds=inds.astype(np.int64)
    return inds

# Find position of zero crossings (or places of another specified value)
# Does linear interpolation
# Will return all zeros unless refine is called
# Refine tests whether two adjacent zeros are too close (less then min_d)
# If the amplitude between the two zeros exceeds max_a*max(abs(function)), it counts as real
# max_a is a fraction => relative to maximum value of the absolute of the function
# i.e. the lower the value, the more zeros will remain
# Otherwise zeros get removed

def get_zeros(func, val=0., min_d=None, max_a=None, refine=False, nans=False):
    if nans:
	if np.nanmin(func)>val or np.nanmax(func)<val:
	    print('Function never is '+str(val))
	    return np.nan
    func = func.flatten()
    ftest = func - val               # For finding values other than zero
    if ftest[0] == 0:                 # Initialize array for (interpolated) zero positions
        zeros = [0]            
    else:
        zeros = []
    for ii in range(1,len(ftest)):
        zz=[]
        if ftest[ii] == 0.:
            zz=ii
        if np.sign(ftest[ii-1]) == -1 and np.sign(ftest[ii]) == 1:
            zz=np.abs(ftest[ii-1])/(np.abs(ftest[ii-1])+np.abs(ftest[ii]))+ii-1
        if np.sign(ftest[ii-1]) == 1 and np.sign(ftest[ii]) == -1:
            zz=np.abs(ftest[ii-1])/(np.abs(ftest[ii-1])+np.abs(ftest[ii]))+ii-1
        zeros=np.append(zeros,zz)
    if refine:                                  
        if not min_d:
            min_d=10
        if not max_a:
            max_a=0.3
        amplitude_limit_pos=max_a*np.nanmax(ftest)  # Introduce two limits for positive/negative amplitudes as QBO is asymmetric in strength
        amplitude_limit_neg=max_a*np.nanmin(ftest)
#        print(amplitude_limit_pos,amplitude_limit_neg)
        zero_dist=zeros[1:]-zeros[0:-1]        # Find zeros that are too close together
        a_crit=[]
        for ii in range(0,len(zero_dist)):      # and where the amplitude in between is too small
            cyc_amps=ftest[int(zeros[ii]):int(zeros[ii+1])]
            if len(cyc_amps)>1:
                if cyc_amps[1]>0:
                    test=np.nanmax(cyc_amps)<amplitude_limit_pos
                if cyc_amps[1]<0:
                    test=np.nanmin(cyc_amps)>amplitude_limit_neg
            else:
                if cyc_amps[0]>0:
                    test=np.nanmax(cyc_amps)<amplitude_limit_pos
                if cyc_amps[0]<0:
                    test=np.nanmin(cyc_amps)>amplitude_limit_neg
            a_crit=np.append(a_crit,test)           
        d_crit=zero_dist < min_d                # Array of critical points by criterion
        b_crit=np.logical_and(a_crit,d_crit)
        for ii in range(0,len(zero_dist)):  
            if b_crit[ii]:       # Remove if both criteria are true
                if ii<(len(zero_dist)-1) and b_crit[ii+1]: # Rare case of 3 critical crossings in a row => remove 1st and 3rd
#                    print 'Remove zeros '+str(zeros[ii])+', '+str(zeros[ii+2])+' (3 crit)'
#                    zeros[ii]=np.nan
#                    zeros[ii+2]=np.nan
#                    b_crit[ii+1]=False
#                    if ii+2<len(zero_dist):
#                        b_crit[ii+2]=False                    
                    if ftest[int(np.ceil(zeros[ii+2]))] < 0:  # Decide which of the zeros are removed based on whether it is a W-E or E-W phase change
                        zeros[ii]=np.nan
                        zeros[ii+1]=np.nan
                    else:
                        zeros[ii+1]=np.nan
                        zeros[ii+2]=np.nan
                        b_crit[ii+1]=False
                    b_crit[ii+1]=False                        
                    if ii+2<len(zero_dist):
                        b_crit[ii+2]=False 
                else:
#                    print 'Remove zeros '+str(zeros[ii])+', '+str(zeros[ii+1]) +' (2 crit)'        
                    zeros[ii]=np.nan
                    zeros[ii+1]=np.nan
        zeros=zeros[np.isfinite(zeros)]
        return zeros
    else:
        return zeros

# Get Easterly/Westerly amplitudes
# Takes wind timeseries and returns amplitudes as array
# For noisy data, refinement of zero positions is recommended
# (otherwise there might be some very low amplitudes detected)
# Can decide whether function should return empty or nan if 
# the value can't be found in the function

def get_amplitudes_old(func, min_d=None, max_a=None, refine=False):
    func=func.flatten()
    if refine:                                                        # Get the zero crossings
        zeropos=get_zeros(func, min_d=min_d, max_a=max_a, refine=True)
    else:
        zeropos=get_zeros(func, refine=False)
    for ii in range(0,len(zeropos)):
        zeropos[ii]=math.floor(zeropos[ii])
    zeropos=zeropos.astype(np.int64)
    if func[zeropos[0]+1]>0:                                        # Easterly or Westerly first?
        amp_one='West'
    else:
        amp_one='East'
    amps_max=[]
    amps_min=[]
    for ii in range(0,len(zeropos)-1):                             # Get the max/min amplitudes between 2 zeros
        amps_max=np.append(amps_max,max(func[zeropos[ii]:zeropos[ii+1]]))
        amps_min=np.append(amps_min,min(func[zeropos[ii]:zeropos[ii+1]]))
    if len(zeropos) % 2 == 0:
        no_amps=len(zeropos)/2-1
    else:
        no_amps=(len(zeropos)-1)/2
        amps_max=np.append(amps_max,np.nan)
        amps_min=np.append(amps_min,np.nan)
    amp_e=[]
    amp_w=[]
    if amp_one == 'East':                                         # Write amplitudes into array
        for ii in range(0,no_amps):
            amp_e=np.append(amp_e,amps_min[2*ii])
            amp_w=np.append(amp_w,amps_max[2*ii+1])
    if amp_one == 'West':
        for ii in range(0,no_amps):
            amp_e=np.append(amp_e,amps_min[2*ii+1])
            amp_w=np.append(amp_w,amps_max[2*ii])
    return amp_e,amp_w

def get_amplitudes(func, min_d=None, max_a=None, refine=False):
    func=func.flatten()
    amp_e=func[find_cycle_extremes(func, 'min', min_d=min_d, max_a=max_a, refine=refine, singmax=True)]
    amp_w=func[find_cycle_extremes(func, 'max', min_d=min_d, max_a=max_a, refine=refine, singmax=True)]
    return amp_e,amp_w

# Get values add odd places (e.g. value at times of zero crossings)
# Does linear interpolation => Only works with linearly spaced input

def get_vals_at(data, odds):
    data=data.flatten()
    vals = []
    if np.size(odds)==1:
        odds=np.array([odds])
    for ii in range(0,np.size(odds)):
	if np.isnan(odds[ii]):
	    vals=np.append(vals,np.nan)
	else:
            flo=int(math.floor(odds[ii]))
            if odds[ii]-flo != 0:                   # True "odd" place
                int_val=data[flo]+(odds[ii]-flo)*(data[flo+1]-data[flo])
            else:
                int_val=data[flo]
            vals=np.append(vals,int_val)
    return vals

# Get the Full width at half maximum of any function
# Needs 1d input of xvalues and yvalues
# Returns FWHM in units of xvalues
# Will return an error if the function does not have a clearly defined FWHM
# (i.e. a number of values equal to half max that's not equal to 2)
# Only gives a resonable value with linearly spaced input

def get_fwhm(xvals,yvals):
    y_halfmax=0.5*np.nanmax(yvals)
    x_halfmax=get_vals_at(xvals,get_zeros(yvals,val=y_halfmax))
    if len(x_halfmax)!=2:
        print('Error: Function does not have a clearly defined FWHM')
        return np.nan
    else:
        width=abs(x_halfmax[1]-x_halfmax[0])
        return width
    
# Get periods of a timeseries
# Uses the get_zeros function
# Default: Will calculate the time between every other zero crossing
# Can calculate the periods of the shear zones separately (per_zone option)

def get_periods(series, time, vv=0, min_d=None, max_a=None, ref=False, per_zone=False):
    data=series.flatten()
    if ref:
        zeros=get_zeros(data, val=vv, min_d=min_d, max_a=max_a, refine=True)
    else:
        zeros=get_zeros(data, val=vv) 
    zerotime=get_vals_at(time,zeros)
    if not per_zone:
        every_other=zerotime[::2]
        periods=every_other[1:]-every_other[0:-1]
        return periods
    else:
        periods=zerotime[1:]-zerotime[0:-1] # Calculate time between subsequent zero crossings
        pick=np.arange(0,len(periods))%2
        if data[int(np.ceil(zeros[0]))]>0:
            wper=periods[pick==0]
            eper=periods[pick==1]
        else:
            wper=periods[pick==1]
            eper=periods[pick==0]
        return wper,eper
    
# Calculate Fourier spectrum
# For N-dimensional arrays assumes
# last one is time index
# Takes the timeseries and the time index
# Gives back spectrum from 0 to fmax
# and corresponding frequencies in spec[amp,freq]
# Returns either the Fourier frequencies (freq=True) or the period
# Period unit is inverse of original time series sampling interval

def get_spectrum(data, time, fr=False, inv=False):
    datar=declim(data,clim=False)
    datafft=np.fft.ifft(datar,axis=-1)
    if inv:
        datafft=np.fft.fft(datar,axis=-1)  # Difference in scaling of transform vs. inverse transform
    amp_val=np.abs(datafft*np.conj(datafft))
    if len(time) % 2 == 0:
        fcut=len(time)/2
    else:
        fcut=(len(time)-1)/2
    freq=np.fft.fftfreq(len(time),d=time[1]-time[0])
    freq=freq[1:fcut]
    if not fr:
        freq=1/freq
    if data.ndim == 1:
        amp=amp_val[1:fcut]
    else:
        amp=amp_val[...,1:fcut]
    return amp, freq

# Gets Fourier amplitudes between set frequencies
# Takes timeseries, time index and bounds
# Bounds have to be in same unit as time index

def get_mean_famp(data, time, lower, upper):
    spec, freq = get_spectrum(data, time, fr=False)
    select = np.logical_and(freq>=lower,freq<=upper)
    amp_scaled = np.nansum(spec[...,select],axis=-1)/np.nansum(spec,axis=-1)*np.sqrt(np.nanvar(data,axis=-1))
    return amp_scaled

# Converts pressure coordinates (hPa)
# to height coordinates (m)
# Can be used in reverse mode (height(m) -> press(hPa))

def press_height(level,Pa=False,reverse=False):
    if not reverse:
        if not Pa:
            height=-scale_height*np.log(level/p_surf)
        else:
            height=-scale_height*np.log(level/(100.*p_surf))
        return height
    else:
        pressure=p_surf*np.exp(-np.asarray(level)/scale_height)
        return pressure

# Calculate descent rates
# from zero crossing in profile
# Takes zmz wind in form u[height,time]
# and pressure in hPa (option of Pa exists)
# Returns 2 arrays: rates(East), rates(West)
# in units of km/interval (depends on sampling interval)

def get_descent_rates(zmu, press, Pa=None, per_zone=False, prev=False, arrays=False):
    change_height=[]
    change_sign=[]
    east_rates=[]
    west_rates=[]
    if press[0]<press[-1]:
        prev=True
    if not Pa:                                               # Convert pressure to height in km
        height_m=press_height(press)/1000.
    else:
        height_m=press_height(press,Pa=True)/1000.
    n_levels=zmu.shape[0]
    n_time=zmu.shape[1]
    for tt in range(0,n_time):                               # Find height of phase change 
        profile=zmu[:,tt]
        changes=get_zeros(profile,0)   
        if len(changes) == 0:                                     # Put NaNs if there is no zero crossing in the profile
            change_height=np.append(change_height,np.nan)
            change_sign=np.append(change_sign,0.)
        else:
            if len(changes) == 1:                                 # Save height of the phase change
                chose_height=changes[0]
            else:                                                 # More than 1 zero crossing
                ref_height=change_height[tt-1]
                if np.isnan(ref_height):
                    chose_height=changes[np.where(get_vals_at(press,changes)==min(get_vals_at(press,changes)))]                  # In lack of reference height, choose the highest level
                else:
                    chose_height=changes[np.where(abs(get_vals_at(height_m,changes)-ref_height) == min(abs(get_vals_at(height_m,changes)-ref_height)))]
            change_height=np.append(change_height,get_vals_at(height_m,chose_height))             # Normal procedure: Take the zero crossing that is closest to the previous one
            if not prev:
                if profile[int(math.ceil(chose_height))]==0.:    # Rare case of value 0 in profile => use sign at level above for sign reference (unless on edge)
                    if int(math.ceil(chose_height))==n_levels-1:
                        change_sign=np.append(change_sign,-np.sign(profile[int(math.ceil(chose_height))-1]))
                    else:
                        change_sign=np.append(change_sign,np.sign(profile[int(math.ceil(chose_height))+1]))
                else:                                           # Sign of descending shear zone is relevant for attributing descent rate
                    change_sign=np.append(change_sign,np.sign(profile[int(math.ceil(chose_height))]))
            else:
                if profile[int(math.floor(chose_height))]==0.:    # Rare case of value 0 in profile => use sign at level above for sign reference (unless on edge)
                    if int(math.floor(chose_height))==n_levels-1:
                        change_sign=np.append(change_sign,-np.sign(profile[int(math.floor(chose_height))-1]))
                    else:
                        change_sign=np.append(change_sign,np.sign(profile[int(math.floor(chose_height))+1]))
                else:                                           # Sign of descending shear zone is relevant for attributing descent rate
                    change_sign=np.append(change_sign,np.sign(profile[int(math.floor(chose_height))]))                
    if not per_zone:
        for tt in range (1,n_time):                               # Calculate descent rates for the two shear zones
            if change_sign[tt-1] == change_sign[tt]:
                if change_sign[tt] == -1:
                    east_rates=np.append(east_rates,change_height[tt-1]-change_height[tt])
                if change_sign[tt] == 1:
                    west_rates=np.append(west_rates,change_height[tt-1]-change_height[tt])
        if not arrays:
            return east_rates, west_rates
        else:
            return east_rates, west_rates, change_height, change_sign
    if per_zone:
        east_rpz=[]
        west_rpz=[]
        current=[]
        tt=1
        for tt in range(1,n_time):
            if change_sign[tt]==change_sign[tt-1]:
                current=np.append(current,change_height[tt-1]-change_height[tt])
            else:
                if change_sign[tt-1]==-1:
                    east_rpz=np.append(east_rpz,np.nanmean(current))
                if change_sign[tt-1]==1:
                    west_rpz=np.append(west_rpz,np.nanmean(current))
                current=[]
        if not arrays:
            return east_rpz, west_rpz
        else:
            return east_rpz, west_rpz, change_height, change_sign       

# Construct a surrogate QBO timeseries (can be used for error estimation)
# Takes a timeseries as input, identifies QBO cycles (between 2 subsequent minima)
# Returns a timeseries of n_cycles (default: 1000)

def surrogate_qbo(func,n_cycles=1000):
    func=func.flatten()
    lmin=find_cycle_extremes(func,'min',refine=True)
    n_orig=len(lmin)-1
    qbo_surr=[]
    for ii in np.arange(0,n_cycles):
        ran_n=random.randrange(0,n_orig)
        qbo_surr=np.concatenate((qbo_surr,func[lmin[ran_n]:lmin[ran_n+1]]))
    return qbo_surr


    
# Define regression analysis
# Needs input y (predicted variable) as vector
# Needs input x (explanatory variables) as matrix

def reg_m(y, x):
    x = np.array(x).T,
    x = sm.add_constant(x)
    results = sm.OLS(endog=y, exog=x).fit()
    return results

# Bootstrap test - Testing whether a sample mean is based on a random selection
# Input: Mean to be tested (sample[space_dims])
#        Number of samples going into the mean
#        Basis distribution (Must be in same format as the sample, with additional dimension, e.g. time: basis[space_dims,time_dim])
#	 Optional: Number of bootstrap samples (default=10,000)

def bootstrap(sample,n_sample,basis,n_test=10000):
    # Get space and time dimensions
    dims=np.shape(sample)
    size=np.size(sample)
    sample=np.reshape(sample,size)
    basis=np.reshape(basis,(size,-1))
    ntime=np.size(basis)/size
    # Bootstrap routine: Create n_test sample means
    test_means=np.zeros((size,n_test))
    for itest in np.arange(0,n_test):
        # Select n_sample arrays from the basis
        pos=random.sample(xrange(ntime),n_sample)
        # Put their mean into the array of means
        test_means[:,itest]=np.mean(basis[:,pos],axis=1)
    # Get the percentile score of the sample compared to the random means
    percentiles=np.zeros(size)
    for pos in np.arange(0,size):
        percentiles[pos]=scipy.stats.percentileofscore(test_means[pos,:],sample[pos])
    percentiles=np.reshape(percentiles,dims)
    return percentiles


