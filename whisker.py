import numpy as np
from scipy import signal
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.style as st
import matplotlib.patches as mpatches
import peakfinder as pf
import os
from start_frames_gcv_repaired import sfgcv
from start_frames_saline_repaired2 import sfsal
gcv_dir = '/Users/alexdeich/Documents/neuro_project/data/puff/csvs/'
st.use('ggplot')

def add_inx_col(dir):
    for fname in os.listdir(dir)[2:]:
        if fname.endswith('csv'):
            with open(dir + fname, 'r') as infile:
                indata = infile.read()
        
            tempname = fname + '.temp'
            with open(dir + tempname, 'w') as outfile:
                outdata = 'inx' + indata
                outfile.write(outdata)
        
            dir2 = dir.replace(' ', '\ ')
            dir2 += '/'
            os.system(f'rm {dir2 + fname}')
            os.system(f'mv {dir2 + tempname} {dir2 + fname}')
    

def get_bins(numbins):
    bins = np.linspace(-np.pi, np.pi, numbins)
    return(bins)
    
def get_bin_stats(data, bins):
    indices = np.digitize(data, bins)
    means = np.array([np.nanmedian(data[indices == i]) for i in range(0, len(bins))])
    stds = np.array([np.nanstd(data[indices == i]) for i in range(0, len(bins))])
    return(means, stds)

def file_loop(filename, numbins = 12):
    data = np.genfromtxt(filename, delimiter=',')
    meanbins = np.zeros((len(data) - 1, numbins))
    stdbins = np.zeros((len(data) - 1, numbins))
    bins = get_bins(numbins)
    for framenum in range(1, len(data)):
        frame = data[framenum][2:]
        if len(np.where(np.isnan(data))) < 0.2 * len(data):
            frame = denan(frame)
            means, stds = get_bin_stats(frame, bins)
            framenum -= 1
            meanbins[framenum] = means
            stdbins[framenum] = stds
    return(meanbins, stdbins)

def get_mean_freq(ff, Pxx_den):
    return(np.nansum(ff * Pxx_den)/np.nansum(Pxx_den))

def get_most_freq(ff, Pxx_den, freqpow):
    ff = ff[np.where(ff < 20)]
    Pxx_den = Pxx_den[np.where(ff < 20)]
    tot_pow = np.nansum(Pxx_den)
    ind90 = np.where(np.nancumsum(Pxx_den) > freqpow * tot_pow)[0][0]
    return(ff[ind90])

def make_aggregate_array(list_of_arrs):
    arr_len = 0
    for thing in list_of_arrs:
        arr_len += len(thing)

    output_arr = np.zeros(arr_len)

    start_point = 0

    for thing in list_of_arrs:
        output_arr[start_point:start_point + len(thing)] = thing
        start_point = start_point + len(thing)

    return(output_arr)

def get_phase(alldata):
    phase_list = []
    for whisker in alldata:
        data = np.copy(whisker)
        data = denan2(data)
        fft_dat = np.fft.fft(data)
        phase = np.angle(fft_dat[:20])
        phase_list.append(phase)
    return(np.array(phase_list))

def get_phase_diffs(all_phase_data):
    means = []
    stds = []
    for i in range(len(all_phase_data)):
        diffs = []
        for n in range(len(all_phase_data)):
            if i!=n:
                diff = np.abs(all_phase_data[i] - all_phase_data[n])
                diffs.append(diff)
        std = np.nanstd(diffs)
        mean = np.nanmedian(diffs)

        if std not in stds:
            stds.append(std)
        if mean not in means:
            means.append(mean)
    return(means, stds)

def get_freq_stats(freqpow, data, filename, filetype,n):
    if filetype == 'gcv':
        sfs = sfgcv
    else:
        sfs = sfsal
    sf = sfs[filename][0]
    
    data = denan2(data)
    try:
        ff, Pxx_den = signal.welch(data, fs = 500, nperseg = len(data))
        mf = get_most_freq(ff,Pxx_den, freqpow)
    except:
        mf = 0
    return(mf)

def nan_helper(y):
    return(np.isnan(y), lambda z: z.nonzero()[0])

def denan(arr):
    retarr = np.copy(arr)
    nans_in_arr = np.where(np.isnan(arr))
    nans, x = nan_helper(arr)
    try:
        retarr[nans] = np.interp(x(nans), x(~nans), retarr[~nans])
    except:
        retarr = np.zeros_like(retarr)
    return(retarr)

def denan2(arr):
    nonans = np.where(~np.isnan(arr))
    retarr = np.zeros_like(len(nonans))
    retarr = arr[nonans]
    return(retarr)
    
def signal_comp(s1, s2):
    minlen = min(len(s1), len(s2))                                                                                                                                                  
    s1 = zeroer(s1)[:minlen]                                                                                                   
    s2 = zeroer(s2)[:minlen]            
    return(s1+s2)   
     
def zeroer(s):                                                                                                                                                            
    if np.nanmean(s) > 0:                                                                                             
        return(s-np.nanmean(s))
    elif np.nanmean(s) < 0:
        return(s-np.nanmean(s))

def smooth(x,window_len=11,window='hanning'):

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat':
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def get_velocity(signal, dt = 0.1):
    shifted_forward = np.zeros(len(signal) + 1)
    shifted_back = np.zeros(len(signal) + 1)
    shifted_forward[1:] = signal
    shifted_back[:-1] = signal
    dx = shifted_back - shifted_forward
    dx = dx[1:-1]
    return(dx/dt)

def reconst(data, upper_bound_freq):
    fft = np.fft.fft(data)
    N = range(len(fft))        
    reconst_sig = np.array([f(fft[:upper_bound_freq], x, len(fft)) for x in N])
    orig = max(data)           
    new = max(reconst_sig)
    scaling_factor = orig/new
    reconst_sig *= scaling_factor
    return(reconst_sig - np.mean(reconst_sig) +  (np.mean(data))) 

def f(Y,x, N):    
    total = 0        
    for ctr in range(len(Y)):
        total += Y[ctr] * (np.cos(x*ctr*2*np.pi/N) + 1j * np.sin(x*ctr*2*np.pi/N))                
    return(np.real(total))

def make_mean_freq_plot(dir, freqpow, type, n=10, numbins = 12, sh = False):
    lm = []
    ls = []
    rm = []
    rs = []
    lpd = []
    rpd = []
    for filename in os.listdir(dir)[1:]:
        if filename in sfgcv.keys() or filename in sfsal.keys():
          print(filename)
          meandata = file_loop(dir + filename, numbins)[0].T
          stddata = file_loop(dir + filename, numbins)[1].T
          for i in range(len(meandata)):
              bindata = meandata[i]
              if np.nanmean(bindata) < 0:
                  if np.nanmean(stddata[i]) > 0:
                      ls.append(get_freq_stats(freqpow, stddata[i], filename, type,n))
                  else:
                      ls.append(0)
                  lm.append(get_freq_stats(freqpow, meandata[i], filename, type,n))
              elif np.nanmean(bindata) > 0:
                  if np.nanmean(stddata[i]) > 0:
                      rs.append(get_freq_stats(freqpow, stddata[i], filename, type,n))
                  else:
                      rs.append(0)
                  rm.append(get_freq_stats(freqpow, meandata[i], filename, type,n))
    lm = denan2(np.array(lm))
    rm = denan2(np.array(rm))
    ls = denan2(np.array(ls))
    rs = denan2(np.array(rs))
    num = min(len(lm), len(rm))
    plt.hist(lm[np.where(lm > 0.)], bins = 20, color='red', alpha = 0.4)
    plt.hist(rm[np.where(rm > 0.)], bins = 20, color='blue', alpha = 0.4)
    if sh:
        plt.show()
    return(lm, rm, ls, rs)
