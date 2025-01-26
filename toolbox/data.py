"""
Data management module
"""

from bs4 import BeautifulSoup
from datetime import datetime as dt
import getpass
from glob import glob
import netCDF4
import numpy as np
import os
import pandas as pd
import requests
import sys
import time
from urllib import request
from urllib.error import *

def ndbc_dl(dstdir:str, station:str, years:list, dtype="stdmet", merge=True):
    """
    Fetches met-ocean data from the NDBC webste. All downloaded files will be 
    saved in the directory specified <br>
    NDBC homepage: https://www.ndbc.noaa.gov/obs.shtml

    Parameters
    ----------
    dstdir : str
        Output directory storing downloaded data
    station : str
        Station ID
    years : list or int
        List of years of which data will be downloaded
    dtype : str, default "stdmet"
        Historical data type. Available options are "stdmet", "cwind" and 
        "wspec" only, which stands for standard meteorological, continuous wind
        , and wave spectra respectively
    merge : bool, default True
        Merge data across years into one single file. stdmet and cwind data
        will be concatenated into a single dataframe, while wspec data will be
        merged into a netCDF file
    """

    # NDBC query dictionary
    NDBC_dict = {
        "stdmet": [['h', "stdmet"]],
        "cwind": [['c', "cwind"]],
        "wspec": [
            ['w', "swden"],
            ['d', "swdir"],
            ['i', "swdir2"],
            ['j', "swr1"],
            ['k', "swr2"]
        ]
    }

    # Column rename dictionary
    colname_dict = {
        "#YY": "YYYY",
        "YY": "YYYY",
        "WD": "WDIR",
        "DIR": "WDIR",
        "BAR": "PRES",
        "GMN": "GTIME"
    }

    # Check input requirements
    assert dtype in ["stdmet", "cwind", "wspec"], "Incorrect request data " +\
        "type. Available options are 'stdmet', 'cwind', and 'wspec' only.\n"

    # Pre-processing
    sname = station.lower()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
              "Sep", "Oct", "Nov", "Dec"]
    subdirs = [q[1] for q in NDBC_dict[dtype]]
    [os.makedirs(os.path.join(dstdir, subdir), exist_ok=True) for subdir in subdirs]

    # Path functions
    url1 = lambda q, year: "https://www.ndbc.noaa.gov/view_text_file.php?" + \
        "filename=%s%c%d.txt.gz&dir=data/historical/%s/"%(sname, q[0], year, q[1])
    url2a = lambda q, year, idx, month: "https://www.ndbc.noaa.gov/view_text_file.php?" +\
        "filename=%s%d%d.txt.gz&dir=data/%s/%s/"%(sname, idx+1, year, q[1], month)
    #url2b = lambda q, month: "https://www.ndbc.noaa.gov/data/" +\
    #    "%s/%s/%s.txt"%(q[1], month, sname)

    # Download data 
    for q in NDBC_dict[dtype]:
        for year in years:

            # Try retrieving data for the whole year
            try:
                fp = os.path.join(dstdir, q[1], "%d.txt"%year)
                request.urlretrieve(url1(q, year), fp)
                print("Downloaded %s data for year %d from station %s."%(q[1],
                                                    year, station))
            
            # If the page doesn't exist
            except HTTPError:
                
                # Try retrieving data by month
                for idx, month in enumerate(months):
                    try:
                        fp = os.path.join(dstdir, q[1], "%d_%02d.txt"%(year, idx+1))
                        request.urlretrieve(url2a(q, year, idx, month), fp)
                        print("Downloaded %s data for month %d-"%(q[1], year) +
                              "%02d from station %s."%(idx+1, station))
                    
                    # If the month page doesn't exist as well
                    except HTTPError:
                        print("No %s data for month %d-"%(q[1], year) +
                            "%02d at station %s."%(idx+1, station))
                        
                        """# Most recent month
                        try:
                            request.urlretrieve(url2b(q, month), fp)
                            print("Downloaded %s data for month %d-"%(q[1], year) +
                              "%02d from station %s."%(idx+1, station))
                        
                        except HTTPError:"""
                        
    # Merge monthly data if they exist, then remove the monthly files
    for q in NDBC_dict[dtype]:
        flist = glob(os.path.join(dstdir, q[1], "*_*"))
        
        try:
            fname = os.path.join(dstdir, q[1], flist[0].\
                                 split("\\")[-1].split('_')[0] + ".txt")
        except IndexError:
            break

        # Loop through each month's data and add to master file
        for idx, fmonth in enumerate(flist):
            data = open(fmonth, 'r').read().split("\n")

            # First month
            if idx == 0:
                f = open(fname, 'w')
                f.writelines([line + "\n" for line in data[:-1]])
            
            # Other months
            else:
                f = open(fname, 'a')
                f.writelines([line + "\n" for line in data[2:-1]])
            
            f.close()
            os.remove(fmonth)
        
        print("Monthly %s data merged."%q[1])
        
    # Merge data across years, using present day format
    # Ignore files starting with the character 'M'
    if merge & (dtype in ["stdmet", "cwind"]):
        flist = glob(os.path.join(dstdir, dtype, "[!M]*"))
        
        data_agg = []
        for f in flist:
            
            # Read first 5 rows and get column names
            dummy = pd.read_csv(f, nrows=5, sep="\s+")
            columns = dummy.columns
            
            # Skip 2 rows if data is obtained on or after 2007, otherwise 
            # skip 1 row only
            skiprows = 2 if int(f.split("\\")[-1].split('.')[0]) >= 2007 else 1
            data_add = pd.read_csv(f, skiprows=skiprows, sep="\s+",
                                   names=columns)
            
            # Process data and convert columns
            data_add = data_add.rename(columns=colname_dict)
            data_add["YYYY"] = data_add["YYYY"].apply(lambda x: "19%d"%int(x)
                        if int(x) < 99 else int(x))
            
            if "mm" not in columns:
                data_add["mm"] = "00"
            
            data_agg.append(data_add)
        
        data_agg = pd.concat(data_agg, ignore_index=True)

        # Final furnishing
        colnameagg = data_agg.columns
        misc_columns = colnameagg.drop(["YYYY", "MM", "DD", "hh", "mm"])
        colnameagg = ["YYYY", "MM", "DD", "hh", "mm"] + misc_columns.tolist()
        data_agg = data_agg[colnameagg]

        # Output merged file
        data_agg.to_csv(os.path.join(dstdir, dtype, "Merged.txt"), index=False)
        print("%s data fully merged."%dtype)
    
    # Merge spectral files into a netcdf file
    elif merge & (dtype == "wspec"):
        _spec2nc(dstdir)
        print("%s data fully merged."%dtype)

    return None

def vessel_name_lookup(MMSI:str):
    """
    Vessel name lookup using an MMSI number. <br>
    Database used: https://www.vesselfinder.com/vessels

    Parameters
    ----------
    MMSI : str
        A unique 9-digit all-numeric idenfier for vessels

    Returns
    -------
    name : str
        Name of the vessel corresponding to the MMSI number. Returns "N/A" if 
        a matching vessel cannot be found
    """

    # Cast inputs to string type
    MMSI_request = str(MMSI)

    # Define URL and haeders for parsing information
    url = "https://www.vesselfinder.com/vessels?name=%s"%MMSI_request
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/" +\
        "537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203",
        "accept-language": "en-GB,en;q=0.9,en-US;q=0.8"
    }

    # Attempt to parse data. If data is not available "N/A" will be used as the 
    # vessel name instead
    try:
        html = requests.get(url, headers=headers, verify=False)
        soup = str(BeautifulSoup(html.text, "html.parser"))
        name = str(soup).split('<div class="slna">')[-1].split("</div")[0]
    
    except:
        name = "N/A"
    
    return name

def _spec2nc(srcdir, dstdir=None, dtheta=5, fixNegative=False):
    """
    Code to convert NDBC spectral data files to netCDF format. Code only works
    for files with newer formats (YY MM DD hh mm) 

    References
    ----------
    Kuik, A.J., G.Ph. van Vledder, and L.H. Holthuijsen, 1998: "Method for
      the Routine Analysis of Pitch-and-Roll Buoy Wave Data", Journal of
      Physical Oceanography, 18, 1020-1034.
    
    Parameters
    ----------
    srcdir : str
        Directory containing raw spectral files downloaded from NDBC
    dstdri : str, default None
        Directory to save the merged netCDF file
    dtheta : float, default 5
        Directional resolution for the reconstruction of the frequency-
        direction spectrum in degrees
    fixNegative : bool, default False
        If true, redistributes negative energy into the positive
        components of the spectrum
    """
    
    # Set destination file 
    if dstdir is None:
        dstdir = srcdir
    
    # Construct directional angle
    angles = np.arange(0.0,360.0,dtheta)
    
    # Time reference
    basetime = dt(1900,1,1,0,0,0)

    #===========================================================================
    # Read file information
    #===========================================================================
    
    # Get years with data
    fpaths = glob(os.path.join(srcdir, "swden", "*.txt"))
    years = sorted([x.split("\\")[-1].split('.')[0] for x in fpaths])   
    
    # Create output netcdf file ------------------------------------------------
    # Global attributes  
    nc = netCDF4.Dataset(dstdir + "wspec.nc", 'w', format="NETCDF4")
    nc.Description = "NDBC Spectral Data"
    nc.Rawdata = 'National Data Buoy Center \nwww.ndbc.noaa.gov'
    nc.Author = getpass.getuser()
    nc.Created = time.ctime()
    nc.Software = 'Created with Python ' + sys.version
    nc.NetCDF_Lib = str(netCDF4.getlibversion())
    nc.Script = os.path.realpath(__file__)
    nc.Notes = 'Nautical convention used for directions'    
    

    # Reconstruct the spectrum -------------------------------------------------        

    # counter variable to create variables in the netcdf file
    masterCnt = 0
    cnt_freq = 0 
    cnt_dir = 0
    tstep_freq = 0
    tstep_dir = 0
    
    # This variable will change if the format changes
    formIdCnt = 0
    formId = '0'
            
    # Frequency array to find if the reported frequencies changed
    freqArray = []
    
    # Loop over years
    for aa in years:
        
        # Info
        print('  Working on year ' + aa)
        tmpfile = srcdir + "swden/%s.txt"%aa
        
        # Info
        print('    Spectral density data')
        
        # Increase master counter variable
        masterCnt += 1
        
        # Read spectral density data (frequency spectra) and identify the time
        # information given
        f_w = open(tmpfile,'r')
        freq = f_w.readline().split()
               

        # Allocate the frequency array and determine if the format changed ----
        freqArray.append(freq)
        if masterCnt > 1:
            if freq != freqArray[masterCnt-2]:
                
                # Reset counter variables                
                cnt_freq = 0 
                cnt_dir = 0
                tstep_freq = 0
                tstep_dir = 0
                
                # Update form Id counter
                formIdCnt += 1
                
                if formIdCnt > 9:
                    print('\n10 Different Formats Found')
                    print('Check your data, quitting ...')
                    nc.close()
                    sys.exit()
                    
                # Update form Id text
                formId = '%01.0f' % formIdCnt
                
                # Message to user
                print('    Different format found')
                print('      New variables introduced')
                
               
        # Find if minutes are given
        if freq[4] == 'mm':
            freqInd0 = 5
        else:
            freqInd0 = 4
               
        # Read frequencies
        freq = np.array(freq[freqInd0:],dtype=float)
        f_w.close()
        
        # Load spectral density
        freq_spec = np.loadtxt(tmpfile,skiprows=1)
        
        # Allocate time and spectral density data    
        freq_time = np.zeros((freq_spec.shape[0]))  
        for bb in range(freq_time.shape[0]):
            tmpYear  = int(freq_spec[bb,0])
            tmpMonth = int(freq_spec[bb,1])
            tmpDay   = int(freq_spec[bb,2])
            tmpHour  = int(freq_spec[bb,3])
            if freqInd0 == 4:
                tmpMin = int(0)
            else:
                tmpMin = int(freq_spec[bb,4])
            
            if tmpYear < 100:
                tmpYear = tmpYear + 1900
            
            freq_time[bb] = (dt(tmpYear,tmpMonth,tmpDay,
                                               tmpHour,tmpMin) -
                             basetime).total_seconds()

        freq_spec = freq_spec[:,freqInd0:]
        
        # No Data Filter (NDBC uses 999.00 when there is no data)
        goodDataInd = freq_spec[:,1] < 990.00
        freq_time = freq_time[goodDataInd]
        freq_spec = freq_spec[goodDataInd,:]
        
        # Create frequency spectra variables
        cnt_freq += 1
        if cnt_freq == 1:
            
            # Create dimensions  (NetCDF4 supports multiple unlimited dimensions)
            nc.createDimension('wave_time'+formId,None)
        
            # Create bulk parameter variables
            nc.createVariable('Hsig'+formId,'f8','wave_time'+formId)
            nc.variables['Hsig'+formId].units = 'meter'
            nc.variables['Hsig'+formId].long_name = 'Significant wave height'
            
            # Create frequency dimension
            nc.createDimension('freq'+formId,freq.shape[0])
            
            nc.createVariable('wave_time'+formId,'f8','wave_time'+formId)
            nc.variables['wave_time'+formId].units = \
            "seconds since 1900-01-01 00:00:00"
            nc.variables['wave_time'+formId].calendar = "julian"
            
            nc.createVariable('freq_spec'+formId,'f8',
                              ('wave_time'+formId,'freq'+formId))
            nc.variables['freq_spec'+formId].units = 'meter2 second'
            nc.variables['freq_spec'+formId].long_name = 'Frequency variance spectrum'            
            
            nc.createVariable('frequency'+formId,'f8',('freq'+formId))
            nc.variables['frequency'+formId].units = 'Hz'
            nc.variables['frequency'+formId].long_name = 'Spectral frequency'
            nc.variables['frequency'+formId][:] = freq
        
    
        # Information
        print('    Computing Bulk Parameters')
        
        # Compute bulk parameters
        moment0 = np.trapezoid(freq_spec.T,freq,axis=0)
        Hsig = 4.004*(moment0)**0.5
        
        # Write to NetCDF file
        if cnt_freq == 1:
            nc.variables['Hsig'+formId][:] = Hsig
            nc.variables['freq_spec'+formId][:] = freq_spec
            nc.variables['wave_time'+formId][:] = freq_time
        else:
            nc.variables['Hsig'+formId][tstep_freq:] = Hsig
            nc.variables['freq_spec'+formId][tstep_freq:,:] = freq_spec
            nc.variables['wave_time'+formId][tstep_freq:] = freq_time

                
        # Check if directional data exists -------------------------------------
        tmp_alpha_1 = srcdir + "swdir/%s.txt"%aa
        tmp_alpha_2 = srcdir + "swdir2/%s.txt"%aa
        tmp_r_1 = srcdir + "swr1/%s.txt"%aa
        tmp_r_2 = srcdir + "swr2/%s.txt"%aa
    
        if (os.path.isfile(tmp_alpha_1) and os.path.isfile(tmp_alpha_2) and
            os.path.isfile(tmp_r_1) and os.path.isfile(tmp_r_2)):

            # Information
            print('    Directional Data')

            # Read spectral data
            try:
                alpha_1 = np.loadtxt(tmp_alpha_1,skiprows=1)
                alpha_2 = np.loadtxt(tmp_alpha_2,skiprows=1)
                r_1 = np.loadtxt(tmp_r_1,skiprows=1) * 0.01
                r_2 = np.loadtxt(tmp_r_2,skiprows=1) * 0.01
            except:
                print('      Error reading the data')
                continue

            # Some years do not have consistent data. I will not attempt to 
            # fix that here, just skip this year.
            tmplen = alpha_1.shape[0]
            if not all([lst.shape[0] == tmplen for lst in [alpha_2, r_1, r_2]]):
                print('    Inconsistent data, skipping year')
                continue

            
            # Read frequency of the directional spectra (not always agree with
            # the spectral densities)
            f_w2 = open(tmp_alpha_1,'r')
            freqDirSpec = f_w2.readline().split()
            if freqDirSpec[4] == 'mm':
                freqInd0 = 5
            else:
                freqInd0 = 4
            freqDirSpec = np.array(freqDirSpec[freqInd0:],dtype=float)
            f_w2.close()  
            
            # Create directional spectra variables
            cnt_dir += 1
            if cnt_dir == 1:
                nc.createDimension('dir_time'+formId,None)
                nc.createDimension('dir'+formId,angles.shape[0])
                
                # Create frequency dimension
                nc.createDimension('freqDir'+formId,freqDirSpec.shape[0])
            
                nc.createVariable('dir_time'+formId,'f8','dir_time'+formId)
                nc.variables['dir_time'+formId].units = \
                "seconds since 1900-01-01 00:00:00"
                nc.variables['dir_time'+formId].calendar = "julian"
            
                nc.createVariable('dir_spec'+formId,'f8',
                                  ('dir_time'+formId,'freqDir'+formId,
                                   'dir'+formId))
                nc.variables['dir_spec'+formId].units = 'meter2 second degree-1'
                nc.variables['dir_spec'+formId].long_name = \
                    'Frequency-Direction variance spectrum'  
                    
                nc.createVariable('direction'+formId,'f8',('dir'+formId))
                nc.variables['direction'+formId].units = 'degree'
                nc.variables['direction'+formId].long_name = \
                    'Degrees from true north in oceanographic convention'
                nc.variables['direction'+formId][:] = angles
                
                nc.createVariable('frequencyDir'+formId,'f8',('freqDir'+formId))
                nc.variables['frequencyDir'+formId].units = 'Hz'
                nc.variables['frequencyDir'+formId].long_name = 'Spectral frequency for dir_spec'
                nc.variables['frequencyDir'+formId][:] = freqDirSpec
    
            # Allocate date
            dir_time = np.zeros((alpha_1.shape[0]))              
            for bb in range(dir_time.shape[0]):
                tmpYear  = int(alpha_1[bb,0])
                tmpMonth = int(alpha_1[bb,1])
                tmpDay   = int(alpha_1[bb,2])
                tmpHour  = int(alpha_1[bb,3])
                if freqInd0 == 4:
                    tmpMin = int(0)
                else:
                    tmpMin = int(alpha_1[bb,4])
                                
                if tmpYear < 100:
                    tmpYear = tmpYear + 1900                                
                
                dir_time[bb] = (dt(tmpYear,tmpMonth,tmpDay,
                                                  tmpHour,tmpMin) - 
                                basetime).total_seconds()

                             
            # Read data
            alpha_1 = alpha_1[:,freqInd0:]
            alpha_2 = alpha_2[:,freqInd0:]
            r_1 = r_1[:,freqInd0:]
            r_2 = r_2[:,freqInd0:]
            
            # No Data Filter (NDBC uses 999.00 when there is no data)
            goodDataInd = np.logical_and(alpha_1[:,1] != 999.00,
                                         alpha_1[:,2] != 999.00)
            alpha_1  = alpha_1[goodDataInd,:]
            alpha_2  = alpha_2[goodDataInd,:]
            r_1      = r_1[goodDataInd,:]
            r_2      = r_2[goodDataInd,:]
            dir_time = dir_time[goodDataInd]
            
            
            # Find where dir_time and freq_time match and compute those values
            # only
            repInd    = np.isin(dir_time,freq_time)
            alpha_1   = alpha_1[repInd]
            alpha_2   = alpha_2[repInd]
            r_1       = r_1[repInd]
            r_2       = r_2[repInd]
            dir_time  = dir_time[repInd]
            
            repInd    = np.isin(freq_time,dir_time)
            freq_spec = freq_spec[repInd] 

            # Interpolate density spectrum into directional bins            
            if not np.array_equal(freq,freqDirSpec):
                freqSpecAll = np.copy(freq_spec)
                freq_spec   = np.zeros_like((r_1)) * np.nan
                for bb in range(freq_spec.shape[0]):
                    freq_spec[bb,:] = np.interp(freqDirSpec,freq,
                                                freqSpecAll[bb,:])
            
            # Construct 2D spectra
            # See http://www.ndbc.noaa.gov/measdes.shtml
            wspec = np.nan * np.zeros((alpha_1.shape[0],
                                       alpha_1.shape[1],angles.shape[0]))
                        
            # Time loop
            for bb in range(wspec.shape[0]):
                # Frequency loop  
                for cc in range(wspec.shape[1]):
                    # Direction loop
                    for dd in range(wspec.shape[2]):
                        wspec[bb,cc,dd] = (freq_spec[bb,cc] * np.pi/180.0 *
                                           (1.0/np.pi) * 
                                           (0.5 + r_1[bb,cc] * 
                                            np.cos((angles[dd]-alpha_1[bb,cc])*
                                                   np.pi/180.0) +
                                            r_2[bb,cc] * 
                                            np.cos(2 * np.pi / 180.0 * 
                                                   (angles[dd]-alpha_2[bb,cc])))
                                           )

            # Get the positive energy
            if fixNegative:
                tmpSpec = wspec.copy()
                tmpSpec[tmpSpec<0] = 0.0
                tmpFreqSpec = np.sum(tmpSpec,axis=-1)*np.abs(angles[2]-angles[1])
                posEnergy = np.trapz(np.abs(tmpFreqSpec),freq,axis=-1)
            
                # Get the total energy
                tmpFreqSpec = np.sum(np.abs(wspec),axis=-1)*np.abs(angles[2]-angles[1])
                totalEnergy = np.trapz(np.abs(tmpFreqSpec),freq,axis=-1)
             
                # Scale the spectrum
                wspec = np.array([wspec[ii,...] * totalEnergy[ii] / posEnergy[ii] 
                                 for ii in range(wspec.shape[0])])
                wspec[wspec<0] = 0.0

            # Write to file
            if cnt_dir == 1:
                nc.variables['dir_spec'+formId][:] = wspec
                nc.variables['dir_time'+formId][:] = dir_time
            else:
                nc.variables['dir_spec'+formId][tstep_dir:,:,:] = wspec
                nc.variables['dir_time'+formId][tstep_dir:] = dir_time
            
            tstep_dir += dir_time.shape[0]
                    
        # Update frequency time step and go to next year        
        tstep_freq += freq_time.shape[0]
        
       
    # Wrap up ------------------------------------------------------------------
    # Close NetCDF File     
    nc.close()

    return None

