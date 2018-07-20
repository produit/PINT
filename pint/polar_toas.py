#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.extern import six
from pint.observatory import get_observatory
from astropy.time import Time


from astropy import log
from root_numpy import tree2array



def load_Polar_TOAs(eventname,weightcolumn="",minunix=0.0, maxunix=2500000000.0):
    '''
    TOAlist = load_Polar_TOAs(eventname)
      Read photon event times out of a POLAR event root file and return
      a list of PINT TOA objects.

      weightcolumn specifies ROOT Branch name to read the photon weights
      from. 

    '''
    import ROOT
    # Load photon times from POLAR event ROOT file
    f = ROOT.TFile.Open(eventname,"read")
    t = f.Get("t_trigger")

    # Read time Branch from ROOT file
    filter = '(tunix > '+str(minunix)+') && (tunix < '+str(maxunix)+')'
    if weightcolumn == "":
        array = tree2array(t,branches=['tunix'],selection=filter,start=0,stop=10000,step=1)
        ut=array['tunix']
    else:
        array = tree2array(t,branches=['tunix',weightcolumn],selection=filter,start=0,stop=10000,step=1)
        ut=array['tunix']
        weights=array[weightcolumn]
    if len(ut) == 0:
        log.error('No unix time read from file!')
        raise

    e=50*u.keV
    mjd1=Time(ut[0],format="unix")
    mjd2=Time(ut[-1],format="unix")
    log.info('Building spacecraft local TOAs, with MJDs in range {0} to {1}'.format(
            mjd1.tt.mjd,mjd2.tt.mjd))
    polarobs = get_observatory('Polar')

    try:
        if weightcolumn == "":
            toalist=[toa.TOA(Time(m,format="unix").tt.mjd, obs='Polar', scale='tt',energy=e)
                            for m in ut]
        else:
            toalist=[toa.TOA(Time(m,format="unix").tt.mjd, obs='Polar', scale='tt',energy=e,weight=w)
                            for m,w in zip(ut,weights)]
    except KeyError:
        log.error('Error processing POLAR TOAs. You may have forgotten to specify an ENG file file with --eng')
        raise

    return toalist
