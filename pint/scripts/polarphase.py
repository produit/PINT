#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
import matplotlib.pyplot as plt
from pint.polar_toas import load_Polar_TOAs
from pint.plot_utils import phaseogram
from pint.observatory.polar_obs import PolarObs
import argparse
import astropy.io.fits as pyfits
from astropy.time import Time
from pint.eventstats import hmw, hm, h2sig
from astropy.coordinates import SkyCoord

from astropy import log

#log.setLevel('DEBUG')

def main(argv=None):

    parser = argparse.ArgumentParser(description="Use PINT to compute H-test and plot Phaseogram from a POLAR event file.")
    parser.add_argument("eventfile",help="Polar event root file name.")
    parser.add_argument("parfile",help="par file to construct model from")
    parser.add_argument("--wei",help="Branch name for event weights",default="")
    parser.add_argument("--eng",help="Path to ENG file.",default="$POLAR_AUX/Alleng.root")
    parser.add_argument("--addphase",help="Add phase tree Friend",
        default=False,action='store_true')
    parser.add_argument("--plot",help="Show phaseogram plot.", action='store_true', default=False)
    parser.add_argument("--plotfile",help="Output figure file name (default=None)", default=None)
    parser.add_argument("--maxMJD",help="Maximum MJD to include in analysis", default=None)
    parser.add_argument("--planets",help="Use planetary Shapiro delay in calculations (default=False)", default=False, action="store_true")
    parser.add_argument("--ephem",help="Planetary ephemeris to use (default=DE421)", default="DE421")
    args = parser.parse_args(argv)


    # Read in model
    modelin = pint.models.get_model(args.parfile)
    if 'ELONG' in modelin.params:
        tc = SkyCoord(modelin.ELONG.quantity,modelin.ELAT.quantity,
            frame='barycentrictrueecliptic')
    else:
        tc = SkyCoord(modelin.RAJ.quantity,modelin.DECJ.quantity,frame='icrs')

    if args.eng is not None:
        # Instantiate PolarObs once so it gets added to the observatory registry
        PolarObs(name='Polar',engname=args.eng)

    # Read event file and return list of TOA objects
    tl  = load_Polar_TOAs(args.eventfile, weightcolumn=args.wei)

    # Discard events outside of MJD range
    if args.maxMJD is not None:
        tlnew = []
        print("pre len : ",len(tl))
        maxT = Time(float(args.maxMJD),format='mjd')
        print("maxT : ",maxT)
        for tt in tl:
            if tt.mjd < maxT:
                tlnew.append(tt)
        tl=tlnew
        print("post len : ",len(tlnew))

    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    ts.filename = args.eventfile
    ts.compute_TDBs()
    ts.compute_posvels(ephem=args.ephem,planets=args.planets)

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(),mjds.max())

    # Compute model phase for each TOA
    phss = modelin.phase(ts.table)[1]
    # ensure all postive
    phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)
    mjds = ts.get_mjds()
    weights = np.array([1]*len(mjds))
    h = float(hmw(phases,weights))
    print("Htest : {0:.2f} ({1:.2f} sigma)".format(h,h2sig(h)))
    if args.plot:
        log.info("Making phaseogram plot with {0} photons".format(len(mjds)))
        phaseogram(mjds,phases,weights,bins=100,plotfile = args.plotfile)

    if args.addphase:
        if len(event_dat) != len(phases):
            raise RuntimeError('Mismatch between length of ROOT tree ({0}) and length of phase array ({1})!'.format(len(event_dat),len(phases)))

    return 0
