"""Various tests to assess the performance of the J0623-0200."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
from pint.residuals import resids
import os

# Using Nanograv data J0623-0200
datadir = '../tests/datafile'
parfile = os.path.join(datadir, 'J0613-0200_NANOGrav_dfg+12_TAI_FB90.par')
timfile = os.path.join(datadir, 'J0613-0200_NANOGrav_dfg+12.tim')

# libstempo calculation
print("tempo2 calculation")
tempo2_vals = tempo2_utils.general2(parfile, timfile,['pre'])
# Build PINT model
print("PINT calculation")
m = mb.get_model(parfile)
# Get toas to pint
toas = toa.get_TOAs(timfile, planets=False, ephem='DE405', include_bipm=False)

t2_resids = tempo2_vals['pre']
presids_us = resids(toas, m).time_resids.to(u.us)
# Plot residuals
plt.errorbar(toas.get_mjds().value, presids_us.value,
            toas.get_errors().value, fmt='x')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.grid()
plt.show()
diff = (presids_us - t2_resids * u.second).to(u.us)
plt.plot(toas.get_mjds(high_precision=False), diff, '+')
plt.xlabel('Mjd (DAY)')
plt.ylabel('residule difference (us)')
plt.title('Residule difference between PINT and tempo2')
plt.show()
