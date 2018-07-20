# special_locations.py
from __future__ import division, print_function

# Special "site" location for Tiangong 2 satelite on top of wich POLAR is installed

from . import Observatory
from .special_locations import SpecialLocation
import astropy.units as u
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from ..utils import PosVel
from ..fits_utils import read_fits_event_mjds
from ..solar_system_ephemerides import objPosVel_wrt_SSB
import numpy as np
from astropy.time import Time
from astropy.table import Table
import astropy.io.fits as pyfits
from astropy.extern import six
from astropy import log
from scipy.interpolate import InterpolatedUnivariateSpline

from root_numpy import tree2array
import ROOT

def load_eng(eng_filename):
    '''Load data from a POLAR eng file
        in this ROOT file we have
        tunix: unix time
        x,y,z,vx,vy,vz: position and velocity in Earth rotating system
        ex,ey,ez: position in Earth non moving frame  
        Parameters
        ----------
        eng_filename : str
            Name of file to load
        Returns
        -------
        astropy Table containing Time, x, y, z, v_x, v_y, v_z data
    '''
    # Load photon times from ENG file
    f = ROOT.TFile.Open(eng_filename,"read")
    t = f.Get("ppd")
    log.info('Opened POLAR ENG file {0}'.format(eng_filename))

    # The X, Y, Z position are for the START time
    filter = 'tunix > 1.4e9'
    array = tree2array(t,branches=['tunix','x','y','z','vx','vy','vz'],selection=filter,start=0,stop=10000,step=1)
    log.info('Building ENG table covering MJDs {0} to {1}'.format(Time(array['tunix'][0],format="unix").tt.mjd,Time(array['tunix'][-1],format="unix").tt.mjd))
    return array

class PolarObs(SpecialLocation):
    """Observatory-derived class for the Polar eng data.
    Note that this must be instantiated once to be put into the Observatory registry.
    Parameters
    ----------
    name: str
        Observatory name
    engname: str
        File name to read spacecraft position information from
    tt2tdb_mode: str
        Selection for mode to use for TT to TDB conversion.
        'none' = Give no position to astropy.Time()
        'geo' = Give geocenter position to astropy.Time()
        'spacecraft' = Give spacecraft ITRF position to astropy.Time()
    """

    def __init__(self, name, engname, tt2tdb_mode = 'spacecraft'):
        self.ENG = load_eng(engname)
        # Now build the interpolator here:
        self.X = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['x'])
        self.Y = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['y'])
        self.Z = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['z'])
        self.Vx = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['vx'])
        self.Vy = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['vy'])
        self.Vz = InterpolatedUnivariateSpline(self.ENG['tunix'],self.ENG['vz'])
        self.tt2tdb_mode = tt2tdb_mode
        super(PolarObs, self).__init__(name=name)

    @property
    def timescale(self):
        return 'utc'

    def earth_location_itrf(self, time=None):
        '''Return POLAR spacecraft location in ITRF coordinates'''

        if self.tt2tdb_mode.lower().startswith('none'):
            log.warning('Using location=None for TT to TDB conversion')
            return None
        elif self.tt2tdb_mode.lower().startswith('geo'):
            log.warning('Using location geocenter for TT to TDB conversion')
            return EarthLocation.from_geocentric(0.0*u.m,0.0*u.m,0.0*u.m)
        elif self.tt2tdb_mode.lower().startswith('spacecraft'):
            # First, interpolate Earth-Centered Inertial (ECI) geocentric
            # location from orbit file.
            # These are inertial coordinates aligned with ICRS, called GCRS
            # <http://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>
            pos_gcrs =  GCRS(CartesianRepresentation(self.X(time.utc.unix)*u.m,
                                                     self.Y(time.utc.unix)*u.m,
                                                     self.Z(time.utc.unix)*u.m),
                             obstime=time)

            # Now transform ECI (GCRS) to ECEF (ITRS)
            # By default, this uses the WGS84 ellipsoid
            pos_ITRS = pos_gcrs.transform_to(ITRS(obstime=time))

            # Return geocentric ITRS coordinates as an EarthLocation object
            return pos_ITRS.earth_location
        else:
            log.error('Unknown tt2tdb_mode %s, using None', self.tt2tdb_mode)
            return None

    @property
    def tempo_code(self):
        return None

    def posvel(self, t, ephem):
        '''Return position and velocity vectors of TG2, wrt SSB.
        These positions and velocites are in inertial coordinates
        (i.e. aligned with ICRS)
        t is an astropy.Time or array of astropy.Times
        '''
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB('earth', t, ephem)
        # Now add vector from Earth to TG2
        tg2_pos_geo = np.array([self.X(t.utc.unix), self.Y(t.utc.unix), self.Z(t.utc.unix)])*u.m
        log.debug("tg2_pos_geo {0}".format(tg2_pos_geo[:,0]))
        tg2_vel_geo = np.array([self.Vx(t.utc.unix), self.Vy(t.utc.unix), self.Vz(t.utc.unix)])*u.m/u.s
        tg2_posvel = PosVel( tg2_pos_geo, tg2_vel_geo, origin='earth', obj='POLAR')
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + tg2_posvel
