'''
Methods to calculate the CEvNS rate in various detectors

ALL ENERGIES IN eV
ALL DISTANCES IN cm
'''


import reactor_tools

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
from scipy.interpolate import UnivariateSpline


keVPerGeV       = 1e6             # [keV / GeV]
hbarc 	        = 0.197*keVPerGeV # [keV fm]
fmPercm		= 1.0e13	# [fm / cm]
sin2thetaW      = 0.2387
cLight	        = 3.0e8           # [m / s]
nAvogadro       = 6.022e23
Mn              = 0.931 * keVPerGeV # keV
Mn_eV           = Mn*1e3          # eV
Gfermi	        = (1.16637e-5 / (keVPerGeV**2.))*(hbarc/fmPercm) # [cm / keV]
Gfermi_cm_eV    = Gfermi*1e-3
Me              = 0.511*1e6       # eV
fineStruct      = 1.0/137.036
joulePereV      = 1.602e-19	  # [J / eV]
electronCharge  = joulePereV      # C
s_per_day       = 60.0*60.0*24.0

#roi_max = 1000 # Region of interest is detector threshold to 1000 eV
roi_max = 1000000 # Don't use a reasonable max

def t_max(enu,M):
    if(enu>0):
        return min(enu/(1.0+M/(2*enu)),roi_max)
    #return enu/(1.0+M/(2*enu))

def e_min(T,M):
    return T/2.0 + 1/2.0 * np.sqrt(T**2+2.0*M*T)

#------------------------------------------------------------------------------
# CEvNS Cross Section
def dsigmadT_cns(T,enu,Z,N):
    # Shape is unitless and the factor gives units of cm^2/eV.
    M = Mn_eV*(N+Z) # eV
    if(T>t_max(enu,M)):
        return 0.
    Qweak = N-Z*(1-4*sin2thetaW)
    factor = Gfermi_cm_eV**2 / (4*np.pi) * Qweak**2 * M # cm^2/eV
    shape = 1. - (M * T) / (2.*enu**2.)
    return shape*factor
dsigmadT_cns = np.vectorize(dsigmadT_cns)

#------------------------------------------------------------------------------
# Rate in events per kg per day per keV
# Integral over reactor neutrino energies of
# reactor flux times the cross section
def dsigmadT_cns_rate(T, Z, N, nu_spec,
                      enu_min=None, enu_max=None):
    M = Mn_eV*(Z+N)
    targets_per_kg = nAvogadro/(Z+N)*1e3
    if enu_min is None:
        enu_min = e_min(T, M)
    else:
        enu_min = max(enu_min, e_min(T, M))
    if enu_max is None:
        enu_max = 1.e7
    res = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                     dsigmadT_cns(T, enu, Z, N),\
                     enu_min, enu_max)
    return res[0]*s_per_day*targets_per_kg
dsigmadT_cns_rate = np.vectorize(dsigmadT_cns_rate)

def dsigmadT_cns_rate_compound(T, Z_arr, N_arr, atom_arr, nu_spec,
                               enu_min=None, enu_max=None):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    xsec_tot = 0*np.array(T)
    for i in range(len(Z_arr)):
        xsec_tot += atom_arr[i]*A_arr[i]/mass_tot*dsigmadT_cns_rate(T, Z_arr[i], N_arr[i], nu_spec, enu_min, enu_max)
    return xsec_tot

#------------------------------------------------------------------------------
# Total CEvNS rate (integrated over recoil energies)

# CNS xsec analyticall integrated over T
def total_XSec_cns(Tmin,enu,Z,N):
    Qweak = N-(1.0-4.0*sin2thetaW)*Z
    M = Mn_eV*(Z+N)
    t_max_ = t_max(enu,M)
    if(Tmin>=t_max_):
        return 0.
    return Gfermi_cm_eV**2/4.0/np.pi * Qweak**2 *M*\
        (t_max_-Tmin-M/2.0/enu**2*(t_max_**2/2.0-Tmin**2/2.0))
total_XSec_cns = np.vectorize(total_XSec_cns)

def cevns_yield(Tmin,enu_max,Z,N,nu_spec,enu_min=0.):
    ''' Return yield in cm^2 per fission'''
    M = Mn_eV*(Z+N)
    e_min_curr = max(enu_min, e_min(Tmin,M))
    return spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                      total_XSec_cns(Tmin,enu,Z,N),\
                      e_min_curr,enu_max)[0]/\
                      nu_spec.nuFlux()
cevns_yield = np.vectorize(cevns_yield)

def cevns_yield_compound(Tmin, enu_max,
                         Z_arr, N_arr, atom_arr,
                         nu_spec, enu_min=0.):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    yield_tot = 0*np.array(Tmin)
    for i in range(len(Z_arr)):
        yield_tot += atom_arr[i]*A_arr[i]/mass_tot*cevns_yield(Tmin, enu_max, Z_arr[i], N_arr[i], nu_spec, enu_min)
    return yield_tot

def total_cns_rate_an(Tmin,enu_max,Z,N,nu_spec,enu_min=0.):
    M = Mn_eV*(Z+N)
    e_min_curr = max(enu_min, e_min(Tmin,M))
    targets_per_kg = nAvogadro/(Z+N)*1e3
    return spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                      total_XSec_cns(Tmin,enu,Z,N),\
                      e_min_curr,enu_max)[0]*\
                      s_per_day*targets_per_kg
total_cns_rate_an = np.vectorize(total_cns_rate_an)

def total_cns_rate_an_compound(Tmin, enu_max, Z_arr, N_arr, atom_arr, nu_spec, enu_min=0.):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    rate_tot = 0*np.array(Tmin)
    for i in range(len(Z_arr)):
        rate_tot += atom_arr[i]*A_arr[i]/mass_tot*total_cns_rate_an(Tmin, enu_max, Z_arr[i], N_arr[i], nu_spec, enu_min)
    return rate_tot

# Slow: Only to be used to validate total_cns_rate_an or to
# calculate the rate up to some Tmax<inf
def cns_total_rate_integrated(Tmin, Z, N, nu_spec, Tmax=roi_max):
    if(Tmin>=Tmax):
        return 0.
    x = np.linspace(Tmin, Tmax, 1000.)
    y = dsigmadT_cns_rate(x, Z, N, nu_spec)
    spl = UnivariateSpline(x, y)
    res = spl.integral(Tmin, Tmax)
    return res
    '''res = spint.quad(lambda T_: dsigmadT_cns_rate(T_, Z, N, nu_spec),
                     Tmin, Tmax)
    return res[0]'''
cns_total_rate_integrated = np.vectorize(cns_total_rate_integrated)

def cns_total_rate_integrated_compound(Tmin, Z_arr, N_arr, atom_arr, nu_spec, Tmax=roi_max):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    rate_tot = 0*np.array(Tmin)
    for i in range(len(Z_arr)):
        rate_tot += atom_arr[i]*A_arr[i]/mass_tot*cns_total_rate_integrated(Tmin, Z_arr[i], N_arr[i], nu_spec, Tmax)
    return rate_tot

#------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
