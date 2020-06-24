'''
Tools for calculating the neutrino rate from a nuclear reactor.

Based on code by:
Adam Anderson
14 April 2016
adama@fnal.gov

Modified by:
Joseph Johnston
12 July 2019
jpj13@mit.edu

Note: Convention on units:
  --all masses are in kg
  --all energies are in keV
'''

import numpy as np
from scipy.integrate import quad
import ROOT

class NeutrinoSpectrum:
    '''
    Class to calculate the neutrino spectrum from a nuclear reactor

    Args:
        distance: Distance from reactor in cm^2
        power: Reactor power in Mw
        fix_2mev: If true, then the spectrum below 2 MeV will be set
            to the value at 2 MeV. (For use for spectral fits that
            were not meant to apply at low energies, eg Mueller)
        frac_u235, frac_u238, frac_pu239, frac_pu241: Fuel fractions
            to be used when calculating the reactor flux
            (Default: u235=1., as in a research reactor)
        include_other: Whether "other" neutrinos should be included
            when calculating the flux
        bump_frac: Fraction of the spectrum to replace with the reactor
            bump. bump_reset will be set to true, and should be reset to
            True whenever the other spectrum parameters are changed,
            because it will force recalculation of the spectrum
            normalization before adding the bump.
    '''

    def __init__(self, distance, power, fix_2mev=False,
                 frac_u235=1., frac_u238=0., frac_pu239=0., frac_pu241=0.,
                 include_other=True, bump_frac=0.,
                 mueller_partial=False):
        self.frac_u235 = frac_u235
        self.frac_u238 = frac_u238
        self.frac_pu239 = frac_pu239
        self.frac_pu241 = frac_pu241
        self.include_other = include_other
        self.distance = distance
        self.power = power
        self.fix_2mev = fix_2mev
        self._flux_use_functions = [True, True, True, True, True]
        def flux_other_default(enu):
            if(type(enu)==float):
                enu = np.asarray([enu])
            else:
                enu = np.asarray(enu)
            return 0.0*enu
        self._flux_functions = [None, None, None, None, flux_other_default]
        self._flux_spline_graphs = [None, None, None, None, None]
        self._flux_spline_evals = [None, None, None, None, None]
        self._mueller_partial = mueller_partial
        self.bump_frac = 0.
        self.bump_reset = True
        self._spectrum_integral = 0.
        

    def get_fractions(self):
        '''return fractions as an array 

        Returns:
            [frac_u235, frac_u238, frac_pu239, frac_pu241]
        '''
        return [self.frac_u235, self.frac_u238, self.frac_pu239, self.frac_pu241]

    def set_fractions(self, fractions):
        '''Set fractions with an array
        
        Args:
            arr: [frac_u235, frac_u238, frac_pu239, frac_pu241]
        '''
        self.frac_u235 = fractions[0]
        self.frac_u238 = fractions[1]
        self.frac_pu239 = fractions[2]
        self.frac_pu241 = fractions[3]

    def _d_r_d_enu_idx(self, enu, idx):
        '''
        Reator anti neutrino spectrum from isotope idx

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        if(type(enu)==float):
            enu = np.asarray([enu])
        else:
            enu = np.asarray(enu)
        spec = 0.0*enu
        if self._flux_use_functions[idx]:
            spec = self._flux_functions[idx](enu)
            if(self.fix_2mev):
                spec[enu<2.e3] = self._flux_functions[idx](2.e3)
        else:
            spec = self._flux_spline_evals[idx](enu)
            if(self.fix_2mev):
                spec[enu<2.e3] = self._flux_spline_evals[idx](2.e3)
            if(self._mueller_partial):
                spec[enu>1.8e3] = self._flux_functions[idx](enu[enu>1.8e3])
        spec[spec<0] = 0
        return spec

    def d_r_d_enu_u235(self, enu):
        '''
        Reator anti neutrino spectrum from U-235

        Must be initialized via initialize_d_r_d_enu

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        return self._d_r_d_enu_idx(enu, 0)

    def d_r_d_enu_u238(self, enu):
        '''
        Reator anti neutrino spectrum from U-238

        Must be initialized via initialize_d_r_d_enu

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        return self._d_r_d_enu_idx(enu, 1)

    def d_r_d_enu_pu239(self, enu):
        '''
        Reator anti neutrino spectrum from Pu-239

        Must be initialized via initialize_d_r_d_enu

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        return self._d_r_d_enu_idx(enu, 2)
    
    def d_r_d_enu_pu241(self, enu):
        '''
        Reator anti neutrino spectrum from Pu-241

        Must be initialized via initialize_d_r_d_enu

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        return self._d_r_d_enu_idx(enu, 3)

    def d_r_d_enu_other(self, enu):
        '''
        Reator anti neutrino spectrum from non-fission isotopes

        Initialized to return 0.

        Args:
            enu: Neutrino energy in keV (array)

        Returns:
            Spectrum in [nu/ keV/ fission] (array)
        '''
        return self._d_r_d_enu_idx(enu, 4)

    def d_r_d_enu_bump(self, enu):
        if(self.bump_reset):
            # Calculate normalization
            frac = self.bump_frac
            self.bump_frac = 0.
            self._spectrum_integral = quad(self.d_r_d_enu, 0., np.inf)[0]
            self.bump_frac = frac
            self.bump_reset = False
        mu = 4200.
        sig = 500.
        return self.bump_frac*self._spectrum_integral*\
            1./(sig*np.sqrt(2.*np.pi))*\
            np.exp(-0.5*((enu-mu)/sig)**2)

    def d_r_d_enu(self, enu):
        tot = self.frac_u235*self.d_r_d_enu_u235(enu) +\
              self.frac_u238*self.d_r_d_enu_u238(enu) +\
              self.frac_pu239*self.d_r_d_enu_pu239(enu) +\
              self.frac_pu241*self.d_r_d_enu_pu241(enu)
        if(self.include_other):
            tot += self.d_r_d_enu_other(enu)
        if(self.bump_frac>0. and self.bump_frac<=1.0):
            tot *= (1-self.bump_frac)
            tot += self.d_r_d_enu_bump(enu)
        return tot

    def fis_per_s(self):
        '''
        returns
        -------
        fis_per_s: fissions per second assuming 200 MeV
            per fission
        '''
        return self.power/200.0/1.602176565e-19

    def nuFlux(self):
        '''
        Computes the total flux per fission of reactor antineutrinos
        at a given distance from the core, assuming a point-like
        flux, and nominal neutrino production
        
        Returns
        -------
        flux : float
            The reactor neutrino flux in fissions/s/cm^2 
        '''
        flux = self.fis_per_s()/ (4*np.pi * self.distance**2.)
        return flux


    def d_phi_d_enu(self, enu):
        '''
        Reactor neutrino spectrum in neutrinos/(keV*s*cm^2)

        Args:
            enu: Neutrino energy in keV

        Returns:
            Reactor neutrino spectrum
        '''
        return self.nuFlux() * self.d_r_d_enu(enu)

    def d_phi_d_enu_ev(self, enu_ev):
        '''
        Reactor neutrino spectrum in neutrinos/(eV*s*cm^2)

        Args:
            enu_ev: Neutrino energy in eV

        Returns:
            Reactor neutrino spectrum
        '''
        enu_kev = enu_ev*1e-3
        return self.d_phi_d_enu(enu_kev) * 1e-3


    def initialize_d_r_d_enu(self, isotope, mode="mueller",
                             filename=None, th1_name=None,
                             scale=1.0):
        '''
        Initialize a reactor antineutrino spectrum

        Args:
            isotope: String specifying isotopes ("u235", "u238",
                "pu239", "pu241", or "other")
            mode: String specifying how the methods should be
                initialized. Options:
                  - "mueller" - Use the fit functions from 
                      arXiv:1101.2663v3. Set the spectrum below
                      2 MeV to the value at 2 MeV
                  - "zero" - Return 0 for all energies
                  - "txt" - Read in a spectrum from a text file,
                      with energy (MeV) in the first column and
                      neutrinos/MeV/fission in the second
                  - "root" - Read in a spectrum from a TH1 object
                      in a root file. A th1_name must be given, 
                      the x axis must be MeV, and it must be
                      normalized to the number of neutrinos/fission
        '''
        isotope_map = {"u235":0, "u238":1, "pu239":2, "pu241":3, "other":4}
        if not isotope in isotope_map.keys():
            print("Invalid isotope selected in initialize_d_r_d_enu")
            print('\tPlease select "u235", "u238", "pu239", "pu241", or "other"')
            return

        if(mode=="txt" or mode=="root"):
            # Create splines
            self._flux_use_functions[isotope_map[isotope]] = False
            enu, spec = list(), list()
            if(mode=="txt"):
                enu, spec = np.loadtxt(filename, usecols=(0,1), unpack=True)
            elif(mode=="root"):
                rootfile = ROOT.TFile(filename)
                th1 = rootfile.Get(th1_name)
                nxbins = th1.GetNbinsX()
                xaxis = th1.GetXaxis()
                for ibin in range(1, nxbins+1):
                    enu.append(xaxis.GetBinCenter(ibin))
                    spec.append(th1.GetBinContent(ibin))
            self._flux_spline_graphs[isotope_map[isotope]] = \
                ROOT.TGraph(len(enu), np.ascontiguousarray(enu),
                            scale*np.ascontiguousarray(spec))
            def spl_eval(enu):
                # Graph has energies in MeV, we want keV
                return 1e-3*self._flux_spline_graphs[isotope_map[isotope]].Eval(enu*1e-3)
            spl_eval = np.vectorize(spl_eval)
            self._flux_spline_evals[isotope_map[isotope]] = spl_eval
        elif(mode=="zero"):
            self._flux_use_functions[isotope_map[isotope]] = True
            def flux_zero(enu):
                if(type(enu)==float):
                    enu = np.asarray([enu])
                else:
                    enu = np.asarray(enu)
                return 0.0*np.array(enu)
            self._flux_functions[isotope_map[isotope]] = flux_zero
        if(mode=="mueller" or self._mueller_partial):
            if(isotope=="u235"):
                def flux_u235(enu):
                    if(type(enu)==float):
                        enu = np.asarray([enu])
                    else:
                        enu = np.asarray(enu)
                    enu_mev = enu / 1.e3
                    return scale * 1e-3 * np.exp(3.217 - 3.111*enu_mev + 1.395*(enu_mev**2.0) - \
    	                                 (3.690e-1)*(enu_mev**3.0) + (4.445e-2)*(enu_mev**4.0) - (2.053e-3)*(enu_mev**5.0))
                self._flux_functions[0] = flux_u235
            elif(isotope=="u238"):
                def flux_u238(enu):
                    if(type(enu)==float):
                        enu = np.asarray([enu])
                    else:
                        enu = np.asarray(enu)
                    enu_mev = enu / 1.e3
                    return scale * 1e-3 * np.exp((4.833e-1) + (1.927e-1)*enu_mev - (1.283e-1)*enu_mev**2.0 - \
    					 (6.762e-3)*enu_mev**3.0 + (2.233e-3)*enu_mev**4.0 - (1.536e-4)*enu_mev**5.0)
                self._flux_functions[1] = flux_u238
            elif(isotope=="pu239"):
                def flux_pu239(enu):
                    if(type(enu)==float):
                        enu = np.asarray([enu])
                    else:
                        enu = np.asarray(enu)
                    enu_mev = enu / 1.e3
                    return scale * 1e-3 * np.exp(6.413 - 7.432*enu_mev + 3.535*enu_mev**2.0 - \
    					 (8.82e-1)*enu_mev**3.0 + (1.025e-1)*enu_mev**4.0 - (4.550e-3)*enu_mev**5.0)
                self._flux_functions[2] = flux_pu239
            elif(isotope=="pu241"):
                def flux_pu241(enu):
                    if(type(enu)==float):
                        enu = np.asarray([enu])
                    else:
                        enu = np.asarray(enu)
                    enu_mev = enu / 1.e3
                    return scale * 1e-3 * np.exp(3.251 - 3.204*enu_mev + 1.428*enu_mev**2.0 - \
    					 (3.675e-1)*enu_mev**3.0 + (4.254e-2)*enu_mev**4.0 - (1.896e-3)*enu_mev**5.0)
                self._flux_functions[3] = flux_pu241
            elif(isotope=="other"):
                def flux_other(enu):
                    if(type(enu)==float):
                        enu = np.asarray([enu])
                    else:
                        enu = np.asarray(enu)
                    return 0.0*np.array(enu)
                self._flux_functions[4] = flux_other
        #else:
            #print("Invalid mode selected, not initializing")
            #print("\tisotope: %s"%isotope)
            #print("\tmode: %s"%mode)
            #print("\tfilename: %s"%filename)
            #print("\tth1name: %s"%th1_name)
