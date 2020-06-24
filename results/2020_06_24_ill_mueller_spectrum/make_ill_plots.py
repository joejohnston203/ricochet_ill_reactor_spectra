from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, total_XSec_cns

from scipy.integrate import quad

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

Mn = cevns_spectra.Mn
Mn_eV = Mn*1e3

s_per_day = 60.0*60.0*24.0


def plot_neutrino_spectrum(nu_spec, nu_spec_2,
                           label1="Spec 1", label2="Spec 2"):
    '''
    Make a plot of the neutrino spectrum

    Args:
        nu_spec: Initialized NeutrinoSpectrum object
        nu_spec_2: Second Initialized NeutrinoSpectrum
            object to compare to
    '''
    initial_fracs = nu_spec.get_fractions()

    # Plot neutrino flux at 0 days and 50 days
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    e_arr = np.linspace(0., 1e7, 1000)

    nu_spec.set_fractions([1.0, 0.0, 0.0, 0.0])
    spec_0d = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    nu_spec.set_fractions([0.98, 0.003, 0.014, 0.003])
    spec_50d = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.axvline(1.8, linestyle=":")
    plt.plot(e_arr*1e-6,spec_0d*1e6,'r-',label='0 days',linewidth=2)
    plt.plot(e_arr*1e-6,spec_50d*1e6,'b--',label='50 days',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.ylim(0., 3.e16)
    plt.title('Neutrino Flux at ILL Reactor')
    plt.savefig('plots/neutrino_spectrum.png')
    fig0.clf()

    # Save spectra to file:
    np.savetxt("results/ill_reactor_nu_spectra.txt", np.column_stack((1e-3*e_arr, 1e3*spec_0d, 1e3*spec_50d)), header="# E_nu (keV), ILL Flux @ 0 days (nu/(keV*day*cm^2), ILL Flux @ 50 days (nu/(keV*day*cm^2)")

    plt.plot(e_arr*1e-6,(spec_50d-spec_0d)/spec_0d,'b-',linewidth=2)
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Fractional Difference (50 days- 0 days)/(0 days)')
    plt.ylim(-0.01, 0.05)
    plt.title('Time Evolution of Spectrum at ILL Reactor')
    plt.savefig('plots/neutrino_spectrum_difference.png')
    fig0.clf()

    nu_spec.set_fractions(initial_fracs)

    # Plot first vs second flux flux
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    e_arr = np.linspace(0., 1e7, 1000)
    spec_new = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_new*1e6,'r-',linewidth=2, label=label1)
    prev_fix_2mev = nu_spec_2.fix_2mev
    nu_spec_2.fix_2mev = False
    spec_2 = s_per_day*nu_spec_2.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_2*1e6,'k--',linewidth=2, label=label2)
    nu_spec_2.fix_2mev = prev_fix_2mev
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.axvline(1.8)
    plt.title('Neutrino Flux at ILL Reactor (0 days)')
    plt.savefig('plots/neutrino_spectrum_1_vs_2.png')
    fig0.clf()

    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    e_arr = np.linspace(0., 1e7, 1000)
    plt.plot(e_arr*1e-6,spec_2/spec_new,'k',linewidth=2)
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title(label1+'/ '+label2+' Ratio')
    plt.ylim([0.7, 1.3])
    plt.savefig('plots/neutrino_spectrum_ratio.png')
    fig0.clf()

    # Plot second flux
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    e_arr = np.linspace(0., 1e7, 1000)
    prev_fix_2mev = nu_spec_2.fix_2mev
    nu_spec_2.fix_2mev = False
    plt.plot(e_arr*1e-6,s_per_day*nu_spec_2.d_phi_d_enu_ev(e_arr)*1e6,'k--',linewidth=2, label=label2)
    nu_spec_2.fix_2mev = prev_fix_2mev
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.axvline(1.8)
    plt.title(label2+'Neutrino Flux at ILL Reactor (0 days)')
    plt.savefig('plots/neutrino_spectrum_2.png')
    fig0.clf()
    # Save mueller spectra to file:
    np.savetxt("results/ill_reactor_nu_spectra_2.txt", np.column_stack((1e-3*e_arr, spec_2*1.e3)), header="# E_nu (keV), ILL Flux @ 0 days (nu/(keV*day*cm^2) ("+label2+")")

def plot_cevns_rate_fixed_T():
    fig2 = plt.figure()
    Tmin = 0.001
    Z = 32
    N = 72.64-32.
    e_arr = np.linspace(0.,1e7, 10000)
    fig2.patch.set_facecolor('white')
    plt.ylim((1e-44, 1e-42))
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(10.,e_arr,Z,N),'k:',label='T=10 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(50.,e_arr,Z,N),'b-',label='T=50 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(100.,e_arr,Z,N),'r--',label='T=100 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(200.,e_arr,Z,N),'g-.',label='T=200 eV',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Differential XSec, cm^2/eV')
    plt.title('Ge Differential CEvNS XSec, Fixed T')
    plt.savefig('plots/diff_xsec_fixed_T.png')
    fig2.clf()


def plot_cevns_rate_fixed_Enu():
    fig2 = plt.figure()
    Tmin = 0.001
    Z = 32
    N = 72.64-32.
    t_arr = np.logspace(0, 4, 10000)
    fig2.patch.set_facecolor('white')
    plt.ylim((1e-44, 1e-42))
    plt.loglog(t_arr,dsigmadT_cns(t_arr,1e6,Z,N),'k:',label='Enu = 1 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,2e6,Z,N),'b-',label='Enu = 2 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,4e6,Z,N),'r--',label='Enu = 4 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,6e6,Z,N),'g-.',label='Enu = 6 MeV',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Energy (eV)')
    plt.ylabel('Differential XSec, cm^2/eV')
    plt.title('Ge Differential CEvNS XSec, Fixed Enu')
    plt.savefig('plots/diff_xsec_fixed_Enu.png')
    fig2.clf()

def plot_cevns_rate_vs_T_Enu(nu_spec):
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    e_arr = np.linspace(0.,1e7, 1000)
    Tmin = 0.001
    t_arr = np.logspace(0, 3, 1000)

    Z = 32
    N = 72.64-32.

    T, E = np.meshgrid(t_arr,e_arr)
    spec = dsigmadT_cns(T,E,Z,N)
    smax = spec.max()
    smin = smax*1e-3
    spec[spec<smin] = smin

    im = plt.pcolor(T, E*1e-6, spec,
                    norm=LogNorm(vmin=smin, vmax=smax),
                    cmap='PuBu_r')
    fig.colorbar(im)
    plt.xlabel("Recoil Energy T (eV)")
    plt.ylabel("Neutrino Energy Enu (MeV)")
    plt.title("Ge Differential XSec, cm^2/eV")
    plt.savefig('plots/diff_xsec_vs_E_T.png')

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    e_arr = np.linspace(0.,1e7, 1000)
    Tmin = 0.001
    t_arr = np.logspace(0, 3, 1000)

    Z = 32
    N = 72.64-32.

    T, E = np.meshgrid(t_arr,e_arr)
    spec_flux = dsigmadT_cns(T,E,Z,N)*nu_spec.d_phi_d_enu_ev(E)*s_per_day*1e6
    smax = spec_flux.max()
    smin = smax*1e-3
    spec_flux[spec_flux<smin] = smin

    im = plt.pcolor(T, E*1e-6, spec_flux,
                    norm=LogNorm(vmin=smin, vmax=smax),
                    cmap=plt.get_cmap('PuBu_r'))
    fig.colorbar(im)
    plt.xlabel("Recoil Energy T (eV)")
    plt.ylabel("Neutrino Energy Enu (MeV)")
    plt.title("Ge Differential XSec * Reactor Flux, nu/(eV*MeV*day)")
    plt.savefig('plots/diff_xsec_flux_vs_E_T.png')


def plot_dsigmadT_cns_rate(nu_spec, site_str,
                           plot_output_name, text_output_name,
                           bounds=[1e-5, 1e-1],
                           use_kev=False):
    if(use_kev):
        factor = 1.e3
    else:
        factor = 1.

    t_arr = np.logspace(0, 3, num=1000)
    
    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.ylim(bounds)
    plt.xlim(1e0/factor, 1e3/factor)
    #plt.loglog(t_arr,dsigmadT_cns_rate(t_arr/factor, 14, 28.08-14, nu_spec)*factor,'g-',label='Si (A=28.1)',linewidth=2)
    zn_rates = dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec)*factor
    plt.loglog(t_arr/factor,zn_rates,'b-',label='Zn (A=64.4)',linewidth=2)
    ge_rates = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)*factor
    plt.loglog(t_arr/factor,ge_rates,'r-',label='Ge (A=72.6)',linewidth=2)
    #plt.loglog(t_arr/factor,dsigmadT_cns_rate_compound(t_arr, [13, 8], [26.982-13., 16.0-8.], [2, 3], nu_spec)*factor,'c-.',label='Al2O3 (A~20)',linewidth=2)
    #plt.loglog(t_arr/factor,dsigmadT_cns_rate_compound(t_arr, [20, 74, 8], [40.078-20., 183.84-74., 16.0-8.], [1, 1, 4], nu_spec)*factor,'m:',label='CaWO4 (A~48)',linewidth=2)
    plt.legend(prop={'size':11})
    if(use_kev):
        plt.xlabel('Recoil Energy T (keV)')
        plt.ylabel('Differential Event Rate (Events/kg/day/keV)')
    else:
        plt.xlabel('Recoil Energy T (eV)')
        plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
    plt.title("%s Differential Rate"%site_str)
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig(plot_output_name)
    fig3.clf()

    np.savetxt(text_output_name, np.column_stack((t_arr, zn_rates, ge_rates)), header="# Recoil Energy (eV), Zn Diff Rate (events/kg/day/eV), Ge Diff Rate (events/kg/day/eV)")


def plot_dsigmadT_cns_systematic(nu_spec, nu_spec_2,
                                 site_str, plot_output_name,
                                 text_output_name,
                                 bounds=[1e-5, 1e-1],
                                 use_kev=False):
    if(use_kev):
        factor = 1.e3
    else:
        factor = 1.

    t_arr = np.logspace(0, 3, num=1000)

    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.ylim(bounds)
    plt.xlim(1e0/factor, 1e3/factor)
    zn_rates = dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec)*factor
    plt.loglog(t_arr/factor,zn_rates,'b-',label='Zn, New',linewidth=2)
    zn_rates_2 = dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec_2)*factor
    plt.loglog(t_arr/factor,zn_rates_2,'k:',label='Zn, Old',linewidth=2)
    ge_rates = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)*factor
    plt.loglog(t_arr/factor,ge_rates,'r-',label='Ge, New',linewidth=2)
    ge_rates_2 = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec_2)*factor
    plt.loglog(t_arr/factor,ge_rates_2,'k-.',label='Ge, Old',linewidth=2)
    plt.legend(prop={'size':11})
    if(use_kev):
        plt.xlabel('Recoil Energy T (keV)')
        plt.ylabel('Differential Event Rate (Events/kg/day/keV)')
    else:
        plt.xlabel('Recoil Energy T (eV)')
        plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
    plt.title("%s Differential Rate, New vs Old Nu Flux"%site_str)
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig(plot_output_name+".png")
    fig3.clf()

    fig4 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.semilogx(t_arr, zn_rates/zn_rates_2, 'b-', label='Zn', linewidth=2)
    plt.semilogx(t_arr, ge_rates/ge_rates_2, 'r-', label='Ge', linewidth=2)
    if(use_kev):
        plt.xlabel('Recoil Energy T (keV)')
    else:
        plt.xlabel('Recoil Energy T (eV)')
    plt.ylabel('Ratio of New/Old Differential Rates')
    plt.title("%s Ratio of Differential Rates, New vs Old Nu Flux"%site_str)
    plt.legend(prop={'size':11})
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig(plot_output_name+"_ratio.png")
    fig4.clf()

def plot_total_cns_rate(nu_spec):
    # Make a plot of integrated event rate per eV vs threshold energy
    t_arr = np.logspace(0, 3, num=100)
    
    fig4 = plt.figure()
    fig4.patch.set_facecolor('white')
    #plt.loglog(t_arr,total_cns)rate_an(t_arr, 1e7, 14, 28.08-14, nu_spec),'g-',label='Si (A=28.1)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an(t_arr, 1e7, 30, 35.38, nu_spec),'b-',label='Zn (A=64.4)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an(t_arr, 1e7, 32, 72.64-32., nu_spec),'r-',label='Ge (A=72.6)',linewidth=2)
    #plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, [13, 8], [26.982-13., 16.0-8.], [2, 3], nu_spec),'c-.',label='Al2O3 (A~20)',linewidth=2)
    #plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, [20, 74, 8], [40.078-20., 183.84-74., 16.0-8.], [1, 1, 4], nu_spec),'m:',label='CaWO4 (A~48)',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Threshold (eV)')
    plt.ylabel('Event Rate (Events/kg/day)')
    plt.title("ILL Total Rate vs Threshold")
    plt.axvline(x=10., color='black')
    plt.axvline(x=50., color='black')
    plt.axvline(x=100., color='black')
    plt.savefig('plots/ill_total_event_rate.png')

    np.savetxt("results/ill_total_event_rate.txt",
               np.column_stack((t_arr,
                                total_cns_rate_an(t_arr, 1e7, 30, 35.38, nu_spec),
                                total_cns_rate_an(t_arr, 1e7, 32, 72.64-32., nu_spec))),
               header="Recoil Energy (eV), Zn total rate (evts/kg/day), Ge total rate (evts/kg/day)")

def plot_total_cns_rate_threshold_dist(nu_spec, make_plots=True):
    distance_init = nu_spec.distance


    for settings in [["Zn", 30, 35.38],
                     ["Ge", 23, 72.64-32]]:
        # Print the rate for a few interesting cases
        print("Rates for %s"%settings[0])
        for threshold in [10., 50., 100., 200.]:
            for dist_m in [7., 8., 9., 10.]:
                dist = dist_m*100.
                nu_spec.distance = dist
                rate  = total_cns_rate_an(threshold, 1e7,
                                          settings[1], settings[2], nu_spec)
                print("\tRate with %.0f eV threshold at %.0f m: %.2f evts/kg/day"%
                      (threshold, dist/100., rate))
        if make_plots:
            fig4 = plt.figure()
            fig4.patch.set_facecolor('white')
            # Make a plot of integrated event rate per eV vs threshold energy
            t_arr = np.logspace(0, np.log10(200.), num=100)
            r_arr = np.linspace(6., 11., num=100)
            T, R = np.meshgrid(t_arr, r_arr)
            def temp_tot_rate(t, r):
                nu_spec.distance = r*100.
                return total_cns_rate_an(t, 1e7, settings[1],
                                         settings[2],  nu_spec)
            temp_tot_rate = np.vectorize(temp_tot_rate)
            im = plt.pcolor(T, R, temp_tot_rate(T, R),
                            cmap='cool')
                            #cmap='PuBu_r')
            fig4.colorbar(im)
            plt.xscale('log')
            plt.xlim(1., 200.)
            plt.xlabel('Recoil Threshold (eV)')
            plt.ylabel('Distance from Reactor (m)')
            plt.title("ILL Total Rate in %s (Events/kg/day)"%settings[0])
            plt.savefig('plots/ill_total_event_rate_distance_%s.png'%settings[0])

    nu_spec.distance = distance_init

def plot_low_vs_high(nu_spec, enu_low=1.8e6,
                     output_path_prefix="plots/lowe_1_8_diff_",
                     differential=True,
                     frac_interest=0.5):
    t_arr = np.logspace(0, 3, num=100)

    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    if(differential):
        plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec, enu_max=enu_low),'b-',label='Zn (A=64.4), enu<%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec, enu_min=enu_low),'b--',label='Zn (A=64.4), enu>%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec, enu_max=enu_low),'r-',label='Ge (A=72.6), enu<%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec, enu_min=enu_low),'r--',label='Ge (A=72.6), enu>%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.legend(prop={'size':11})
        plt.xlabel('Recoil Energy (eV)')
        plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
        plt.title("ILL Differential Rate")
        plt.ylim(1e-4, 1e1)
    else:
        plt.loglog(t_arr,total_cns_rate_an(t_arr, enu_low, 30, 35.38, nu_spec),'b-',label='Zn(A=64.4), enu<%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,total_cns_rate_an(t_arr, 1.e7, 30, 35.38, nu_spec, enu_min=enu_low),'b--',label='Zn(A=64.4)), enu>%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,total_cns_rate_an(t_arr, enu_low, 32, 72.64-32., nu_spec),'r-',label='Ge (A=72.6), enu<%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.loglog(t_arr,total_cns_rate_an(t_arr, 1.e7, 32, 72.64-32., nu_spec, enu_min=enu_low),'r--',label='Ge (A=72.6), enu>%.1f MeV'%(enu_low/1.e6),linewidth=2)
        plt.legend(prop={'size':11})
        plt.xlabel('Recoil Threshold (eV)')
        plt.ylabel('Total Event Rate (Events/kg/day)')
        plt.title("ILL Total Rate")
        plt.ylim(1e-3, 1e3)
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig(output_path_prefix+'rate_low_vs_high_example.png')
    fig3.clf()

    Z_arr = [[30], [32]]
    N_arr = [[35.38], [72.64-32]]
    weights_arr = [[1], [1]]
    labels_arr = ["Zn (A=64.4)", "Ge (A=72.6)"]
    lines_arr = ['b-', 'r-']
    point_colors = ['k', 'k']

    plot_A_arr = []
    plot_threshold_arr = []

    t_arr = np.logspace(-2, 4, num=200)
    fig4 = plt.figure()
    fig4.patch.set_facecolor('white')

    for i in range(len(Z_arr)):
        A_sum = 0
        weight_sum = 0
        for j in range(len(Z_arr[i])):
            A_sum += (Z_arr[i][j]+N_arr[i][j])*weights_arr[i][j]
            weight_sum += weights_arr[i][j]
        plot_A_arr.append(A_sum/weight_sum)

        if(differential):
            frac = dsigmadT_cns_rate_compound(t_arr, Z_arr[i], N_arr[i], weights_arr[i], nu_spec, enu_max=enu_low)/\
                   dsigmadT_cns_rate_compound(t_arr, Z_arr[i], N_arr[i], weights_arr[i], nu_spec)
        else:
            frac = total_cns_rate_an_compound(t_arr, enu_low, Z_arr[i], N_arr[i], weights_arr[i], nu_spec)/\
                   total_cns_rate_an_compound(t_arr, 1e7, Z_arr[i], N_arr[i], weights_arr[i], nu_spec)
        frac[np.isnan(frac)] = 0
        frac[frac>1.] = 0 # It's a fraction, so it should be <1.0
        idx = np.argwhere(np.diff(np.sign(frac - frac_interest))).flatten()
        plot_threshold_arr.append(t_arr[idx[0]])
        plt.semilogx(t_arr,frac,lines_arr[i],label=labels_arr[i],linewidth=2)
        np.savetxt("results/ill_reactor_lt_1_8_total_fraction_%s.txt"%
                   labels_arr[i][:2],
                   np.column_stack((t_arr, frac)),
                   header="# T (eV), Fraction")
    plt.legend(prop={'size':11})
    #plt.axhline(y=frac_interest)
    plt.ylim(0., 0.3)
    plt.xlim(1e0, 1e3)
    plt.xlabel('Recoil Energy (eV)')
    if(differential):
        plt.ylabel('Differential Event Rate Fraction, enu<%.1f MeV Over Total'%(enu_low/1.e6))
        plt.title("Chooz Very Near Site (80 m) Differential Rate,  Fraction from enu<%.1f MeV"%(enu_low/1.e6))
    else:
        plt.ylabel('Total Event Rate Fraction, enu<1.8 MeV Over Total')
        plt.title("Chooz Very Near Site (80 m) Total Rate,  Fraction from enu<%.1f MeV"%(enu_low/1.e6))

    plt.axvline(x=1.)
    plt.axvline(x=10.)
    plt.axvline(x=50.)
    plt.axvline(x=100.)
    plt.savefig(output_path_prefix+'rate_low_vs_high_fraction.png')
    fig4.clf()

    fig = plt.figure()
    plt.scatter(plot_A_arr, plot_threshold_arr, color=point_colors)
    plt.xlabel("A")
    if(differential):
        plt.ylabel("Recoil Energy with %.1f%% of Enu<%.1f"%((frac_interest*100), (enu_low/1.e6)))
        plt.title("Recoil Energy for %.1f%% of Low Energy nu"%(frac_interest*100))
    else:
        plt.ylabel("Recoil Threshold with %.1f%% of Enu<%.1f"%((frac_interest*100), (enu_low/1.e6)))
        plt.title("Recoil Threshold for %.1f%% of Low Energy nu"%(frac_interest*100))
    plt.savefig(output_path_prefix+'lowe_domination_threshold.png')

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])       

def plot_flux_xsec(nu_spec):
    # Ge
    Z = 32
    N = 72.64-32.

    e_arr = np.linspace(0., 1e7, 100000)

    # Plot neutrino flux
    fig, host = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.75)
    fig.patch.set_facecolor('white')

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))

    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    #make_patch_spines_invisible(par2)
    # Second, show the right spine.
    #par2.spines["right"].set_visible(True)

    lines = []

    # Spectrum in nu/(MeV*day*cm^2)
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)*1e6
    p_spec, = host.plot(e_arr*1e-6,spec_tot, "k-", label="Neutrino Flux")
    lines.append(p_spec)

    xsec_1eV = total_XSec_cns(1., e_arr, Z, N)
    p_xsec_1, = par1.plot(e_arr*1e-6,xsec_1eV, "c-", label="Ethr=1eV")
    lines.append(p_xsec_1)
    prod_1eV = spec_tot*xsec_1eV
    p_prod_1, = par2.plot(e_arr*1e-6,spec_tot*xsec_1eV, ":", color=lighten_color('c', 0.7))

    xsec_10eV = total_XSec_cns(10., e_arr, Z, N)
    p_xsec_10, = par1.plot(e_arr*1e-6,xsec_10eV, "m-", label="Ethr=10eV")
    lines.append(p_xsec_10)
    prod_10eV = spec_tot*xsec_10eV
    p_prod_10, = par2.plot(e_arr*1e-6,spec_tot*xsec_10eV, ":", color=lighten_color('m', 0.7))

    xsec_50eV = total_XSec_cns(50., e_arr, Z, N)
    p_xsec_50, = par1.plot(e_arr*1e-6,xsec_50eV, "g-", label="Ethr=50eV")
    lines.append(p_xsec_50)
    prod_50eV = spec_tot*xsec_50eV
    p_prod_50, = par2.plot(e_arr*1e-6,spec_tot*xsec_50eV, ":", color=lighten_color('g', 0.7))

    xsec_100eV = total_XSec_cns(100., e_arr, Z, N)
    p_xsec_100, = par1.plot(e_arr*1e-6,xsec_100eV, "b-", label="Ethr=100eV")
    lines.append(p_xsec_100)
    prod_100eV = spec_tot*xsec_100eV
    p_prod_100, = par2.plot(e_arr*1e-6,spec_tot*xsec_100eV, ":", color=lighten_color('b', 0.7))

    xsec_200eV = total_XSec_cns(200., e_arr, Z, N)
    p_xsec_200, = par1.plot(e_arr*1e-6,xsec_200eV, "r-", label="Ethr=200eV")
    lines.append(p_xsec_200)
    prod_200eV = spec_tot*xsec_200eV
    p_prod_200, = par2.plot(e_arr*1e-6,spec_tot*xsec_200eV, ":", color=lighten_color('r', 0.7))

    '''host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)'''

    host.set_xlabel("Neutrino Energy (MeV)")
    host.set_ylabel("Flux [nu/(MeV*day*cm^2)]")
    par1.set_ylabel("CEvNS XSec [cm^2]")
    par2.set_ylabel("Product [nu/(MeV*day)]")
    plt.text(9.8, 1.07*1.e-24, "1.e-40", bbox=dict(facecolor='white', alpha=1.0))
    plt.text(12., 1.07*1.e-24, "1.e-24", bbox=dict(facecolor='white', alpha=1.0))

    host.yaxis.label.set_color(p_spec.get_color())
    par1.yaxis.label.set_color(p_xsec_1.get_color())
    par2.yaxis.label.set_color(p_prod_1.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p_spec.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p_xsec_1.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p_prod_1.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    host.legend(lines, [l.get_label() for l in lines], loc=4, prop={'size':9})
    #plt.legend(loc=4)

    plt.axvline(1.8, color='gray')

    plt.title('Neutrino Flux, CEvNS XSec, and product')
    plt.savefig('plots/flux_xsec_product.png')
    fig.clf()

    # Save results to file
    np.savetxt("results/nu_spectra_xsec_prod.txt",
               np.column_stack((1e-6*e_arr,spec_tot,
                                xsec_1eV, prod_1eV,
                                xsec_10eV, prod_10eV,
                                xsec_50eV, prod_50eV,
                                xsec_100eV, prod_100eV,
                                xsec_200eV, prod_200eV)),
               header="Neutrino Energies: MeV\n"+
               "Neutrino Flux: nu/(MeV*day*cm^2)\n"+
               "Cross Sections: cm^2\n"+
               "Product: nu/(MeV*day)\n"+
               "Neutrino Energy, Neutrino Flux, Ethr=1eV xsec, Ethr=1eV xsec,"+
               "Ethr=10eV xsec, Ethr=10eV xsec, Ethr=50eV xsec, Ethr=50eV xsec,"+
               "Ethr=100eV xsec, Ethr=100eV xsec, Ethr=200eV xsec, Ethr=200eV xsec")

def flux_in_region(nu_spec, low_mev, high_mev):
    flux = quad(nu_spec.d_phi_d_enu_ev, low_mev*1.e6, high_mev*1.e6)[0]
    print("Flux between %.2e MeV and %.2e MeV: %.3e"%
          (low_mev, high_mev, flux))

def main():
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    try:
        os.mkdir('results')
    except OSError as e:
        pass

    # At 0 days, assume all U-235:
    #  [1.0, 0.0, 0.0, 0.0]
    # After 50 days, assume:
    #  [0.98, 0.003, 0.014, 0.003]
    fractions = [1.0, 0.0, 0.0, 0.0]

    # Pessimistic case: 50 MW, 3 50-day cycles per year
    # Optimistic case: 58 MW, 4 50-day cycles per year
    power = 57.8

    # STEREO is located at 10-11 m. Ricochet can be inside this,
    # at 7-10 m. Assume 8 m
    distance = 800 # cm

    nu_spec = NeutrinoSpectrum(distance, power, True,
                               *fractions)
    nu_spec.initialize_d_r_d_enu("u235", "txt",
                                "data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec.initialize_d_r_d_enu("u238", "mueller")
    nu_spec.initialize_d_r_d_enu("pu239", "txt",
                                "data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec.initialize_d_r_d_enu("pu241", "txt",
                                "data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec.initialize_d_r_d_enu("other", "mueller")

    nu_spec_dont_fix = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, False,
                                       *nu_spec.get_fractions())
    nu_spec_dont_fix.initialize_d_r_d_enu("u235", "txt",
                                "data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_dont_fix.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_dont_fix.initialize_d_r_d_enu("pu239", "txt",
                                "data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_dont_fix.initialize_d_r_d_enu("pu241", "txt",
                                "data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_dont_fix.initialize_d_r_d_enu("other", "mueller")


    # Neutrino Spectrum
    plot_neutrino_spectrum(nu_spec, nu_spec_dont_fix,
                           "Fixed<2 MeV", "Extrapolated<2 MeV")

    # CEvNS Differential Cross Section
    plot_cevns_rate_fixed_T()
    plot_cevns_rate_fixed_Enu()
    plot_cevns_rate_vs_T_Enu(nu_spec)

    # Integrated CEvNS Rates
    plot_dsigmadT_cns_rate(nu_spec, "ILL",
                           "plots/ill_differential_cevns_rate.png",
                           "results/ill_differential_cevns_rate.txt",
                           [1e-4,1e0])
    plot_total_cns_rate(nu_spec)

    # Plot to get an idea of systematic from low energy shape
    #plot_dsigmadT_cns_systematic(nu_spec, nu_spec_dont_fix,
    #                             "ILL", "plots/ill_differential_rate_systematic",
    #                             "results/ill_differential_rate.txt",
    #                             [1e-4,1e0])

    # Total Rate vs Threshold and Distance
    plot_total_cns_rate_threshold_dist(nu_spec)

    # Fraction of rate from neutrinos with E<1.8 MeV
    '''flux_in_region(nu_spec, 0., 1.8)
    flux_in_region(nu_spec, 1.8, 10.)
    flux_in_region(nu_spec, 1.8, 11.)
    flux_in_region(nu_spec, 1.8, 15.)
    plot_low_vs_high(nu_spec, 1.8e6,
                     "plots/lowe_1_8_diff_",
                     True)
    plot_low_vs_high(nu_spec, 1.8e6,
                     "plots/lowe_1_8_total_",
                     False,
                     frac_interest=0.1)'''

    # Validation: Plot DC spectrum
    nu_spec_dc = NeutrinoSpectrum(40000, 8500, False, 0.556, 0.071, 0.326, 0.047)
    nu_spec_dc.initialize_d_r_d_enu("u235", "txt",
                                "data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_dc.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_dc.initialize_d_r_d_enu("pu239", "txt",
                                "data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_dc.initialize_d_r_d_enu("pu241", "txt",
                                "data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_dc.initialize_d_r_d_enu("other", "mueller")
    
    plot_dsigmadT_cns_rate(nu_spec_dc, "Validation: Chooz Near Site (400 m)",
                           "plots/dc_differential_rate.png",
                           "results/dc_differential_rate.txt",
                           bounds=[1e-2, 1e2],
                           use_kev=True)

    #plot_flux_xsec(nu_spec)
    
if __name__ == "__main__":
    main()
