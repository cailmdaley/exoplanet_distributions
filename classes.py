import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from astropy import units as u
import astropy.constants as c
from astrocail import utils
import pymultinest as pmn
import json
from astropy.stats import LombScargle
import subprocess as sp
from glob import glob
import warnings
    
noise=1.

def colplot(x, y, color, **kwargs):
    plt.scatter(x, y, c=color, **kwargs)
    
class Planet:
    def __init__(self, system, m, a, e=0, omega=0, t0=0):
        """initialize a planet instance.
        system : System instance
        m      : mass (Earth masses)
        a      : semi-major axis (au)
        e      : eccentricity
        omega  : argument of pericenter (radians)
        t0     : time since pericenter passage (days)
        """
        self.system = system

        self.a = a
        self.m = m
        self.e = e
        self.t0=t0
        self.omega=omega
        
        self.P = system.calculate_P(self.m, self.a)
class System:
    def __init__(self, m_star, i):
        """
        initialize an exoplanet system.
        m_star : stellar mass (Solar masses)
        i      : inclination (degrees)
        """
        self.m_star = m_star
        self.i = i
        self.lnlike = -1 * np.inf
        
        self.planets = []
        self.dict = {}
        
    def add_planet(self, m, a, e, omega=0, t0=0):
        """add a planet to current system.
        m     : mass (Earth masses)
        a     : semi-major axis (au)
        e     : eccentricity
        omega : argument of pericenter (radians)
        t0    : time since pericenter passage (days)
        """
        self.planets.append(Planet(self, m, a, e, omega, t0))
        
        param_dict = {
        'm' : float(m),
        'a' : float(a),
        'e' : float(e),
        'omega' : float(omega),
        't0' : float(t0)}
        self.dict['p{}'.format(len(self.planets))] = param_dict
        
        return self.planets[-1]
    
    def calculate_P(self, m_planet, a):
        """
        calculate period of for a given planet mass and separation.
        m_planet : mass (Earth masses)
        a        : semi-major axis (au)
        """
        # convert to SI
        m_planet *= u.M_earth.to('kg')
        a        *= u.au.to('m')
        m_star    = self.m_star * u.Msun.to('kg')
        
        P =  np.sqrt(4 * np.pi**2 * a**3 
            / (c.G.si.value * (m_star + m_planet))) # s
        return P * u.s.to('yr')
        
    def calculate_RV(self, date, m_planet, a, e, omega=0, t0=0):
        """ calculate the radial velocity on a certain date for a given set 
        of exoplanet parameters.
        date     : date of observations (days)
        m_planet : mass (Earth masses)
        a        : semi-major axis (au)
        e        : eccentricity
        omega    : argument of pericenter (radians)
        t0       : time since pericenter passage (days)
        """
        # convert to SI
        P = self.calculate_P(m_planet, a) * u.yr.to('day')
        m_planet *= u.M_earth.to('kg')
        a        *= u.au.to('m')
        m_star    = self.m_star * u.Msun.to('kg')
        i = np.radians(self.i)
        
        M = 2 * np.pi * (date-t0) / P
        E = scipy.optimize.fsolve(lambda E: E - e*np.sin(E)-M,M)
        f = 2*np.arctan2( np.sqrt(1+e)*np.sin(E*.5), np.sqrt(1-e)*np.cos(E*.5) )
        RV = np.sqrt( c.G.value / ((m_star + m_planet) * a*(1-e**2))) \
            * m_planet * np.sin(i) \
            * (np.cos(omega + f) + e * np.cos(omega))
        return RV
    
    def take_lightcurve(self, days=365, n_obs=50):
        """
        'observe' lightcurve n_obs times over a given number of days. observations are spaced logarithmically between 1e15 and 1e15+days, then shifted back by 1e15 so that the first observation is at t=0. this is to provide good phase coverage for (almost) all periods. the spacing ends up being *almost* linear.
        days  : observing period (days.. lol)
        n_obs : number of observations to take
        """
        
        self.lightcurve = pd.DataFrame(columns=['JD', 'intrinsic', 'observed'])
        # self.lightcurve.JD = np.logspace(np.log10(1), np.log10(days+1), n_obs) - 1
        self.lightcurve.JD = np.logspace(np.log10(1e5), np.log10(days+1e5), n_obs) - 1e5
        self.lightcurve.intrinsic = [
            sum([self.calculate_RV(date, planet.m, planet.a, planet.e, 
            planet.omega, planet.t0) for planet in self.planets])[0] 
            for date in self.lightcurve.JD]
        
        
        self.lightcurve.observed = self.lightcurve.intrinsic + np.random.normal(size=n_obs) * noise
    
    def make_periodogram(self):
        
        frequency, power = LombScargle(self.lightcurve.JD * u.day.to('year'), self.lightcurve.observed, dy=noise).autopower()
        plt.semilogx(1./frequency, power); plt.show()
        # plt.semilogx(frequency, power); plt.show()
        
    def show_lightcurve(self, model=None, period = None, save=None):
        
        if period:
            self.lightcurve['Phase-Folded JD'] = self.lightcurve.JD % (period * u.yr.to('day'))
            x = 'Phase-Folded JD'
            title = 'Phase-Folded Lightcurve'
        else:
            x = 'JD'
            title = 'Time Series Lightcurve'
        fig, ax = plt.subplots()
        ax.legend()
        self.lightcurve.plot(x=x, y='intrinsic', kind='scatter', ax=ax, color='black', label='intrinsic', title=title)
        self.lightcurve.plot(x=x, y='observed', kind='scatter', ax=ax, label='observed')
        if model:
            self.lightcurve.plot(x=x, y=[model], kind='scatter', ax=ax, color='red', label='fit')
        ax.set_ylabel('m/s')
        if save:
            plt.savefig(save)
        plt.show()
    
    def fit_lightcurve(self, n_planets, path, log_a_range):
        """
        fit observed lightcurve with nested sampling, as implemented by pymultinest.
        n_planets : number of planets to fit
        """
        
        params = []
        for i in range(n_planets):
            params.append('planet{}_m'.format(i))
            params.append('planet{}_a'.format(i))
            # params.append('planet{}_e'.format(i))
            # params.append('planet{}_omega'.format(i))
            params.append('planet{}_t0'.format(i))
        
        def prior(cube, ndim, nparams):
            for i in range(int(nparams/3)):
                # log mass between log(.01) and  log(brown dwarf limit)+0.5
                m_bd = 13*u.Mjup.to('Mearth')
                cube[3*i]   = cube[3*i] * (np.log10(m_bd) + 0.5 + 2.) - 2. 
                # log semi-major axis between log_a_range bounds
                cube[3*i+1] = cube[3*i+1] * (log_a_range[1] - log_a_range[0]) + log_a_range[0] 
                # time since pericenter between 0 and 30 years
                cube[3*i+2] *= 30 * u.year.to('day') 
                
                # cube[3*i+2] *= 1 # eccentricity between 0 and 1
                # cube[3*i+3] *= 2*np.pi # argument of periapse between 0 and 2pi
                
        def lnlike(cube, ndim, nparams):
            
            model_RVs = [
                sum([self.calculate_RV(date, 10**cube[3*i], 10**cube[3*i+1], 0, 
                0, cube[3*i+2]) for i in range(int(nparams/3))])[0] 
                for date in self.lightcurve.JD ]
            
            lnlike = -0.5 * utils.chi(
                data=self.lightcurve.observed, 
                model=model_RVs)         
            
            return lnlike
        
            # some informational print outs for lnlike():
            # if lnlike > self.lnlike: 
            #     self.lnlike = lnlike
            #     # print('model     : m={:.0}, a={:.0}, e={:.1}, omega={:.1}, t0={:.1}'.format(cube[0], cube[1], cube[2], cube[3], cube[4]))
            #     print('planet1 : m={}, a={}, e={}, omega={}, t0={}'.format(self.planets[0].m, self.planets[0].a, self.planets[0].e, self.planets[0].omega, self.planets[0].t0))
            #     print('model1  : m={}, a={}, e={}, omega={}, t0={}'.format(cube[0], cube[1], cube[2], cube[3], cube[4]))
            #     # print('planet2 : m={}, a={}, e={}, omega={}, t0={}'.format(self.planets[1].m, self.planets[1].a, self.planets[1].e, self.planets[1].omega, self.planets[1].t0))
            #     # print('model2  : m={}, a={}, e={}, omega={}, t0={}'.format(cube[5+0], cube[5+1], cube[5+2], cube[5+3], cube[5+4]))
            #     print(lnlike)
            #     print('')
            #     self.lightcurve['model'] = model_RVs

        pmn.run(
            n_live_points=500,
            LogLikelihood=lnlike, 
            Prior=prior, 
            n_dims=len(params),
            wrapped_params = [0,0,1] * int(len(params)/3),
            outputfiles_basename=path, 
            resume=True, verbose=True)
        json.dump(params,open(path + 'params.json','w'))

                
class Distribution:
    
    def fit(self, ms, aas, es=None, omegas=None):
        # a between 0 and 10 au
        # m between 0 and 13 Mjup
        for a in aas:
            for m in ms:
                e=0; omega=0
                path = self.dirs[0] + 'm{}_a{}_e{}_omega{:.1f}'.format(m,a,e,omega)
                
                if glob(path + '*') != []: #ensures lightcurves are overwritten for already-fit systems
                    print('skipping!')
                    continue
                
                sys = System(1, 90)
                planet = sys.add_planet(m=m, a=a, e=e, omega=omega, t0=0)
                
                # hacky way to set orbital phase shift within period bounds
                planet.t0 = np.random.rand() * planet.P; 
                sys.dict['p1']['t0'] = planet.t0
                
                sys.take_lightcurve(days=20*u.yr.to('day'), n_obs=50)
                sys.lightcurve.to_json(path+'_lightcurve.json')
                json.dump(sys.dict,open(path + '_info.json','w'))
                print(sys.dict)
            
                n_planets = 1
                if 0 <= a < 0.1: log_a_range = np.log10([0.005, 0.15])
                elif 0.1 <= a < 2.: log_a_range = np.log10([0.05, 2.5]) 
                else: np.log10([1, 11])
                sys.fit_lightcurve(
                    n_planets=1, 
                    path=path+'_ps{}_'.format(n_planets), 
                    log_a_range=log_a_range)
                # sys.analyzer = pmn.Analyzer(
                #     n_planets*3, 
                #     outputfiles_basename=path+'_ps{}_'.format(n_planets))
                # m_fit, a_fit, t0_fit = sys.analyzer.get_best_fit()['parameters']
    
    def get_system(self, path):
        sys = System(m_star=1, i=90)
        
        try:
            sys.info = json.load(open(path + 'info.json'))
        except IOError:
            warnings.warn('{}info.json does not exist'.format(path), UserWarning)
        
        try:
            sys.lightcurve = pd.read_json(path+'lightcurve.json')
        except ValueError:
            warnings.warn('{}lightcurve.json does not exist'.format(path), UserWarning)
        
        for i, planet in enumerate(sys.info):
            sys.add_planet(m=sys.info[planet]['m'], a = sys.info[planet]['a'], e=sys.info[planet]['e'],
                omega=sys.info[planet]['omega'], t0=sys.info[planet]['t0'])
                
            try:
                params = json.load(open(path + 'ps{}_params.json'.format(i+1)))
            except IOError:
                warnings.warn('{}ps{}_params.json does not exist'.format(path, i+1), UserWarning)
            try:
                sys.analyzer = pmn.Analyzer(len(sys.info) * 3, 
                    outputfiles_basename = path + 'ps{}_'.format(i+1))
                bf_params = sys.analyzer.get_best_fit()['parameters']
                # model_RVs = [
                #     sum([sys.calculate_RV(date, bf_params[3*i], bf_params[3*i+1], 
                #     0, 0, bf_params[3*i+2]) 
                #     for i in range(len(sys.info))])[0] 
                #     for date in sys.lightcurve.JD ]
                model_RVs = [
                    sum([sys.calculate_RV(date, 10**bf_params[3*i], 10**bf_params[3*i+1], 
                    0, 0, bf_params[3*i+2]) 
                    for i in range(len(sys.info))])[0] 
                    for date in sys.lightcurve.JD ]
                sys.lightcurve[planet] = model_RVs
            except (IOError, IndexError):
                sys.analyzer = None
                warnings.warn('{}ps{}_.txt does not exist'.format(path, i+1), UserWarning)
            
        return sys
    def collect_distribution(self, from_json=False):
        # multiindex = pd.MultiIndex(levels=[[]]*2, labels=[[]]*2, names=['system', 'planet'])
        if from_json is False:
            self.library = pd.DataFrame(columns=['m_true','a_true', 'm_fit', 'a_fit', 'lnZ', 'fit_SNR'], dtype=float)
            
            self.paths = [path[:-9] for directory in self.dirs for path in glob(directory + '/*info.json')]
            for i, path in enumerate(self.paths):
                sys = self.get_system(path)
                for j, planet in enumerate(sys.planets):
                    self.library.loc[10*i+j+1, ['m_true', 'a_true']] = np.log10([planet.m, planet.a])
                    if sys.analyzer:
                        m_fit, a_fit = [sys.analyzer.get_stats()['modes'][0]['maximum']][0][:-1]
                        self.library.loc[10*i+j+1, ['m_fit', 'a_fit']] =m_fit, a_fit 
                        self.library.loc[10*i+j+1, 'lnZ'] = \
                            sys.analyzer.get_stats()['modes'][0]['local log-evidence']
                        self.library.loc[10*i+j+1, 'fit_SNR'] = \
                            sys.lightcurve.loc[:,'p{}'.format(j+1)].abs().max()
            self.library.to_json(self.dirs[0] + '_table.json')    
        else:
            print(self.dirs)
            self.library = pd.read_json(self.dirs[0] + '_table.json')
                            
    def plot(self, kind, save=None, exclude=True):
        if kind is 'intrinsic':
            library = self.library
            x = 'm_true'; y = 'a_true'
            xlim = library.m_true.describe()[['min','max']] + np.array([-0.15, .15])
            ylim = library.a_true.describe()[['min','max']] + np.array([-0.05, .05])
            
            #hexbin params
            C = np.log10(self.library.fit_SNR) 
            vmin = int(np.floor(C.min())); vmax = int(np.ceil(C.max()))
            n_colors = vmax - vmin
            cbar_label = 'log Fit SNR'
            mincnt=None 
            
        elif kind is 'observed':
            library = self.library[self.library.fit_SNR > 1.].dropna() \
                if exclude is True else self.library.dropna()
            x = 'm_fit'; y = 'a_fit'
            xlim = library.m_fit.describe()[['min','max']] + np.array([-0.15, .15])
            ylim = library.a_fit.describe()[['min','max']] + np.array([-0.05, .05])
            
            #hexbin params
            C = None; mincnt=1 
            n_colors=5; 
            vmin = vmax = None
            cbar_label = 'Counts'
            
        else:
            raise NameError("kind arg must be either 'observed' or 'intrinsic'")
        
        g = sns.JointGrid(data=self.library, x=x, y=y, xlim=xlim, ylim=ylim)
        g.set_axis_labels(xlabel=r'log $m$ ($M_\oplus$)', ylabel='log $a$ (au)')
        cmap = ListedColormap(sns.color_palette('Blues_d', n_colors).as_hex()[::-1])
        
        # g.plot_joint(colplot, color=IOError, cmap=cmap)
        # g.plot_joint(sns.kdeplot, cmap=cmap, n_levels=5)
        g.plot_joint(plt.hexbin, gridsize=45, mincnt=mincnt,
            C=C, vmin=vmin, vmax=vmax, cmap=cmap)
        
        g.plot_marginals(sns.distplot, hist=False, kde=True, rug=False, 
            kde_kws={'shade': True})#, kde_kws={'cut':0, 'bw':0.4})
            
        cax = g.fig.add_axes([1, .095, .03, .75])
        cbar = plt.colorbar(cax=cax)
        cbar.ax.set_ylabel(cbar_label)
            
        if save:
            g.savefig(save)
        plt.show()
    def compare(self, save=None):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8,6)); sns.despine(left=True);
        sns.distplot(self.library.m_true, ax=ax1, 
            rug=False, hist=False, kde=True, kde_kws={'shade': True},
            label=r'True $m$')
        sns.distplot(self.library[self.library.fit_SNR > 1.].m_fit, ax=ax1, 
            rug=False, hist=False, kde=True, kde_kws={'shade': True},
            label=r'Fit $m$', axlabel=r'log $m$ ($M_\oplus$)')
        sns.distplot(self.library.a_true, ax=ax2, 
            rug=False, hist=False, kde=True, kde_kws={'shade': True},
            label=r'True $a$')
        sns.distplot(self.library[self.library.fit_SNR > 1.].a_fit, ax=ax2, 
            rug=False, hist=False, kde=True, kde_kws={'shade': True},
            label=r'Fit $a$', axlabel=r'log $a$ (au)')    
            
        # add common ylabel
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.ylabel("Normalized Probability Density")
        # ax1.set_title('Mass'); ax2.set_title('Semi-major axis')
        # get rid of meaningless y axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis='y', which='both', labelcolor='none', left='off')
        plt.tight_layout() 
        
        if save:
            fig.savefig(save)
        plt.show()
    def correlations(self, save=None):
        cmap = ListedColormap(sns.color_palette('Blues_d').as_hex()[::-1])
        
        data = self.library[self.library.fit_SNR > 1.].dropna()
        
        g = sns.PairGrid(data=data, palette='Blues_d', size=5,
            x_vars=['m_true', 'a_true'], y_vars=['a_fit', 'm_fit'])
        g.map(plt.hexbin, gridsize=45, cmap=cmap, mincnt=1)
        g.map_offdiag(sns.regplot, scatter=False)
        
        x_labels = [r'True log $m$ (Earth Masses)', r'True log $a$ (au)'] 
        y_labels = [r'Fit log $a$ (au)', r'Fit log $m$ (Earth Masses)']
        for y in range(len(y_labels)):
            for x in range(len(x_labels)):
                if g.axes[y][x].get_ylabel() is not '':
                    g.axes[y][x].set_ylabel(y_labels[y])
                if g.axes[y][x].get_xlabel() is not '':
                    g.axes[y][x].set_xlabel(x_labels[x])
                
        cax = g.fig.add_axes([1, .13, .03, .8])
        cbar = plt.colorbar(cax=cax)
        cbar.ax.set_ylabel('Counts')
        
        if save:
            g.savefig(save)
        plt.show()
    def __init__(self, directories, from_json=False):
        self.dirs = [directories] if type(directories) is str else directories
        self.collect_distribution(from_json)
        
dist = Distribution('planets2', True)    
dist.library
