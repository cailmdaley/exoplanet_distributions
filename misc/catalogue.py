import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import astrocail.mcmc as mcmc
import astrocail.utils as utils
%reload_ext autoreload
%autoreload 2

print(planets.columns[:])

planets = pd.read_csv('planets.csv', skiprows=range(406), index_col=0); 
rv_planets = planets[planets['pl_discmethod'] == 'radial velocity']

plot_dist()
def plot_dist():
    df = np.log10(rv_planets.loc[:, ['pl_msinij', 'pl_orbsmax'] ])
    df.columns = ['m', 'a']
    df.m += np.log10(u.Mjup.to('Mearth'))

    x = 'm'; y = 'a'
    xlim = df.m.describe()[['min','max']] + np.array([-0.15, .15])
    ylim = df.a.describe()[['min','max']] + np.array([-0.05, .05])

    #hexbin params
    C = None; mincnt=1 
    n_colors=5; 
    vmin = vmax = None
    cbar_label = 'Counts'

    g = sns.JointGrid(data=df, x=x, y=y, xlim=xlim, ylim=ylim)
    g.set_axis_labels(xlabel=r'log $m$ ($M_\oplus$)', ylabel='log $a$ (au)')
    cmap = ListedColormap(sns.color_palette('Blues_d', n_colors).as_hex()[::-1])

    g.plot_joint(plt.hexbin, gridsize=50, mincnt=mincnt,
    C=C, vmin=vmin, vmax=vmax, cmap=cmap)

    g.plot_marginals(sns.distplot, hist=False, kde=True, rug=False, 
    kde_kws={'shade': True})#, kde_kws={'cut':0, 'bw':0.4})

    cax = g.fig.add_axes([1, .095, .03, .75])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.set_ylabel(cbar_label)
    cbar.set_ticks([1,2,3,4,5])
    
    g.savefig('../figures/real_world_dist.png')
    plt.show()


df.plot(kind='scatter', x='m', y='a'); plt.show()

rv_planets.reset_index(inplace=true)

rv_planets.plot(kind='scatter', x='st_dist', y='pl_rvamperr1', loglog=True); plt.show()

df = rv_planets[['st_nts', 'pl_rvamperr1']].dropna().sort_values('st_nts')
df = df[df.st_nts > 0]

np.polyfit(df.iloc[:,0], df.iloc[:,1], deg=3)
df.plot(kind='scatter',x=0,y=1, xlim=[-1,10], ylim=[0,20]); plt.plot(sorted(df.iloc[:,0]), np.sum([coef*df.iloc[:,0]**n for n, coef in enumerate(reversed(np.polyfit(df.iloc[:,0], df.iloc[:,1], deg=3)))], axis=0), color='r'); plt.show()


f1 = lambda x: x/df.st_nts
run1 =mcmc.run_emcee_simple('run1', nsteps=500, nwalkers=10, 
    lnprob=lambda x: -1 * utils.chi(df.pl_rvamperr1, f1(*x)) / 2.,
    to_vary = (('c1',  10,  3, -np.inf,  np.inf),) );
run1.kde(); plt.show()
run1.evolution(); plt.show()
run1.groomed.lnprob.max()
coeffs = run1.groomed[run1.groomed.lnprob == run1.groomed.lnprob.max()][['c1']].values[0]

# df.plot(kind='scatter',x=0,y=1, xlim=[-1,10], ylim=[0,20])
# plt.plot(df.iloc[:,0], f1(*coeffs), color='r');
# plt.show()


ax = sns.violinplot(x=df.iloc[:,0], y=df.iloc[:,1], scale='width'); ax.set_yscale('log'); ax.set_xlim([-1,10]);plt.show()

blah = df.copy(); blah.pl_rvamperr1 -= f1(*coeffs)
ax = sns.violinplot(x=blah.iloc[:,0], y=blah.iloc[:,1], scale='width'); ax.set_xlim([-1,10]); ax.set_ylim([-15,10]); plt.show()

ax = sns.violinplot(x=blah.iloc[:,0], y=blah.iloc[:,1], scale='width'); ax.set_xlim([-1,10]); ax.set_ylim([-15,10]); plt.show()
