# %reload_ext autoreload
# %autoreload 2


#plot lightcurve with relevant tables
# planet_params = [np.random.rand(4) * coeff for coeff in [10, 10, 0.8]]
# for a, m, e, in zip(*planet_params): 
#     lusy.add_planet(a, m, e)
# table = pd.DataFrame(planet_params, index=['a (au)', 'm (M earth)', 'e']).T
# table.loc[:, 'P (years)'] = [planet.P * u.s.to('yr') for planet in lusy.planets]
# ax = lusy.lightcurve.plot(); pd.plotting.table(ax, table.round(2), loc='top') 



# compare phasecurve spacings of short and long periods to show they are both well sampled
# lusy = System(m_star=1, i=90)
# p = lusy.add_planet(1,0.005,0)
# lusy.take_lightcurve(days=20*365, n_obs=50)
# lusy.show_lightcurve(save='short_P_folded.png', period=lusy.planets[0].P)
# 
# lusy = System(m_star=1, i=90)
# p = lusy.add_planet(1000,7,0)
# lusy.take_lightcurve(days=20*365, n_obs=50)
# lusy.lightcurve.JD %= (p.P * u.yr.to('day'))
# lusy.show_lightcurve(save='long_P_no_fold.png')



# dist.paths[9]
# sys = dist.get_system(dist.paths[9])
# sys.planets[0].P
# sys.show_lightcurve(model='p1', period=sys.planets[0].P)
# 
# dist.library.sort_values('lnZ')



# sys = System(1, 90)
# p = sys.add_planet(35, 8.5, 0)
# sys.take_lightcurve(365.25*20, 50)
# sys.lightcurve.intrinsic.abs().max()
# sys.show_lightcurve()
