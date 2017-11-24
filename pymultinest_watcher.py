import pymultinest as pmn
import seaborn as sns 
import matplotlib.pyplot as plt

output='test_planets/m1000.0_a0.1_e0_omega0.0_ps1_'
plotter = pmn.watch.ProgressPlotter(3, 
    outputfiles_basename=output,
    interval_ms=1000)
plotter.run()
plotter._plot_live()

a = pmn.Analyzer(3, outputfiles_basename=output)
p = pmn.plot.PlotMarginal(a)
p1 = pmn.plot.PlotMarginalModes(a)
p.plot_conditional(0,1); plt.show()
p1.plot_modes_marginal(1,0); plt.show()
p1.plot_conditional(0,1); plt.show()
