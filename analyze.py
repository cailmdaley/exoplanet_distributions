import classes

# dist = classes.Distribution(['planets3_bottomup', '../exo_project/planets3'])

dist = classes.Distribution('planets3_bottomup')
dist.collect_distribution(from_json=True)
dist.library.shape

#%reload_ext autoreload
#%autoreload 2
dist.plot('intrinsic', save='figures/planets3_intrinsic')
dist.plot('observed', save='figures/planets3_observed')
dist.compare(save='figures/planets3_comparison')
dist.correlations(save='figures/planets3_correlations')

# lib = dist.library
# (lib.m_true - lib.m_fit).mean()
# (lib.a_true - lib.a_fit).mean()
