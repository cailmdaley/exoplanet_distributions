import classes


# dist = classes.Distribution('planets2/')
# dist = classes.Distribution('../exo_project/planets3')

#dist = classes.Distribution('../exo_project/planets3')
#dist.plot('intrinsic', save='figures/planets3_intrinsic')
#dist.plot('observed', save='figures/planets3_observed')
#dist.compare(save='figures/planets3_comparison')
#dist.correlations(save='figures/planets3_correlations')

dist = classes.Distribution('planets3_bottomup')
dist.plot('intrinsic', save='figures/planets3_bottomup_intrinsic')
dist.plot('observed', save='figures/planets3_bottomup_observed')
dist.compare(save='figures/planets3_bottomup_comparison')
dist.correlations(save='figures/planets3_bottomup_correlations')
