import numpy as np
import astropy.units as u
import classes

dist = classes.Distribution('planets3_bottomup/')
ms = np.round(np.logspace(np.log10(0.1), np.log10(13*u.Mjup.to('Mearth')),100),3)
aas = np.round(np.logspace(np.log10(0.1), np.log10(10),100), 3)[::-1]
aas2 = np.round(np.logspace(np.log10(0.05), np.log10(10),100), 3)[::-1]
spliced_aas = np.append(aas[:-51], aas2[43:-13])

dist.fit(ms=ms, aas=aas)
