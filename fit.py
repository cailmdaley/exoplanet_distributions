import numpy as np
import astropy.units as u
import classes

dist = classes.Distribution('planets3_bottomup/')
ms = np.round(np.logspace(np.log10(0.1), np.log10(13*u.Mjup.to('Mearth')),100),2)
aas = np.round(np.logspace(np.log10(0.05), np.log10(10),100), 3)

#dist.fit(ms=ms, aas=aas[0:5])
#dist.fit(ms=ms, aas=aas[5:10])
#dist.fit(ms=ms, aas=aas[10:15])
#dist.fit(ms=ms, aas=aas[15:20])
#dist.fit(ms=ms, aas=aas[20:25])
#dist.fit(ms=ms, aas=aas[25:30])
#dist.fit(ms=ms, aas=aas[30:35])
#dist.fit(ms=ms, aas=aas[35:40])
#dist.fit(ms=ms, aas=aas[40:45])
#dist.fit(ms=ms, aas=aas[45:50])
#dist.fit(ms=ms, aas=aas[50:55])
#dist.fit(ms=ms, aas=aas[55:60])
dist.fit(ms=ms, aas=aas)
