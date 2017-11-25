dist = Distribution('planets3/')
ms = np.round(np.logspace(np.log10(0.1), np.log10(13*u.Mjup.to('Mearth')),100),3)[::-1]
aas = np.round(np.logspace(np.log10(0.05), np.log10(10),100), 3)[::-1]
dist.fit(ms=ms, aas=aas)
