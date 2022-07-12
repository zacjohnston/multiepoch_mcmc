Observation Data
----------------
The files containing observational burst data are located here, as used by the `obs_data.ObsData` class.

The data is taken from the MINBAR archive.
See [Galloway et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJS..249...32G/abstract) 
and the associated [github repository](https://github.com/outs1der/minbar).

Currently, the only file used by `ObsData` is `gs1826.dat`, which contains average burst/system properties for each epoch, as follows:

Name              | units       | Description
------------------|-------------|------------
`epoch`           | year        | epoch year
`fluence`         | erg/cm^2    | time-integrated burst flux
`fper`            | erg/s/cm^2  | persistent flux between bursts
`cbol`            | --          | bolometric correction factor
`peak`            | erg/s/cm^2  | peak burst flux
`rate`            | bursts/day  | burst rate
`fedd`            | erg/s/cm^2  | inferred Eddington flux limit

Each burst/system property has an associated uncertainty value, indicated with a `u` prefix, e.g., `u_fluence`.

While not used by this package, files containing average burst lightcurve data are also provided for reference and potential future implementation.  