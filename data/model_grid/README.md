Burst model grid
----------------
The file `johnston_2020.txt` covers the full grid of 3840 KEPLER burst simulations used in Johnston et al. (2020).\
Please refer to the paper for additional detail on the models and quantities below.

Each row corresponds to a model, and the columns list the input parameters and the resulting burst properties.

### Input parameters
 - `mdot`: Accretion rate, as fraction of Eddington [8.775e4 g/cm^2/s]
 - `qb`: Base heat flux [MeV/nucleon]
 - `x`: Accreted hydrogen [mass fraction]
 - `z`: Accreted CNO metallicity [mass fraction]
 - `g`: Surface gravity [10^14 cm/s^2]

### Output burst properties
- `n_bursts`: Number of bursts averaged over (excluding burn-in and short recurrence-time bursts)
- `rate`: Burst rate [per day]
- `dt`: Recurrence time [hr] (inverse of burst rate)
- `energy`: Total burst energy [erg]
- `peak`: Peak burst luminosity [erg/s]
 
Each burst property is the average taken over `n_bursts` and has an associated 1-sigma standard deviation (e.g., `u_rate`).

**IMPORTANT**: all properties are in the local Newtonian frame of KEPLER, and must be corrected for GR in order to compare with other codes (e.g. MESA) or with observations (see paper).


Loading table in Python
-----------------------

    import pandas as pd
    table = pd.read_csv('burst_model_grid.txt', delim_whitespace=True)

Note that the table is not strictly ordered, so the easiest way to
select certain models is using a boolean mask:

    mask = table['mdot'] == 0.10
    subset = table[mask]

You can combine masks to specify multiple parameters:
    
    import numpy as np
    mask_1 = table['mdot'] == 0.10
    mask_2 = table['x'] == 0.70
    mask = np.logical_and(mask_1, mask_2)
    subset = table[mask]