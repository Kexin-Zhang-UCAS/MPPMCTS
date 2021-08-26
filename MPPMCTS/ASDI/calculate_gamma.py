import cCOSMO
import os
import numpy as np


def COSMOSAC_gamma(path, sigmas, x, T, using2010=True):
    with open(path + "/sigmas3.txt", "w") as f:
        for id, i in enumerate(sigmas):
            print(str(id) + "      " + i, file=f)
    db = cCOSMO.DelawareProfileDatabase(path + "/sigmas3.txt", path + "/")
    for iden in sigmas:
        db.add_profile(db.normalize_identifier(iden))
    COSMO = cCOSMO.COSMO1(sigmas, db)
    os.system("rm -f sigmas3.txt")
    if using2010:
        return np.exp((COSMO.get_lngamma_comb(T, x) + COSMO.get_lngamma_resid(T, x)).tolist()).tolist()
    return np.exp(np.array(COSMO.get_lngamma(T, x).tolist())).tolist()

# print("pf6 pf6 co2 (0.5 0.5 0) 298.15:".ljust(20),np.exp(np.array(COSMOSAC_lngamma([ "pf6","pf6", "co2" ],[0.5,0.5,0],298.15,False))))


# print(COSMOSAC_gamma([],[],298))
