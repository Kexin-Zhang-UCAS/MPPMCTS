from rdkit import Chem
from rdkit.Chem import AllChem
import os
from . import to_sigma3
from rdkit.Chem import Descriptors
from .calculate_gamma import COSMOSAC_gamma
import math
import zipfile
import datetime,random
def random_name():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")+str(random.randint(100,999))

def vapor(T,type="co2"):
    p={
        "co2":[140.54 ,-4375 ,-21.268 ,4.0909E-02, 1],
        "h2": [39.543 ,-1320.21, -3.527 ,3.48E-05 ,2],
        "n2": [30.895 ,-847.518, -1.999, 3.7676E-03 ,1],
        "h2s": [85.584, -3839.9, -11.199 ,1.88E-02 ,1]
    }
    return math.exp(p[type][0]+p[type][1]/T+p[type][2]*math.log(T)+p[type][3]*(T**p[type][4]))/100000




def ASDI(smi,type="h2",sigma_db=None):
    if sigma_db==None:
        sigma_db=__file__.rstrip("asdi.py")+"SigmaDB.zip"
    profiles = zipfile.ZipFile(sigma_db, "a")
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    tmpname=random_name()
    os.mkdir(tmpname)
    profiles.extract("tf2n.sigma", tmpname)
    profiles.extract("co2.sigma", tmpname)
    profiles.extract("n2.sigma", tmpname)
    profiles.extract("h2.sigma", tmpname)
    profiles.extract("h2s.sigma", tmpname)

    try:
        if smi+".sigma" in profiles.namelist():
            profiles.extract(smi+".sigma", tmpname)
            # os.system("mv profiles_rnn/* ./;rm -rf profiles_rnn")
        else:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            with open(tmpname+"/"+smi+"1.com","w") as f:
                print("%chk=c.chk\n%mem=20GB\n%nproc=32\n#p b3lyp/6-31g* p\n\ncosmo\n",file=f)
                blocks=Chem.MolToXYZBlock(mol).split("\n")
                print("1 1",file=f)
                for i in blocks[2:]:
                    print(i,file=f)
            with open(tmpname+"/"+smi+"2.com","w") as f:
                print("%chk=c.chk\n%mem=20GB\n%nproc=32\n#P BP86/TZVP scf=(tight,novaracc) SCRF=COSMORS guess=read geom=checkpoint",file=f)
                print("\nBP86/TZVP COSMO SINGLE POINT\n\n1 1\n\n"+smi+".cosmo",file=f)
            os.system("cd "+tmpname+";g09 \""+smi+"1.com\"")
            os.system("cd "+tmpname+";g09 \"" + smi + "2.com\"")
            to_sigma3.convert_cosmo_to_sigma(tmpname+"/"+smi + ".cosmo")
            profiles.close()
            profiles = zipfile.ZipFile(sigma_db, "a")
            if smi + ".sigma" not in profiles.namelist():
                profiles.write(tmpname+"/"+smi+".sigma",smi+".sigma")


        # os.system("mv profiles_rnn/* ./;rm -rf profiles_rnn")
        Hco2 = COSMOSAC_gamma(tmpname,[smi, "tf2n", "co2"], [1, 1, 0], 298.15)[-1] * vapor(298.15, "co2")
        Hco22 = COSMOSAC_gamma(tmpname,[smi, "tf2n", "co2"], [1, 1, 0], 323.15)[-1] * vapor(323.15, "co2")
        H1=Hco2
        MW = Descriptors.MolWt(mol) + 280.14
        H1_=H1*MW/44.001
        D_ = Hco2 / Hco22
        SIJ_ = 0
        asdi = 0
        if type=="h2":
            H2=COSMOSAC_gamma(tmpname,[smi, "tf2n", "h2"], [1, 1, 0], 298.15)[-1] * vapor(298.15, "h2")
            SIJ=H2/H1*44.001/2.016
            SIJ_=1/SIJ
            asdi = H1_*SIJ_*D_
        elif type=="n2":
            H2 = COSMOSAC_gamma(tmpname,[smi, "tf2n", "n2"], [1, 1, 0], 298.15)[-1] * vapor(298.15, "n2")
            SIJ = H2 / H1 * 44.001 / 28
            SIJ_ = 1 / SIJ
            asdi = H1_ * SIJ_ * D_
        elif type=="h2s":
            H2 = COSMOSAC_gamma(tmpname,[smi, "tf2n", "h2s"], [1, 1, 0], 298.15)[-1] * vapor(298.15, "h2s")
            SIJ = H2 / H1 * 44.001 / 34.076
            SIJ_ = 1 / SIJ
            asdi = H1_ * SIJ_ * D_
        os.system("rm -rf "+tmpname)
        return (H1,H1_,SIJ_,D_,asdi,MW)
    except:
        try:
            os.system("rm -rf " + tmpname)
        except:
            pass


if __name__=="__main__":
    print("CC(Cn1cc[nH+]c1)CC(C)(C)NC(=O)NCC(C)(C)c1ccncc1",ASDI("CC(Cn1cc[nH+]c1)CC(C)(C)NC(=O)NCC(C)(C)c1ccncc1","h2"))






# def smi_r(smi):
#     return smi.replace("[","L").replace("]","R").replace("(","l").replace(")","r").replace("+","_").replace("@","A")
# def r_smi(r):
#     return r.replace("L","[").replace("R","]").replace("l","(").replace("r",")").replace("_","+").replace("A","@")

