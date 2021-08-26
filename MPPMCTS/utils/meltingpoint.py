# A melting point estimation function
# source:
#       A group contribution method to predict the melting point of ionic liquids
#       https://doi.org/10.1016/j.fluid.2011.09.018


import rdkit.Chem as Chem


def melting_point(smi):
    # ''' melting point estimation
    # :param smi: cation smi (seperated [H] connected to the imidozole ring must be indicated)
    # :return: The melting point of ILs (anion is NTF2)
    # '''
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    patts = [
        ["[n+]1c([H])c([H])nc1", 22.837],
        ["S(F)(F)(F)(F)F", -0.409],
        ["[C^3](F)(F)F", 25.393],
        ["C(F)F", 2.247],
        ["c1ccccc1", -6.072],
        ["C([H])=C([H])[H]", -9.956],
        ["S(=O)(=O)", 8.119],
        ["Cl", 68.576],
        ["Br", 73.041],
        ["F", 18.028],
        ["N([H])[H]", -8.971],
        ["C#N", 72.688],
        ["[C^2](=O)O", 28.709],
        ["[O^3][H]", -14.994],
        ["[O]", -10.468],
        ["[C^3]([H])([H])[H]", -4.384],
        ["[C^3]([H])[H]", -3.759],
        ["[C^3][H]", 13.127],
        ["[C^3]", 76.190],
        ["[H]", -20.980]
    ]
    tc = 0
    rest = mol
    num = []
    for patt in patts:
        pattsmi = Chem.MolFromSmarts(patt[0])
        nn = len(rest.GetSubstructMatches(pattsmi))
        tc += nn * patt[1]
        num.append(nn)
        rest = Chem.DeleteSubstructs(rest, pattsmi)

    ta = -(113.15 * 1) + 114.170 * 2 - 51.844 * 2
    tm = tc + ta + 288.7

    rest_num = len(rest.GetAtoms())
    if rest_num != 0:
        return None
    return tm


if __name__ == "__main__":
    print(melting_point("C[n+]9ccn(OCCCCCO)c9"))
