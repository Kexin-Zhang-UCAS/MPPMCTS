# This file have several functions that used to process alkyl brach generation.


import sys
from math import ceil
from zss import simple_distance, Node
import pickle
import random
from rdkit import Chem
from rdkit.Chem import AllChem
sys.setrecursionlimit(10000000)

def divide_n(n):
    '''
    :param n: numbers of carbon in a partial tree
    :return: a list that lists MPPMCTS combination of that tree, e.g. 10=[[3,3,3],...]
    '''
    nlist=list()
    for i in range(n-1,ceil((n-1)/3)-1,-1):
        rest=n-1-i
        jmax=min(rest,i)
        jmin=n-1-i-jmax+int(jmax/i)-((n-1)%3==0 and i==(n-1)/3)
        for j in range(jmax,jmin-1,-1):
            # print(n,"- 1 = ",i,j,rest-j)
            nlist.append([i,j,rest-j])
    return nlist

def num_pn(n):
    if n==0 or n==1:
        return 1
    else:
        nlist=divide_n(n)
        total=0
        for i in nlist:
            if i[0]==i[1] and i[0]==i[2] and i[0]!=0:
                for j in range(1,num_pn(i[0])+1):
                    total+=j+(j-1)**2
                pass
            else:
                total+=num_pn(i[0])*num_pn(i[1])*num_pn(i[2])
        return total

def get_nsmiles(n):
    if n==0:
        return ["[H]"],[Node("H")]
    if n==1:
        return ["C"],[Node("C")]
    else:
        nlist=divide_n(n)
        target=[]
        nodes=[]
        for i in nlist:
            if i[0]==i[1] and i[0]==i[2] and i[0]!=0:
                for j1 in range(len(get_nsmiles(i[0])[0])):
                    for j2 in range(j1):
                        for j3 in range(j1):
                            target.append("C("+get_nsmiles(i[0])[0][j1]+")("+get_nsmiles(i[0])[0][j2]+
                                          ")("+get_nsmiles(i[0])[0][j3]+")")
                    for j2 in range(j1+1):
                        target.append("C(" + get_nsmiles(i[0])[0][j1] + ")(" + get_nsmiles(i[0])[0][j1] +
                                      ")(" + get_nsmiles(i[0])[0][j2] + ")")
                for j1 in range(len(get_nsmiles(i[0])[1])):
                    for j2 in range(j1):
                        for j3 in range(j1):
                            a=Node("C")
                            if get_nsmiles(i[0])[1][j1].label=="C":
                                a.addkid(get_nsmiles(i[0])[1][j1])
                            if get_nsmiles(i[0])[1][j2].label=="C":
                                a.addkid(get_nsmiles(i[0])[1][j2])
                            if get_nsmiles(i[0])[1][j3].label=="C":
                                a.addkid(get_nsmiles(i[0])[1][j3])
                            nodes.append(a)
                    for j2 in range(j1+1):
                        a=Node("C")
                        if get_nsmiles(i[0])[1][j1].label=="C":
                            a.addkid(get_nsmiles(i[0])[1][j1]).addkid(get_nsmiles(i[0])[1][j1])
                        if get_nsmiles(i[0])[1][j2].label=="C":
                            a.addkid(get_nsmiles(i[0])[1][j2])
                        nodes.append(a)
                pass
            else:
                for smile1 in get_nsmiles(i[0])[0]:
                    for smile2 in get_nsmiles(i[1])[0]:
                        for smile3 in get_nsmiles(i[2])[0]:
                            if smile2=="[H]" and smile3==smile1:
                                target.append("C("+smile1+")("+smile3+")")
                            elif smile3=="[H]" and smile2==smile1:
                                target.append("C("+smile1+")("+smile2+")")
                            elif smile2=="[H]" and smile3!="[H]":
                                target.append("[C@@H](" + smile1 + ")(" + smile3 + ")")
                            elif smile3=="[H]" and smile2!="[H]":
                                target.append("[C@H](" + smile1 + ")(" + smile2 + ")")
                            elif smile2=="[H]" and smile3=="[H]":
                                target.append("C("+smile1+")")
                            else:
                                target.append("C("+smile1+")("+smile2+")("+smile3+")")
                for node1 in get_nsmiles(i[0])[1]:
                    for node2 in get_nsmiles(i[1])[1]:
                        for node3 in get_nsmiles(i[2])[1]:
                            a = Node("C")
                            if node1.label == "C":
                                a.addkid(node1)
                            if node2.label == "C":
                                a.addkid(node2)
                            if node3.label == "C":
                                a.addkid(node3)
                            nodes.append(a)
        return target,nodes


smiles = []
nodes = []
for i in range(10):
    for j in get_nsmiles(i)[0]:
        smiles.append(j)
    for j in get_nsmiles(i)[1]:
        nodes.append(j)

smiles_len=[len(i) for i in smiles]
# print(max(smiles_len))
for i in range(len(smiles)):
    smiles[i]=smiles[i].ljust(40,"&")
# for i in smiles: print(i)

try:
    f=open(__file__.rstrip("carbon_smiles.py")+"smiles_dict", 'rb+')
    smiles_dict=pickle.loads(f.read())
    f.close()
except:
    smiles_dict=dict()
    for i in range(len(smiles)):
        smiles_dict[smiles[i]+smiles[i]]=0.0
    for i in range(len(smiles)-1):
        for j in range(i+1,len(smiles)):
            value=simple_distance(nodes[i],nodes[j])
            smiles_dict[smiles[i]+smiles[j]]=value
            smiles_dict[smiles[j] + smiles[i]] = value
    with open("../../model/smiles_dict", 'wb') as f:
        pickle.dump(smiles_dict,f)

def next_smiles(smile):
    smile_len = len(smile)
    next_smiles_list=[]
    for i in smiles:
        if smile ==i[:smile_len]:
            next_smiles_list.append(i)
    return next_smiles_list
def next_symbol(smile):
    smile_len = len(smile)
    next_smiles_list = []
    for i in smiles:
        if smile ==i[:smile_len] and i[:smile_len+1] not in next_smiles_list:
            next_smiles_list.append(i[:smile_len+1])
    return next_smiles_list
# print(next_symbol(""))

def most_accepted_smiles(smiles):
    smiles_list=next_smiles(smiles)
    if len(smiles_list)>10:
        all_value=[]
        for i in smiles_list:
            value_list=[]
            for j in smiles_list:
                value_list.append(smiles_dict[i+j])
            all_value.append(sum(value_list)/len(value_list))
        minimum_value=min(all_value)
        condition_smiles=[]
        for i in range(len(all_value)):
            if all_value[i]==minimum_value:
                condition_smiles.append(smiles_list[i])
        rand_index=random.randint(0,len(condition_smiles)-1)
        return condition_smiles[rand_index]
    else:
        rand_index = random.randint(0, len(smiles_list) - 1)
        return smiles_list[rand_index]
def draw_smile_from_pdb(smile,fname):
    mol=Chem.MolFromSmiles(smile)
    mol=Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    with open(fname,'w') as f:
        print(Chem.MolToPDBBlock(mol),file=f)




if __name__=="__main__":
    pass
    print(divide_n(5))

