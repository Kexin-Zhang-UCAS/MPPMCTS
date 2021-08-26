import math
import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import pickle
import fcntl
from .utils import smiles,sascorer
from .ASDI import asdi



def reward(T,constant):
    return math.tanh(constant*T)

class Node():
    def __init__(self):
        self.symbol=[0]
        self.depth=0
        self.father=None
        self.children=[]
        self.smiles=dict()
        self.visits=0
        self.value=0
        self.root=False
        self.activate=False
        self.index=[]
    def get_root(self):
        node = self
        while node.root ==False:
            node = node.father
        return node
    def children_index(self,index):
        root=self.get_root()
        node=root
        for i in index:
            node=node.children[i]
        return node
    def ucb(self,C=0.5):
        if self.visits==0:
            return math.inf
        else:
            return self.value/self.visits+C*math.sqrt(math.log(self.father.visits)/(self.visits))

class P_MCTS():
    def selection(self,root):
        node=root
        ## if one node is simulation, it can alos be selected, yet it will only be expanded.
        while node.children!=[]:
            selected_node=[child for child in node.children]
            ucb_list=[child.ucb(2) for child in selected_node]
            node=selected_node[ucb_list.index(max(ucb_list))]
        return node
    # expand node at give place
    def expansion(self,node):
        if node.visits==0 and node.activate==False:
            node.activate=True
            index=node.index
            node=node
        else:
            labels = smiles.next_lists(node.symbol)
            for i in range(3):
                new_node=Node()
                new_node.father=node
                new_node.depth=node.depth+1
                new_node.symbol=labels[i]
                new_node.index=node.index+[i]
                node.children.append(new_node)
            node.children[0].activate=True
            index=node.children[0].index
            node=node.children[0]

        newnode = node
        smi=smiles.list2string(smiles.final_list(newnode.symbol))
        cation = smi[:1] + "(n1c[n+]([H])cc1)" + smi[1:]
        if Chem.MolFromSmiles(cation) == None:
            test=0
            for id, i in enumerate(smi[1:]):
                if i in ["N", "C"]:
                    cation = smi[:id + 2] + "(n1c[n+]([H])cc1)" + smi[id + 2:]
                    test+=1
                    if Chem.MolFromSmiles(cation)!=None:
                        break
                    if test==5:
                        break
        mol = Chem.MolFromSmiles(cation)
        if mol!=None:
            cation = Chem.MolToSmiles(mol)
        current_node = newnode
        first_built=True if cation not in current_node.get_root().smiles else False
        if cation not in current_node.get_root().smiles:
            current_node.get_root().smiles[cation] = 0
        return index,cation,first_built

    def simulation(self,cation,first_built,ty,sac=False):
        if first_built:
            mol=Chem.MolFromSmiles(cation)
            if mol==None:
                return cation,-1,0
            value= asdi.ASDI(cation, ty)
            # value=None if rand_int==0 else [1,1,1,1] # DEBUG
            # value = [jrvalue(cation)["Molecular Refractivity"]]
            if value is not None:
                if sac:
                    s=sascorer.calculateScore(Chem.MolFromSmiles(cation))
                    value=(*value,s)
                    return cation,reward(1/value[4]/s,10),value
                return cation,reward(1/value[4],10),value
            else:
                return cation,0,0 # wrong systems case (just delete node and random run later)
        else:
            return cation,0,0 # repeat case

    def backpropagation(self,node,results):
        smiles,R,V=results
        ## TODO: The value -1 are not suitable for this case, because reasonable cases are rare.
        ## If I set -1, that may cause local minimum
        # if R==-1:
        #     node.activate = False
        #     del node.get_root().smiles[smiles]
        #     node.value += 0
        #     node.visits += 1
        #     while node.root != True:
        #         node = node.father
        #         node.value += 0
        #         node.visits += 1
        # else:
        node.value+=R
        node.visits+=1
        if R!=0:
            node.get_root().smiles[smiles]=(R,V)
        node.activate=False
        while node.root!=True:
            node=node.father
            node.value += R
            node.visits += 1

    def run(self,tree,ty,sac):
        try:
            f = open( tree, 'rb+')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            root = pickle.load(f)
            node = self.selection(root)
            node_index,cation,first_built= self.expansion(node)
            f.seek(0)
            f.truncate()
            pickle.dump(root,f)
            f.close()
        except:
            f = open(tree, 'wb')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            root = Node()
            root.father = root
            root.current_player = -1
            root.root = True
            node = self.selection(root)
            node_index,cation,first_built= self.expansion(node)
            f.seek(0)
            f.truncate()
            pickle.dump(root,f)
            f.close()
        results = self.simulation(cation,first_built,ty,sac)
        f = open(tree, 'rb+')
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        root = pickle.load(f)
        self.backpropagation(root.children_index(node_index), results)
        f.seek(0)
        f.truncate()
        pickle.dump(root,f)
        f.close()
        return results,first_built,root.smiles

    def run_n(self,n,tree,results_file,ty,sac):
        '''
        :param n: numbers of valid results
        :param tree: MCT path
        :param results_file: log path
        :param ty: gas adsorption type
        :param sac: whether to
        :return:
        '''
        counter=0
        while counter<n:
            f = open(results_file, "a")
            results,first_built,smiles=self.run(tree,ty,sac)
            if first_built==True:
                if results[1]!=-1 and results[1]!=0:
                    print(results[0], results[1], results[2])
                    print(results[0], results[1], *results[2],file=f)
                    counter+=1
            f.close()
if __name__=="__main__":
    P_MCTS().run_n(
        n=20,
        tree="tree_rnn_h2",
        results_file="results_rnn_h2",
        ty="h2",
        sac=True
    )
