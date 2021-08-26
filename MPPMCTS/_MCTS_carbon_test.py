import math
from .utils.carbon_smiles import *
import fcntl
import pickle
import random
from .ASDI import asdi
from .utils import meltingpoint,sascorer
import rdkit.Chem as Chem

def newfun(x):#2*sigmoid-1
    return 2/(1+math.exp(-x))-1

def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def reward(T,constant):
    return tanh(constant*T)

class triple_player_node():
    def __init__(self):
        self.symbol=""
        self.depth=0
        self.current_player=0
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

class TP_P_MCTS():
    def __init__(self):
        pass
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
            next_player=(node.current_player+1)%3
            next_player_current_symbol=node.father.father.symbol
            next_player_next_symbol=next_symbol(next_player_current_symbol)
            for i in range(len(next_player_next_symbol)):
                new_node=triple_player_node()
                new_node.current_player=next_player
                new_node.father=node
                new_node.depth=node.depth+1
                new_node.symbol=next_player_next_symbol[i]
                new_node.index=node.index+[i]
                node.children.append(new_node)
            node.children[0].activate=True
            index=node.children[0].index
            node=node.children[0]
        current_player = node.current_player
        player_smiles = [0,0,0]
        newnode = node
        for num in range(3):
            player_smiles[(current_player + 3 - num) % 3] = most_accepted_smiles(newnode.symbol)
            newnode = newnode.father
        cation = "[n+]1(" + player_smiles[0].strip("&") + ")ccn(" + player_smiles[1].strip("&") + ")c1(" + player_smiles[
            2].strip("&") + ")"
        mol = Chem.MolFromSmiles(cation)
        cation = Chem.MolToSmiles(mol)
        current_node = newnode
        first_built=True if cation not in current_node.get_root().smiles else False
        if cation not in current_node.get_root().smiles:
            current_node.get_root().smiles[cation] = 0
        return index,cation,first_built

    def simulation(self,cation,first_built,ty,sac=False):
        if first_built:
            value=asdi.ASDI(cation,ty)
            if value is not None:
                if sac:
                    s = sascorer.calculateScore(Chem.MolFromSmiles(cation))
                    value = (*value,s)
                    return cation,reward(1/value[4]/s,10),value # 10 might be a good normalized constant
                return cation, reward(1 / value[4], 10), value
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
            root = pickle.loads(f.read())
            node = self.selection(root)
            node_index,cation,first_built= self.expansion(node)
            f.seek(0)
            f.truncate()
            f.write(pickle.dumps((root)))
            f.close()
        except:
            f = open(tree, 'wb')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            root = triple_player_node()
            root.father = root
            root.current_player = -1
            root.root = True
            node = self.selection(root)
            node_index,cation,first_built= self.expansion(node)
            f.seek(0)
            f.truncate()
            f.write(pickle.dumps((root)))
            f.close()
        results = self.simulation(cation,first_built,ty,sac)
        f = open(tree, 'rb+')
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        root = pickle.loads(f.read())
        self.backpropagation(root.children_index(node_index), results)
        f.seek(0)
        f.truncate()
        f.write(pickle.dumps((root)))
        f.close()
        return results,first_built,root.smiles

    def run_n(self,n,tree,results_file,ty,sac):
        counter=0
        while counter<n:
            results,first_built,smiles=self.run(tree,ty,sac)
            f=open(results_file,"a")
            if first_built==True and results[1]!=0:
                print(results[0],results[1],*results[2])
                print(results[0],results[1],*results[2],file=f)
                counter+=1
            f.close()

if __name__=="__main__":
    TP_P_MCTS().run_n(
        n=20,
        tree="tree_ted_h2_sac",
        results_file="results_ted_h2_sac",
        ty="h2",
        sac=True)








