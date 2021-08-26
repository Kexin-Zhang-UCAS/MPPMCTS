import argparse
from . import MCTS_all
parser = argparse.ArgumentParser(prog='MPPMCTS')
parser.add_argument("tree", help="Monte Carlo Tree Path")
parser.add_argument("out", help="Results File Path")
parser.add_argument("--turn", default="20", help="Numbers of valid SMILES in one job")
parser.add_argument("--gas", default="h2", choices=["h2", "n2", "h2s"], help="Gas type")
parser.add_argument("--usesac", action='store_true', help="Wheter to include SAC score in generation")
args = parser.parse_args()
def run():
    MCTS_all.P_MCTS().run_n(
        n=int(args.turn),
        tree=args.tree,
        results_file=args.out,
        ty=args.gas,
        sac=args.usesac
    )

# if __name__=="__main__":
#     main()
