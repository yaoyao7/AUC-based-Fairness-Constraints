
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=float, default=1)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--outer_iter', type=int, default=10)
parser.add_argument('--T', type=int, default=10)
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--lr_0', type=float, default=0.1) # initial learning rate
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--tol', type=float, default=0.5)
parser.add_argument('--theta', type=float, default=1.0)
parser.add_argument('--fair_con', type=float, default=0.5)
parser.add_argument('--B', type=float, default=1.0)
parser.add_argument('--tau_0', type=float, default=0.1)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--dataname', type=str, default='a9a')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--mu', type=float, default=0.1)
parser.add_argument('--class_id', type=int, default=0)



para = parser.parse_args()