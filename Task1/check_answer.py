#!/usr/bin/python2.7
import sys, os, argparse
import numpy as np
from numpy import linalg as LA

parser = argparse.ArgumentParser(description='check solutions')
parser.add_argument('-s', '--vecShape', required=True, type=str, nargs='?', help='vector shape')
parser.add_argument('-f1', '--file1', required=True, type=str, nargs='?', help='input file2')
parser.add_argument('-f2', '--file2', required=True, type=str, nargs='?', help='input file1')

EPS=0.01

def main(file1, file2, vec_shape):
    vec1, vec2 = None, None
    with open(file1, 'r') as fin:
        line = fin.readlines()[0]
        vec1 = line.strip().split(' ')
    vec1 = np.asarray(vec1, dtype=np.float)
    with open(file2, 'r') as fin:
        line = fin.readlines()[0]
        vec2 = line.strip().split()
    vec2 = np.asarray(vec2, dtype=np.float)
    if LA.norm(vec1 - vec2) < EPS:
        sys.stdout.write("Pass test\n")
    else:
        sys.stdout.write("Fail test\n")
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.file1, args.file2, args.vecShape)
