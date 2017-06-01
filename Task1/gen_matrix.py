#!/usr/bin/python2.7
import sys, os, argparse
from random import uniform
import numpy as np

parser = argparse.ArgumentParser(description='resize images')
parser.add_argument('-s', '--matrixSize', required=True, type=int, nargs='?', help='matrix size')
parser.add_argument('-o', '--outputFileMatrix', default='matrix.txt',type=str, nargs='?', help='output file name')
parser.add_argument('-a', '--outputFileAnswer', default='answer.txt', type=str, nargs='?', help='file with linear system solution')

def main(matrix_size, output_file_matrix, output_file_solution):
    matrix = np.zeros((matrix_size, matrix_size))
    right_parts = np.zeros(matrix_size)     
    with open(output_file_matrix, 'w') as fout:
        fout.write(str(matrix_size) + '\n')
        for i in range(matrix_size):
            for j in range(matrix_size):
                matrix[i][j] = uniform(1, matrix_size + 1)
                
            matrix[i][i] = sum(matrix[i]) + 1
            fout.write(' '.join(map(str, matrix[i])) + ' ')    
            right_parts[i] = uniform(0, matrix_size + 1)
            fout.write(str(right_parts[i]) + '\n')  
    solution = np.linalg.solve(matrix, right_parts)
    with open(output_file_solution, 'w') as fout:
        for i in range(matrix_size):
            fout.write(str(solution[i]) + ' ')
        fout.write('\n') 
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.matrixSize, args.outputFileMatrix, args.outputFileAnswer)
