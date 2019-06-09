#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:28:44 2019

@author: mandeepsingh
"""
"""
Maximum path sum that starting with any cell of 0-th row and ending with any cell of (N-1)-th row
Given a N X N matrix Mat[N][N] of positive integers. There are only three possible moves from a cell (i, j)

(i+1, j)
(i+1, j-1)
(i+1, j+1)
Starting from any column in row 0, return the largest sum of any of the paths up to row N-1.

Input : mat[4][4] = { {4, 2, 3, 4},
                      {2, 9, 1, 10},
                      {15, 1, 3, 0},
                      {16, 92, 41, 44} };
Output :120
path : 4 + 9 + 15 + 92 = 120 

"""
mat = [[4,2,3,4],[2,9,1,10],[15,1,3,0],[16,92,41,44]]
print(mat)
N= 4
def getMaxSum(mat):

    path = [[0 for i in range(N+2)] for j in range(N)]
    
    for i in range(N):
        for j in range(i+1,N+1):
            path[i][j] = 
            
for i in range(4):
    print(i)