# generate_adjacency_matrix.py
# By Forrest Sheldon
# This is a copy of the functions I have written for generating
# adjacency matrices from my ipython notebook
# Generating_Adjacency_Matrices.py
# At the moment, development is a bit sparse and this is intended as
# a convenience

import numpy as np
import scipy as sp
import scipy.sparse as sparse


#==========================================================
# 1D Chain
#==========================================================

def chain_1d(N, directed=False, bias=0.5):
    """
    Returns the adjacency matrix for a 1D chain in CSR format.  The
    default behavior returns an undirected network.
    The bias gives the probability that a bond goes from i to i+1
    versus i+1 to i
    """
    A = sparse.lil_matrix((N, N))
    for node in range(N-1):
        if directed:
            if np.random.rand() > bias:
                A[node+1, node] = 1.
            else:
                A[node, node+1] = 1.
        else:
            A[node, node+1] = 1.
    return A.tocsr()

#==========================================================
# 2D Cubic
#==========================================================

def cubic_2d(lattice_shape, undirected=True, xbias=1, ybias=1 ):
    """
    Returns an adjacency matrix for a 2D cubic lattice with number of nodes specified by
    lattice_shape.  If a directed network is requested with no bias, the default configuration is
    all bonds going from left to right and top to bottom. (recalling that we index nodes across
    rows then columns).  The xbias and ybias give the probability that a bond goes from left to
    right versus RL and top to bottom versus BT respectively.
    """
    num_ynodes, num_xnodes = lattice_shape
    num_nodes = num_xnodes * num_ynodes
    
    A = sparse.lil_matrix((num_nodes, num_nodes))
    
    # Form bond arrays to fill in row bonds and column bonds of the lattice
    x_bonds = np.ones(num_xnodes-1)
    y_bonds = np.ones(num_ynodes-1)
    
    # connect each row node to its neighbor to the right
    for first_row_node in range(0, num_nodes, num_xnodes):
         A[range(first_row_node, first_row_node + num_xnodes - 1),\
          range(first_row_node + 1, first_row_node + num_xnodes)] = x_bonds
    
    # connect each column node to its neighbor below
    for first_col_node in range(0, num_xnodes):
         A[range(first_col_node, num_nodes - num_xnodes, num_xnodes),\
          range(first_col_node + num_xnodes, num_nodes, num_xnodes)] = y_bonds
    
    # If we want an undirected network, just return the symmetrized form
    if undirected:
        A = A.tocsr()
        return A + A.T
    else:
        # If we want to toggle the direction of the elements (default direction is right and down)
        if (xbias != 1) or (ybias != 1):
            rows, cols = A.nonzero()
        
            for i, j in zip(rows, cols):
                if np.abs(i-j) == 1: # row bond
                    if np.random.rand() > xbias: # overcome the bias with probability 1-xbias
                        A[i, j] = 0
                        A[j, i] = 1
                else: #column bond
                    if np.random.rand() > ybias:
                        A[i, j] = 0
                        A[j, i] = 1
        return A.tocsr()

#==========================================================
# 2D Cubic Random
#==========================================================

def cubic_2d_random(lattice_shape, concentration, undirected=True, single_bond=False, xbias=1, ybias=1):
    """
    Returns a random 2d lattice with specified concentration in CSR format.  Besides an undirected
    network, we may also generate random directed networks of a specified concentration. The
    single_bond variable specified whether we may have bonds in both directions or only in one
    at a time. The xbias and ybias give the probability that a bond goes from left to
    right versus RL and top to bottom versus BT respectively.
    """
    # for an undirected network, we begin with a directed network, choose which bonds to keep and then symmetrize
    # Changing the sparsity structure of LIL matrices is faster
    if undirected:
        A = cubic_2d(lattice_shape, undirected=False).tolil()
    # if we want a multiple bond network, we begin with a full undirected network
    elif not single_bond:
        A = cubic_2d(lattice_shape).tolil()
    # for a single bond network, we begin with the directed network and then prune
    elif single_bond:
        A = cubic_2d(lattice_shape, undirected=False, xbias=xbias, ybias=ybias).tolil()
    else:
        print "Invalid parameters defining lattice.  Check undirected and single_bond"
    
    # Get nonzero indices
    rows, cols = A.nonzero()
    # Loop over nonzero elements
    for i, j in zip(rows, cols):
        if np.random.rand() > concentration:   # Delete the bond with probability 1-concentration
            A[i, j] = 0
    
    A = A.tocsr()
    if undirected: # symmetrize before returning
        return A + A.T
    else:
        return A

