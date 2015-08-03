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

#==========================================================
# 2D cubic diagonal periodic
#==========================================================

def cubic_2d_diagonal_periodic(lattice_shape):
    """
    Returns the adjacency matrix for a 2D square lattice in sparse matrix form. The lattice is meant to be
    thought of as a displayed 'diagonally' with the corners of the lattice pointing up and periodic boundary conditions.
    """
    num_ynodes, num_xnodes = lattice_shape
    num_nodes = num_xnodes * num_ynodes
    
    A = sparse.lil_matrix((num_nodes, num_nodes))
    
    # Connect all nodes to the row below them
    for node in range(num_nodes - num_xnodes):
        A[node, node + num_xnodes] = 1
        
        row_index = node / num_xnodes
        row_parity = row_index % 2
        if row_parity == 0:
            if node % num_xnodes == 0:
                A[node, node + 2*num_xnodes - 1] = 1
            else:
                A[node, node + num_xnodes - 1] = 1
        elif row_parity == 1:
            if node % num_xnodes == num_xnodes - 1:
                A[node, node + 1] = 1
            else:
                A[node, node + num_xnodes + 1] = 1
        else:
            print "Seems there's a problem"
        
    A.tocsr()
    
    return A + A.T


#==========================================================
# random_graph
#==========================================================

def random_graph(num_nodes, p, undirected=True):
    """
    Generates an adjacency matrix for a random graph of num_nodes nodes at a concentration p. If undirected=False, bonds
    are considered in both directions (ij and ji) independently.
    """
    # Generate a random array between 0 and 1
    A = np.random.rand(num_nodes, num_nodes)
    # Set all elements less than p to 1 and the rest to 0
    A = np.asarray(A < p, dtype=float)
    # Nodes cannot be connected to themselves
    A[np.diag_indices(num_nodes)] = 0
    # If we don't care about direction, return this
    if undirected==False:
        return sparse.csr_matrix(A)
    else:                                                   # Otherwise, only keep the upper triangle and symmetrize
        A = np.triu(A)
        return sparse.csr_matrix(A + A.T)

#==========================================================
# Nearest Neighbor
#==========================================================

def nearest_neighbor(num_nodes, k, undirected=True):
    """
    Generates an adjacency matrix for a ring of nodes, each connected to their k nearest neighbors.  Note that k must be even.
    If the directed option is selected, each bond has an even chance to point forward or backward.  Takes as arguments:
    
    num_nodes - the number of nodes in the network
    k         - the number of nearest neighbors in the network
    """
    if k % 2 == 1:
        print "k must be even"
        return None
    
    rows = []
    cols = []
    # connect each node to it's k/2 neighbors to the right
    for i in range(num_nodes):
        for j in range(i+1, i+int(k/2)+1):
            rows.append(i)
            cols.append(j % num_nodes)
    
    A = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), shape = (num_nodes, num_nodes))
    
    if undirected == True:
        return A + A.T
    else:
        return undirected2directed(A + A.T)     #works because A is upper triangle of undirected network
                
#==========================================================
# Small World/Watts Strogatz
#==========================================================

def rewire(Adj, p):
    """
    Rewiring takes an existing UNDIRECTED network with Adjacency matrix given by Adj and returns a matrix with the same number of
    bonds but with a scrambled connectivity.  The nodes are iterated through in order.  At each node n_i, all bonds (n_i, n_j)
    with j > i are rewired with probability p.  In rewiring, the bond to n_j is connected to a new node n_k with k selected
    uniformly from the nodes not currently connected to i.
    """
    
    # first pull the existing bonds in the network
    rows, cols = sparse.triu(Adj, k=1).nonzero()
    
    A = Adj.tolil()               # LIL matrices are cheaper to rewire

    # rewire each bond with probability p
    for i, j in zip(rows, cols):
        if np.random.rand() < p:
            # pull list of candidate nodes to be reconnected to
            A[i, i] = 1    # as a placeholder for the moment
            temp, disconnected_nodes = (A[i, :] == 0).nonzero()
            # Draw the new node
            new_node = np.random.choice(disconnected_nodes)
            A[i, i] = 0                   # remove self-link
            A[i, j] = 0                   # remove old link
            A[j, i] = 0
            A[i, new_node] = 1            # replace with new link
            A[new_node, i] = 1
    
    return A.tocsr()

def undirected2directed(Adj):
    """
    Given the adjacency matrix for an undirected network, bonds are given a direction such that if i < j, they point from
    i to j and from j to i with equal probability
    """
    A = Adj.tolil()
    # Pull a list of bonds as the nonzero entries in the upper triangle
    rows, cols = sparse.triu(Adj, k=1).nonzero()
    
    for node_i, node_j in zip(rows, cols):
        if np.random.rand() > 0.5:          # With 0.5 probability, delete the forward bond
            A[node_i, node_j] = 0
        else:                               # otherwise delete the backward bond
            A[node_j, node_i] = 0
    
    return A.tocsr()

def watts_strogatz(num_nodes, neighbors, p, undirected=True):
    """
    Generates the adjacency matrix for a small world network of num_nodes nodes.  It is constructed by rewiring a nearest
    neighbor network with probability p.  If undirected=False, bonds are given an arbitrary direction
    """
    A = rewire(nearest_neighbor(num_nodes, neighbors), p)
    
    if undirected==True:
        return A
    elif undirected==False:
        return undirected2directed(A)

#==========================================================
# Toggle
#==========================================================

def toggle(Adj, p, undirected=True):
    """
    Given an adjacency matrix, this goes through and toggles the state of every potential connection in the network
    with a probability p. If the undirected option is selected, the upper triangle of the matrix is considered and
    symmetrized afterwards. If the network is directed, the network is symmetrized and the upper triangle is again
    considered.  Bonds are placed in the network in forward and backward directions with equal probability.
    """
    # Initialize the new row and column matrices
    toggled_rows = []
    toggled_cols = []
    
    # Loop over all possible bonds
    r, c = np.triu_indices(Adj.shape[0], 1)
    for i, j in zip(r, c):
        if np.random.rand() < p:          # If this succeeds, toggle the state of the bond
            if undirected == True:        # For an undirected network, check the state of the bond and flip it
                if Adj[i, j] == 0:        # If there is no bond there, place one
                    toggled_rows.append(i)
                    toggled_cols.append(j)
            elif undirected == False:
                if Adj[i, j] == 0 and Adj[j, i] == 0:    # If there is no bond there, place one in either direction
                    if np.random.rand() > 0.5:
                        toggled_rows.append(i)
                        toggled_cols.append(j)
                    else:
                        toggled_rows.append(j)
                        toggled_cols.append(i)
                
        else:                             # if not, keep the bond the same
            if undirected == True:
                if Adj[i, j] == 1:
                    toggled_rows.append(i)
                    toggled_cols.append(j)
            elif undirected == False:
                if Adj[i, j] == 1:
                    toggled_rows.append(i)
                    toggled_cols.append(j)
                elif Adj[j, i] == 1:
                    toggled_rows.append(j)
                    toggled_cols.append(i)
    
    A = sparse.csr_matrix((np.ones_like(toggled_rows), (toggled_rows, toggled_cols)), shape = Adj.shape)
    
    if undirected == True:
        return A + A.T
    else:
        return A

#==========================================================
# Locally Connected
#==========================================================

def locally_connected(lattice_shape, distribution):
    """
    Returns an adjacency matrix for a lattice of size lattice_shape whose connectivity is specified by a local distribution.
    Distances between nodes are normalized such that the lattice spacing is 1. The distribution function gives the probability of
    a connection at a given distance.  The networks returned are undirected.
    """
    def node2xy(node_idx):
        """
        returns the x and y coordinates of a node index in our grid supposing that the 0,0 point is in the upper left
        and the positive y-axis points down
        """
        return node_idx % lattice_shape[1], int(node_idx / lattice_shape[1])
    
    def distance(nodei_idx, nodej_idx):
        """
        Returns the distance between nodes i and j assuming a cubic lattice indexed across rows
        """
        x1, y1 = node2xy(nodei_idx)
        x2, y2 = node2xy(nodej_idx)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    num_nodes = lattice_shape[0] * lattice_shape[1]
    
    A = sparse.lil_matrix((num_nodes, num_nodes), dtype='float')
    
    for node_i in range(num_nodes):
        for node_j in range(node_i+1, num_nodes):
            if np.random.rand() < distribution(distance(node_i, node_j)):
                A[node_i, node_j] = 1
    
    return (A + A.T).tocsr()
            
    
