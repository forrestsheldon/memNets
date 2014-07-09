import numpy as np
import scipy as scp
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import scipy.optimize
import matplotlib.pyplot as plt
import random
import itertools

#=======================================================================
# Adjacency matrix Routines
#=======================================================================

def gen_adj_rand(N, p, undirected=True):
    """
    Generates an adjacency matrix for a random graph of N nodes.
    The default behavior is directed. The undirected option generates a
    symmetric graph
    """
    if undirected:
        A = np.random.choice([1, 0], (N, N), p = [p, 1-p])
        Aup = np.triu(A, 1)
        return Aup + Aup.T
    else:
        A = np.random.choice([1, 0], (N, N), p = [p, 1-p])
        A[np.diag_indices(N)] = 0
        return A
    
def gen_adj_rand_sparse(N, p, undirected=True):
    """
    Generates an adjacency matrix for a random graph of N nodes in sparce csc format.
    The default behavior is directed. The undirected option generates a
    symmetric graph
    """
    rows = []
    cols = []
    if undirected:
        row_idxs, col_idxs = np.triu_indices(N, 1)
        for i, j in itertools.izip(row_idxs, col_idxs):
            if random.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        elements = np.ones_like(rows)
        return scipy.sparse.csc_matrix((elements, (rows, cols)), shape=(N,N))
    else:
        for i, j in np.ndindex(N, N):
            if random.random() < p and i != j:
                rows.append(i)
                cols.append(j)
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        elements = np.ones_like(rows)
        return scipy.sparse.csc_matrix((elements, (rows, cols)), shape=(N,N))
    
def gen_adj_neighbor(N, k):
    """
    Generates an adjacency matrix for an N node network in which each node is connected to its
    k nearest neighbors
    """
    row = np.roll(np.hstack((np.ones((1, k+1), dtype=int), np.zeros((1, N-k-1), dtype=int))), -k/2)
    A = np.zeros((N,N), dtype=int)
    for i in np.arange(N):
        A[i, :] = np.roll(row, i)
    A[np.diag_indices(N)] = 0
    return A

def gen_adj_neighbor_sparse(N, k):
    A = gen_adj_neighbor(N, k)
    return scipy.sparse.csc_matrix(A)

def gen_adj_grid(n):
    """
    Generates an adjacency matrix for an nxn grid of nodes that are connected to their nearest neighbors.
    This graph is kept undirected as this does not require choosing a convention for direction.
    """
    A = np.zeros((n**2, n**2))
    # Loop through every pair of nodes
    for i, j in np.ndindex(n,n):
        for k, l in np.ndindex(n,n):
            if (np.abs(i - k) == 1 and np.abs(j - l) == 0) or (np.abs(i - k) == 0 and np.abs(j - l) == 1):
                A[i + n*j, k + n*l] = 1
    return np.array(A, dtype=int)

def gen_adj_rand_grid(n, p):
    """
    For fun, this generates an adjacency matrix for a random 2-D lattice such that neighboring nodes are connected
    with a probability p
    """
    A = np.zeros((n**2, n**2))
    # Loop through every pair of nodes
    for i, j in np.ndindex(n,n):
        for k, l in np.ndindex(n,n):
            if (np.abs(i - k) == 1 and np.abs(j - l) == 0) or (np.abs(i - k) == 0 and np.abs(j - l) == 1):
                if random.random() < p:
                    A[i + n*j, k + n*l] = 1
    return np.array(np.triu(A) + np.triu(A).T, dtype=int)

def gen_adj_grid_sparse(n):
    """
    Generates an adjacency matrix for an nxn grid of nodes that are connected to their nearest neighbors.
    This graph is kept undirected as this does not require choosing a convention for direction.
    The matrix is returned in CSR format
    """
    rows = []
    cols = []
    for i, j in np.ndindex(n, n):
        for k, l in np.ndindex(n, n):
            if (np.abs(i - k) == 1 and np.abs(j - l) == 0) or (np.abs(i - k) == 0 and np.abs(j - l) == 1):
                rows.append(i + n*j)
                cols.append(k + n*l)
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    elements = np.ones_like(rows)
    return scipy.sparse.csr_matrix((elements, (rows, cols)), shape=(n**2, n**2))

def gen_adj_rand_grid_sparse(n, p):
    """
    Generates an adjacency matrix for an nxn random grid of nodes that are connected to their nearest neighbors.
    This graph is kept undirected as this does not require choosing a convention for direction.
    The matrix is returned in CSR format
    """
    rows = []
    cols = []
    # Loop over every pair of elements
    for i, j in np.ndindex(n, n):
        for k, l in np.ndindex(n, n):
            # calculate the row and column index for the pair in the final array.  As the graph is undirected
            # we only need to consider the pair once
            r = i + n*j
            c = k + n*l
            if r < c:
                if (np.abs(i - k) == 1 and np.abs(j - l) == 0) or (np.abs(i - k) == 0 and np.abs(j - l) == 1):
                    if random.random() < p:
                        rows.extend([r, c])
                        cols.extend([c, r])
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    elements = np.ones_like(rows)
    return scipy.sparse.csr_matrix((elements, (rows, cols)), shape=(n**2, n**2))

#=============================================================================
# Dense Solver
#=============================================================================

def external_voltage( num_nodes, ext_voltage_list ):
    """
    Returns an Nx1 vector of external voltages and an external conductance matrix whose value
    is 1/r_tiny given
    
    num_nodes - the number of nodes in the network
    ext_voltage_list - a list of tuples of the form (node_idx, voltage)
    
    There is a concern that if a node is set to zero, it will be indistinguishable from nodes not set to
    zero in the vector.  Nodes set to a specific voltage should be confirmed through the conductance matrix
    for this reason
    """
    
    voltage_vec = np.zeros((num_nodes, 1), dtype=float)
    G_ext = np.zeros((num_nodes, num_nodes), dtype=float)
    r_tiny = 10**-10
    
    for node_idx, V in ext_voltage_list:
        voltage_vec[node_idx, 0] = V
        G_ext[node_idx, node_idx] = 1. / r_tiny
    
    return voltage_vec, G_ext

def form_eqns_matrix( cond_mat ):
    """
    Forms the homogeneous equations matrix by summing along the rows of the conductance matrix, placing
    these values along the diagonal, and subtracting the conductance matrix
    """
    return np.diag( np.sum(np.array(cond_mat, dtype=float), axis=0) ) - cond_mat

def solve_node_voltages( G , ext_voltage_list ):
    """
    Returns a vector of voltages at each node given,
    G - a matrix of conductances where G_ij is the conductance between node i and node j
    ext_voltage_list - list of tuples of the form (node_idx, voltage) given the external voltage set at node_idx
    """
    D = form_eqns_matrix( G )
    num_nodes = G.shape[0]
    v_ext, G_ext = external_voltage( num_nodes, ext_voltage_list)
    return np.linalg.solve( D + G_ext , np.dot( G_ext, v_ext) )


def external_voltage_vec(n, ext_voltage_list):
    """
    Returns a vector whose value is nan for nodes that are unconstrained and v for
    nodes constrained to external voltage v
    """
    v_ext = np.empty(n) * np.nan
    for idx, voltage in ext_voltage_list:
        v_ext[idx] = voltage
    return v_ext

#============================================================================
# Connected Components Solver
#============================================================================

def solve_node_voltages_cc( G_sparse, ext_voltage_list):
    """
    This solver uses the scipy csgraph module to split the graph into connected
    components and then solve each one.  Components not set to any voltage are
    set to zero. Components set to a single voltage are set to that voltage.
    Finally, components set to more than one voltage are solved.  For now, they
    are solved through one of the solvers above. (although I would like to attempt
    a constrained conjugate gradient solution as well)
    """
    # Pull dimensions and set up external voltage and solution vectors
    n = G_sparse.shape[0]
    
    v_external = external_voltage_vec(n, ext_voltage_list)
    voltages = np.zeros_like(v_external)
    
    # Divide the graph into connected components
    num_comp, labels = scipy.sparse.csgraph.connected_components(G_sparse)
    
    # For each connected component
    for cc in range(num_comp):
        # Pull a boolean array of the node indices and use that to get the external voltages
        cc_nodes = (labels == cc)
        cc_v_ext = v_external[cc_nodes]
        
        # Find the maximum and minimum voltages on the component
        cc_max = np.nanmax(cc_v_ext)
        cc_min = np.nanmin(cc_v_ext)
        
        # If the component is not set to any voltage, set it to zero
        if np.isnan(cc_max):
            voltages[cc_nodes] = 0
        
        # If it is set to a single external voltage, set all nodes to that value
        elif cc_max == cc_min:
            voltages[cc_nodes] = cc_max
        
        # Otherwise, it must be set to two external voltages, in which case we must solve it
        else:
            # Use the node mask to get the conductance matrix for this component
            cc_G = np.array(G_sparse[cc_nodes, :][:, cc_nodes].todense())
            # Reconstruc the external voltage list over this component
            v_list = []
            for idx, v in enumerate(cc_v_ext):
                if not (np.isnan(v)):
                    v_list.append((idx, v))
            voltages[cc_nodes] = solve_node_voltages( cc_G, v_list)
    return voltages

#===========================================================================
# Displaying Networks
#===========================================================================

def display_grid_voltages(ax, G, voltages, nodesize=12, bondwidth=3):
    """
    Plots the voltages from the solution of resistors on a grid.  The positions of the nodes
    are pulled from the conductance matrix with the convention that the 0th node is at the top
    left and the index increases as we move down the columns of the grid. Arguments are:
    
    ax - an axes object for the figure
    G - a dense conductance array.  Note that this is, at the moment incompatible with the
    matrix type due to the different behavior of nonzero()
    voltages - a vector of the node voltages
    """
    # make sure we have no extraneous dimensions and
    # find the dimensions of the grid
    voltages = voltages.flatten()
    n = np.sqrt(voltages.size)
    
    def node2xy(num_nodes, node_idx):
        """
        returns the x and y coordinates of a node index in our grid
        """
        return np.floor(node_idx / num_nodes), (num_nodes-1) - node_idx % n


    # First draw the connections between nodes
    rows, cols = np.triu(G).nonzero()
    
    for node_i, node_j in itertools.izip(rows, cols):
        x_i, y_i = node2xy(n, node_i)
        x_j, y_j = node2xy(n, node_j)
        ax.plot([x_i, x_j], [y_i, y_j], 'k', lw = bondwidth)
        
    # Normalize our voltage colormap to the max and min of voltages
    reds = plt.get_cmap("Reds")
    norm = plt.Normalize()
    norm.autoscale(voltages)
    
    # Now draw the nodes and their voltages
    for node, volt in enumerate(voltages):
        x, y = node2xy(n, node)
        ax.plot(x, y, 's', markersize=nodesize,
                   color=reds(norm(volt)))
    ax.set_xlim( -1, n )
    ax.set_ylim( -1, n )
    
def display_grid_power(ax, G, voltages, nodesize=5, bondwidth=3):
    """
    Plots the power dissipated in each leg of the grid.  The positions of the nodes
    are pulled from the conductance matrix with the convention that the 0th node is at the top
    left and the index increases as we move down the columns of the grid. Arguments are:
    
    ax - an axes object for the figure
    G - a dense conductance array
    voltages - a vector of the node voltages
    """
    # make sure we have no extraneous dimensions and
    # find the dimensions of the grid
    voltages = voltages.flatten()
    n = np.sqrt(voltages.size)
    
    def node2xy(num_nodes, node_idx):
        """
        returns the x and y coordinates of a node index in our grid
        """
        return np.floor(node_idx / num_nodes), (num_nodes-1) - node_idx % n
    
    # Form a matrix of power dissipation in each resistor
    Power = np.zeros_like(G, dtype=float)
    
    rows, cols = np.triu(G).nonzero()
    for node_i, node_j in itertools.izip(rows, cols):
        Power[node_i, node_j] = G[node_i, node_j] * (voltages[node_i] - voltages[node_j])**2
    
    colormap = plt.get_cmap("YlOrRd")
    norm_pow = plt.Normalize()
    norm_pow.autoscale(Power)
    
    # draw the connections between nodes
    for node_i, node_j in itertools.izip(rows, cols):
        x_i, y_i = node2xy(n, node_i)
        x_j, y_j = node2xy(n, node_j)
        ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm_pow(Power[node_i, node_j])), lw=bondwidth)
        
    # Normalize our voltage colormap to the max and min of voltages
    #norm_v = plt.normalize()
    #norm_v.autoscale(voltages)
    
    # Now draw the nodes and their voltages
    for node, volt in enumerate(voltages):
        x, y = node2xy(n, node)
        ax.plot(x, y, 'ws', markersize=nodesize)
         #       color=reds(norm_v(volt)))
    ax.set_xlim( -1, n )
    ax.set_ylim( -1, n )
    return Power
