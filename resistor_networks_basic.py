#resistornetwork_basic.py
#Forrest Sheldon
#July 23, 2015
#This version of resistor network is meant to avoid the complications of finding percolating clusters etc.

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, cg
import itertools

class ResistorNetwork(object):
    """
    This is a basic class for solving a resistor network.  Initializing the network requires:
    
        G                 - An NxN sparse CSR matrix containing the conductances in the network of N nodes
        external_voltages - An Nx1 dense vector of external voltages.  Nodes not set to an external voltages contain a Nan.
                            The shape (N,) is preferred

    Other available data attributes are:
    
        voltages          - These are the voltages of the internal nodes in the network.  They are initally set to None and
                            are filled in upon calling self.solve()
        nodes             - The number of nodes in the network
        interior          - The index of interior nodes not set to an external voltage
        boundary          - The index of boundary nodes set to an external voltage
    """
    
    def __init__(self, G, external_voltages):
        self.G = G
        self.external_voltages = external_voltages
        self.voltages = None
        self.nodes, tmp = self.G.shape
        self.interior, = np.isnan(self.external_voltages).nonzero()
        self.boundary, = np.logical_not(np.isnan(self.external_voltages)).nonzero()
        
        
    def solve_voltages(self, solver, V_0=None):
        """
        This method solves for the node voltages of the resistor network.  It also assumes the network is well
        defined, i.e. all nodes are part of a single connected component.  If this is not the case, run a
        connected component algorithm on it and feed in the percolating clusters individually. Current solvers are:
        
            'spsolve' - Scipy's exact sparse solver
            'cg'      - Scipy's sparse conjugate gradient solver. Note that conj grad seems to do well for solving a
                        network with separate components without breaking it up first but I'm not sure how safe this
                        is if we begin changing it's initial state.
        
        The second optional argument is
        
            V_0       - (optional) An initial guess for the voltages in the network for the conjugate gradient solver. I
                        think this may be useful for memristor networks where the current and previous voltages are only
                        infinitesimally separated.
        """
        
        # First we form the equations matrix.  To do this, we remove the rows and columns of boundary nodes,
        # trading our Neumann boundary conditions for Dirichlet
        L = self.graph_laplacian()
        D = L[self.interior, :][:, self.interior]
        # The columns corresponding to boundary nodes give a constant vector on the interior nodes yielding
        # the equation Dv = b (the -1 subtracts it to the other side of the equation)
        b = -1. * L[self.interior, :][:, self.boundary].dot(self.external_voltages[self.boundary])
        
        # Put our boundary values in for the voltages
        self.voltages = np.zeros_like(self.external_voltages)
        self.voltages[self.boundary] = self.external_voltages[self.boundary]
        
        # and solve!
        if solver == 'spsolve':
            self.voltages[self.interior] = spsolve(D, b)
        elif solver == 'cg':
            #I'd like to include an optional parameter to give the initial guess for the voltages in the network
            if V_0 == None:
                self.voltages[self.interior], convergence = cg(D, b)
            else:
                self.voltages[self.interior], convergence = cg(D, b, V_0[self.interior])
            #print "Conjugate Gradient converges with %d" % convergence
        else:
            print "Solver not specified.  Try 'spsolve' or 'cg'"
            
    
    def graph_laplacian(self):
        """
        Returns the graph laplacian for the resistor network.  This is L = D - G where D is the 'degree' matrix
        (for us a diagonal matrix of the sum of the incident conductances to each node) and G is the 'adjacency'
        matrix (for us the conductance matrix G)
        """
        # Note that for CSR matrices it is faster to sum across rows
        return sparse.dia_matrix((self.G.sum(1).flat, [0]), shape=(self.nodes,self.nodes)).tocsr() - self.G
    
    def power(self):
        """
        Returns a sparse matrix in CSR form containing the power dissipated between nodes i and j.  Requires that
        self.solve() have been called to populate self.voltages
        """
        # Pull nonzero values to iterate only over occupied bonds
        # as G is symmetric we can take only the upper trianglular part
        rows, cols = sparse.triu(self.G).nonzero()
        
        # Fill in the entries in the power matrix
        power = sparse.lil_matrix(self.G.shape)
        for node_i, node_j in itertools.izip(rows, cols):
            power[node_i, node_j] = self.G[node_i, node_j] * (self.voltages[node_i] - self.voltages[node_j])**2
            power[node_j, node_i] = power[node_i, node_j]
        return power.tocsr()
    
    def voltage_drop(self):
        """
        Return a sparse matrix in CSR form containing the voltage drop between nodes i and j.  Requires that self.solve()
        have been called to populate self.voltages
        """
        rows, cols = sparse.triu(self.G).nonzero()
        
        # fill in the entries in the voltage drop matrix
        voltage_drop = sparse.lil_matrix(self.G.shape)
        for node_i, node_j in itertools.izip(rows, cols):
            voltage_drop[node_i, node_j] = self.voltages[node_j] - self.voltages[node_i]
            voltage_drop[node_j, node_i] = -1 * voltage_drop[node_i, node_j]
        return voltage_drop.tocsr()
    
    def voltage_drop_abs(self):
        """
        Return a sparse matrix in CSR form containing the voltage drop between nodes i and j.  Requires that self.solve()
        have been called to populate self.voltages
        """
        rows, cols = sparse.triu(self.G).nonzero()
        
        # fill in the entries in the voltage drop matrix
        voltage_drop = sparse.lil_matrix(self.G.shape)
        for node_i, node_j in itertools.izip(rows, cols):
            voltage_drop[node_i, node_j] = abs(self.voltages[node_j] - self.voltages[node_i])
            voltage_drop[node_j, node_i] = voltage_drop[node_i, node_j]
        return voltage_drop.tocsr()
        
    def external_current(self):
        """
        Returns the currents entering the nodes on the boundary.  These are calculated from,
        
        graph_laplacian[boundary,:].dot(self.voltages)
        
        and thus occur in the order specified by self.boundary
        """
        return self.graph_laplacian()[self.boundary, :].dot(self.voltages)
    
    def conductivity(self):
        """
        The total conductivity of the network is calculated as the sum of the positive external currents divided
        by the voltage difference across the network.  In order for this to work, the device must be regarded as
        as a two terminal device, ie there is only one high voltage end.
        """
        I_external = self.external_current()
        return I_external[I_external > 0].sum() / (np.nanmax(self.external_voltages) - np.nanmin(self.external_voltages))


