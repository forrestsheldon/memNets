# resistor_network.py
# By Forrest Sheldon
# This is a transcription of the classes I have created to solve
# and display resistor networks from my notebook
# Resistor_Networks_V2.ipynb

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import itertools

#================================================================
# ResistorNetwork
#================================================================

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
    
    def external_current(self):
        """
        Returns the currents entering the nodes on the boundary.  These are calculated from,
        
        graph_laplacian[boundary,:].dot(self.voltages)
        
        and thus occur in the order specified by boundary
        """
        return self.graph_laplacian()[self.boundary, :].dot(self.voltages)
    
    def conductivity(self):
        """
        The total conductivity of the network is calculated as the sum of the positive external currents divided
        by the voltage difference across the network.  In order for this to work, 
        """
        I_external = self.external_current()
        return I_external[I_external > 0].sum() / (np.nanmax(self.external_voltages) - np.nanmin(self.external_voltages))

#================================================================
# ResistorNetworkCC
#================================================================

class ResistorNetworkCC(object):
    """
    This class solves resistor network by first breaking the network into connected components and then solving each component
    separately.  Aside from this fact, it works by using composition with resistor_network.  Initializing a network requires:
    
        G                 - An NxN sparse CSR matrix containing the conductances in the network of N nodes
        external_voltages - An Nx1 dense vector of external voltages.  Nodes not set to an external voltages contain a Nan.
                            The shape (N,) is preferred

    Other available data attributes are:
    
        voltages          - These are the voltages of the internal nodes in the network.  They are initally set to None and
                            are filled in upon calling self.solve()
        nodes             - The number of nodes in the network
        num_comp          - The number of components in the undirected network as given by the connected_components function
                            in scipy.sparse.csgraph
        comp_labels       - An array containing the component of each node in the network from 
        interior          - The index of interior nodes not set to an external voltage
        boundary          - The index of boundary nodes set to an external voltage
    """
    
    def __init__(self, G, external_voltages):
        self.G = G
        self.external_voltages = external_voltages
        self.voltages = None
        self.nodes, tmp = self.G.shape
        self.num_comp, self.comp_labels = connected_components(self.G, directed=False)
        self.interior, = np.isnan(self.external_voltages).nonzero()
        self.boundary, = np.logical_not(np.isnan(self.external_voltages)).nonzero()
        
    def solve_voltages(self, solver, V_0=None):
        """
        This method solves for the node voltages of the resistor network by first breaking the network into connected
        components and then defining a resistor_network object for each percolating component. Current solvers are:
        
            'spsolve' - Scipy's exact sparse solver
            'cg'      - Scipy's sparse conjugate gradient solver.
        
        The second optional argument is
        
            V_0       - (optional) An initial guess for the voltages in the network for the conjugate gradient solver. I
                        think this may be useful for memristor networks where the current and previous voltages are only
                        infinitesimally separated.
        """
        
        self.voltages = np.zeros_like(self.external_voltages)
        
        # For each connected component
        for cc in range(self.num_comp):
            
            # Pull a boolean array of the node indices and use that to get the external voltages
            cc_nodes = (self.comp_labels == cc)
            cc_external_voltages = self.external_voltages[cc_nodes]
        
            # Find the maximum and minimum voltages on the component.  Nan's will only be found if no other number
            # is in the array
            cc_max = np.nanmax(cc_external_voltages)
            cc_min = np.nanmin(cc_external_voltages)
        
            # If the component is not set to any voltage, set it to zero
            if np.isnan(cc_max):
                self.voltages[cc_nodes] = 0
        
            # If it is set to a single external voltage, set all nodes to that value
            elif cc_max == cc_min:
                self.voltages[cc_nodes] = cc_max
        
            # Otherwise, it must be set to two external voltages, in which case we must solve it
            else:
                # Solve a percolating component over the limited conductance matrix and external voltage
                CC_RNet = ResistorNetwork(self.G[cc_nodes, :][:, cc_nodes], cc_external_voltages)
                if V_0 == None:
                    CC_RNet.solve_voltages(solver)
                else:
                    CC_RNet.solve_voltages(solver, V_0[cc_nodes])
                self.voltages[cc_nodes] = CC_RNet.voltages
    
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
    
    def external_current(self):
        """
        Returns the currents entering the nodes on the boundary.  These are calculated from,
        
        graph_laplacian[boundary,:].dot(self.voltages)
        
        and thus occur in the order specified by boundary
        """
        return self.graph_laplacian()[self.boundary, :].dot(self.voltages)
    
    def conductivity(self):
        """
        The total conductivity of the network is calculated as the sum of the positive external currents divided
        by the voltage difference across the network.  In order for this to work, 
        """
        I_external = self.external_current()
        return I_external[I_external > 0].sum() / (np.nanmax(self.external_voltages) - np.nanmin(self.external_voltages))
    


#================================================================
# ResistorLattice2DCubic
#================================================================

class ResistorLattice2DCubic(ResistorNetworkCC):
    
    def __init__(self, G, external_voltage, lattice_shape):
        ResistorNetworkCC.__init__(self, G, external_voltage)
        self.lattice_shape = lattice_shape
        if self.nodes != (self.lattice_shape[0] * self.lattice_shape[1]):
            print "Number of nodes is not consistent with the shape of the lattice!"
    
    def display(self, ax, display_variable, nodesize=5, bondwidth=3):
        """
        This method displays a 2D cubic resistor lattice of shape self.shape = (y, x).  The variables that may be displayed are:
        
            voltage
            power
            resistance
            
        Nodes are indexed across rows such that the first row has nodes 0 through x-1.  This is because I typically
        like to set up networks with a vertical bus bar architecture and it makes setting the nodes as simple as possible.
        """ 
        def node2xy(node_idx):
            """
            returns the x and y coordinates of a node index in our grid supposing that the 0,0 point is in the upper left
            and the positive y-axis points down
            """
            return node_idx % self.lattice_shape[1], int(node_idx / self.lattice_shape[1]) 
        
        # Pull nonzero values to plot bonds
        rows, cols = sparse.triu(self.G).nonzero()
        
        # Set up colors
        if display_variable == 'voltage':
            
            # Normalize our voltage colormap to the max and min of voltages
            reds = plt.get_cmap('Reds')
            norm = plt.Normalize()
            norm.autoscale(self.voltages)
        
        elif display_variable == 'power':
            
            # Calculate the power dissipated in each resistor
            power = self.power()
            # Normalize the colormap to the minimum and maximum power in the network
            colormap = plt.get_cmap('YlOrRd')
            norm = plt.Normalize(vmin=power.min(), vmax=power.max())
        else:
            print 'Invalid display variable %s' % display_variable
            
        # Draw the bonds between nodes
        for node_i, node_j in itertools.izip(rows, cols):
            x_i, y_i = node2xy(node_i)
            x_j, y_j = node2xy(node_j)
            if display_variable == 'voltage':
                ax.plot([x_i, x_j], [y_i, y_j], 'k', lw = bondwidth)
            elif display_variable == 'power':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(power[node_i, node_j])), lw=bondwidth)
        
        # Now draw the nodes
        if display_variable == 'voltage':
            for node, volt in enumerate(self.voltages):
                x, y = node2xy(node)
                ax.plot(x, y, 's', markersize=nodesize, color=reds(norm(volt)))
        elif display_variable == 'power':
            for node in range(self.nodes):
                x, y = node2xy(node)
                ax.plot(x, y, 'ws', markersize=nodesize)
        
        # And finally set the axes to be just outside the grid spacing and invert the y_axis
        ax.set_xlim( -1,  self.lattice_shape[1])
        ax.set_ylim( -1, self.lattice_shape[0])
        ax.invert_yaxis()
        ax.xaxis.set_tick_params(labelbottom='off', labeltop='on')

