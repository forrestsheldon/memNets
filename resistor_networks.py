# resistor_network.py
# By Forrest Sheldon
# This is a transcription of the classes I have created to solve
# and display resistor networks from my notebook
# Resistor_Networks_V3.ipynb

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
    
        G                     - An NxN sparse CSR matrix containing the conductances in the network of N nodes
        external_voltages     - An Nx1 dense vector of external voltages.  Nodes not set to an external voltages
                                contain a Nan. The shape (N,) is preferred

    Other available data attributes are:
    
        voltages              - These are the voltages of the internal nodes in the network.  They are initally set
                                to None and are filled in upon calling self.solve()
        nodes                 - The number of nodes in the network
        num_comp              - The number of connected components in the graph
        comp_labels           - An array containing the connected component label for each node
        interior              - A boolean array of the interior nodes of the network
        boundary              - A boolean array of the boundary nodes set to an external voltage
        percolating_nodes     - A boolean array of the nodes in the percolating cluster
        non_percolating_nodes - An array containing the labels of the connected components that do not percolate
        interior_percolating  - A boolean array of the interior nodes of the percolating cluster
        boundary_percolating  - A boolean array of the boundary nodes of the percolating cluster
    """
    
    def __init__(self, G, external_voltages):
        self.G = G
        self.external_voltages = external_voltages

        self.voltages = None
        self.nodes, tmp = self.G.shape
        self.num_comp, self.comp_labels = connected_components(self.G, directed=False)
        self.interior = np.isnan(self.external_voltages)
        self.boundary = np.logical_not(np.isnan(self.external_voltages))
        
        self.percolating_nodes = None
        self.non_percolating_comp = None
        self.find_percolating_nodes()
        self.interior_percolating = np.logical_and(self.interior, self.percolating_nodes)
        if not np.any(self.interior_percolating):
            print "No interior nodes in percolating cluster"
        self.boundary_percolating = np.logical_and(self.boundary, self.percolating_nodes)
        
        
    
    def find_percolating_nodes(self):
        """
        This method creates an array of indices of percolating nodes in the network
        """
        
        self.percolating_nodes = np.zeros_like(self.external_voltages, dtype='bool')
        self.non_percolating_comp = []
        
        #Loop over connected components in graph
        for cc in range(self.num_comp):
            
            # Pull a boolean array of the node indices and use that to get the external voltages
            cc_nodes = (self.comp_labels == cc)
            cc_external_voltages = self.external_voltages[cc_nodes]
            
            # Find the maximum and minimum voltages on the component.  Nan's will only be found if no other number
            # is in the array
            cc_max = np.nanmax(cc_external_voltages)
            cc_min = np.nanmin(cc_external_voltages)
            
            # If the component is set to at least one voltage and it does not equal some other, it percolates
            if np.isnan(cc_max) or cc_max == cc_min:
                self.non_percolating_comp.append(cc)
            else:
                # Add the connected component to the percolating nodes
                self.percolating_nodes[cc_nodes] = True
            
        # If no nodes percolate, give a message
        if not np.any(self.percolating_nodes):
            print "This graph does not percolate"
            
                
            
    def solve_voltages_percolating(self, solver, V_0=None):
        """
        This method solves for the node voltages of the percolating cluster. Current solvers are:
        
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
        D = L[self.interior_percolating, :][:, self.interior_percolating]
        # The columns corresponding to boundary nodes give a constant vector on the interior nodes yielding
        # the equation Dv = b (the -1 subtracts it to the other side of the equation)
        b = -1. * L[self.interior_percolating, :][:, self.boundary_percolating].dot(self.external_voltages[self.boundary_percolating])
        
        # Put our boundary values in for the voltages
        self.voltages = np.zeros_like(self.external_voltages)
        self.voltages[self.boundary] = self.external_voltages[self.boundary]
        
        # and solve!
        if solver == 'spsolve':
            self.voltages[self.interior_percolating] = spsolve(D, b)
        elif solver == 'cg':
            #I'd like to include an optional parameter to give the initial guess for the voltages in the network
            if V_0 == None:
                self.voltages[self.interior_percolating], convergence = cg(D, b)
            else:
                self.voltages[self.interior_percolating], convergence = cg(D, b, V_0[self.interior_percolating])
            #print "Conjugate Gradient converges with %d" % convergence
        else:
            print "Solver not specified.  Try 'spsolve' or 'cg'"
            
    def fill_nonpercolating_voltages(self):
        """
        Uses the non-percolating components to fill in the appropriate voltages for elements of the network not solved for
        """
        
        #for each nonpercolating component
        for cc in self.non_percolating_comp:
            
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
    
    def solve_voltages(self, solver, V_0=None):
        """
        To solve the network, solve the percolating cluster and fill in the non_percolating components
        """
        
        self.solve_voltages_percolating(solver, V_0)
        self.fill_nonpercolating_voltages()
    
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
        by the voltage difference across the network.  In order for this to work, 
        """
        I_external = self.external_current()
        return I_external[I_external > 0].sum() / (np.nanmax(self.external_voltages) - np.nanmin(self.external_voltages))
    
    def display_ring(self, ax, display_variable, nodesize=5, bondwidth=2, colormin=None, colormax=None):
        """
        This method displays a  resistor network of N nodes on the unit circle with resistors displayed as bonds between the
        nodes.  Indexing begins with the node at the 3 o'clock position and procedes counter clockwise around the circle.
        The variables that may be displayed are:
        
            'voltage'
            'power'
            'conductance'
            'voltage_drop'
            'log_voltage_drop'
            
        """ 
        
        delta_theta = 2 * np.pi / self.nodes
        
        def node2xy_circle(node_idx):
            """
            returns the x and y coordinates of a node index on the unit circle assuming that the 0 node is 
            """
            complex_coord = np.exp(node_idx * delta_theta * 1j)
            return complex_coord.real, complex_coord.imag
        
        # Pull nonzero values to plot bonds
        rows, cols = sparse.triu(self.G).nonzero()
        
        
        # Set up colormap normalization
        
        if colormin != None:
            norm = plt.Normalize(vmin=colormin, vmax=colormax)
        elif display_variable == 'voltage':
            norm = plt.Normalize()
            norm.autoscale(self.voltages)
        elif display_variable == 'power':
            power = self.power()
            norm = plt.Normalize(vmin=power.min(), vmax=power.max())
        elif display_variable == 'conductance':
            conductances = self.G[rows, cols]
            # I'd like the OFF grey to be lighter than the minimum of hte color map
            # so I'm setting it so that it falls 1/3 through the colormap
            mincond = conductances.min()
            maxcond = conductances.max()
            low_colormap = maxcond - 1.5 * (maxcond-mincond)
            norm = plt.Normalize(vmin=low_colormap, vmax=maxcond)
        elif display_variable == 'voltage_drop':
            voltage_drop = self.voltage_drop_abs()
            norm = plt.Normalize(vmin=0, vmax=voltage_drop.max())
        elif display_variable == 'log_voltage_drop':
            voltage_drop = self.voltage_drop_abs()
            norm = plt.Normalize(vmin=np.log(voltage_drop.data.min()),
                                 vmax=np.log(voltage_drop.max()))
        
        if display_variable == 'voltage':
            colormap = plt.get_cmap('Reds')
        elif display_variable == 'power':
            colormap = plt.get_cmap('YlOrRd')
        elif display_variable == 'conductance':
            colormap = plt.get_cmap('RdGy_r')
        elif display_variable == 'voltage_drop':
            colormap = plt.get_cmap('jet')
        elif display_variable == 'log_voltage_drop':
            colormap = plt.get_cmap('jet')
        else:
            print 'Invalid display variable %s' % display_variable
        
        
            
        # Draw the bonds between nodes
        for node_i, node_j in itertools.izip(rows, cols):
            x_i, y_i = node2xy_circle(node_i)
            x_j, y_j = node2xy_circle(node_j)
            if display_variable == 'voltage':
                ax.plot([x_i, x_j], [y_i, y_j], 'k', lw = bondwidth)
            elif display_variable == 'power':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(power[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'conductance':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(self.G[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'voltage_drop':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(voltage_drop[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'log_voltage_drop':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(np.log(voltage_drop[node_i, node_j]))), lw=bondwidth)
        
        # Now draw the nodes
        if display_variable == 'voltage':
            for node, volt in enumerate(self.voltages):
                x, y = node2xy_circle(node)
                ax.plot(x, y, 'o', markersize=nodesize, color=colormap(norm(volt)))
        elif display_variable == 'power' or 'conductance' or 'voltage_drop' or 'log_voltage_drop':
            for node in range(self.nodes):
                x, y = node2xy_circle(node)
                ax.plot(x, y, 'wo', markersize=nodesize)
        
        # And finally set the axes to be just outside the grid spacing and invert the y_axis
        ax.set_xlim( -1.1, 1.1)
        ax.set_ylim( -1.1, 1.1)
        
    def display_grid(self, ax, lattice_shape, display_variable, nodesize=5, bondwidth=3, colormin=None, colormax=None,
                     colormap_name=None):
        """
        This method displays a 2D cubic resistor lattice of shape lattice_shape = (y, x).  The variables
        that may be displayed are:
        
            'voltage'
            'power'
            'conductance'
            'log-power'
            
        Nodes are indexed across rows such that the first row has nodes 0 through x-1.  This is because I typically
        like to set up networks with a vertical bus bar architecture and it makes setting the nodes as simple as possible.
        """ 
        def node2xy(node_idx):
            """
            returns the x and y coordinates of a node index in our grid supposing that the 0,0 point is in the upper left
            and the positive y-axis points down
            """
            return node_idx % lattice_shape[1], int(node_idx / lattice_shape[1]) 
        
        # Pull nonzero values to plot bonds
        rows, cols = sparse.triu(self.G).nonzero()
        
        
        # Set up colormap normalization
        
        if colormin != None:
            norm = plt.Normalize(vmin=colormin, vmax=colormax)
        elif display_variable == 'voltage':
            norm = plt.Normalize()
            norm.autoscale(self.voltages)
        elif display_variable == 'power':
            power = self.power()
            norm = plt.Normalize(vmin=power.min(), vmax=power.max())
        elif display_variable == 'conductance':
            conductances = self.G[rows, cols]
            # I'd like the OFF grey to be lighter than the minimum of hte color map
            # so I'm setting it so that it falls 1/3 through the colormap
            mincond = conductances.min()
            maxcond = conductances.max()
            low_colormap = maxcond - 1.5 * (maxcond-mincond)
            norm = plt.Normalize(vmin=low_colormap, vmax=maxcond)
        elif display_variable == 'voltage_drop':
            voltage_drop = self.voltage_drop_abs()
            norm = plt.Normalize(vmin=0, vmax=voltage_drop.max())
        elif display_variable == 'log_voltage_drop':
            voltage_drop = self.voltage_drop_abs()
            norm = plt.Normalize(vmin=np.log(voltage_drop.data.min()),
                                 vmax=np.log(voltage_drop.max()))
        
        if colormap_name != None:
            colormap = plt.get_cmap(colormap_name)
        else:
            if display_variable == 'voltage':
                colormap = plt.get_cmap('Reds')
            elif display_variable == 'power':
                colormap = plt.get_cmap('YlOrRd')
            elif display_variable == 'conductance':
                colormap = plt.get_cmap('RdGy_r')
            elif display_variable == 'voltage_drop':
                colormap = plt.get_cmap('jet')
            elif display_variable == 'log_voltage_drop':
                colormap = plt.get_cmap('jet')
        
        
            
        # Draw the bonds between nodes
        for node_i, node_j in itertools.izip(rows, cols):
            x_i, y_i = node2xy(node_i)
            x_j, y_j = node2xy(node_j)
            if display_variable == 'voltage':
                ax.plot([x_i, x_j], [y_i, y_j], 'k', lw = bondwidth)
            elif display_variable == 'power':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(power[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'conductance':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(self.G[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'voltage_drop':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(voltage_drop[node_i, node_j])), lw=bondwidth)
            elif display_variable == 'log_voltage_drop':
                ax.plot([x_i, x_j], [y_i, y_j], color=colormap(norm(np.log(voltage_drop[node_i, node_j]))), lw=bondwidth)
        
        # Now draw the nodes
        if display_variable == 'voltage':
            for node, volt in enumerate(self.voltages):
                x, y = node2xy(node)
                ax.plot(x, y, 's', markersize=nodesize, color=colormap(norm(volt)))
        elif display_variable == 'power' or 'conductance' or 'voltage_drop' or 'log_voltage_drop':
            for node in range(self.nodes):
                x, y = node2xy(node)
                ax.plot(x, y, 'ws', markersize=nodesize)
        
        # And finally set the axes to be just outside the grid spacing and invert the y_axis
        ax.set_xlim( -1, lattice_shape[1])
        ax.set_ylim( -1, lattice_shape[0])
        ax.invert_yaxis()
        ax.xaxis.set_tick_params(labelbottom='off', labeltop='on')
