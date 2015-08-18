#fusethresholdnetworks.py
#Forrest Sheldon
#August 17 2015
#==================================================================
#A rendition of fuse_networks in which the thresholds are assigned
#individually.  This is designed to accept a completely connected
#network and will not work with an underconstrained network.
#==================================================================

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, cg
import generate_adjacency_matrix as gen_adj
import resistor_networks_basic as rnets

from __future__ import division

class FuseThresholdNetwork(rnets.ResistorNetwork):
    """
    This class inherits from ResistorNetwork with the aim of adding some dynamical behavior to the
    resistors.  In particular, when a resistor has past a certain current threshold, the conductance
    is switched from G_OFF to G_ON. To initialize, we require:
    
        G                     - An NxN sparse CSR matrix containing the conductances in the network of N nodes
        external_voltages     - An Nx1 dense vector of external voltages.  Nodes not set to an external voltages
                                contain a Nan. The shape (N,) is preferred
        I_threshold           - A sparse array of thresholds for each memristor
        G_OFF                 - The conductance in the low conductivity OFF state.  Generally set to 1.
        G_ON                  - The conductance in the high conductivity ON state.  Set to 100 by default.
    
    Other available attributes after initialization are:
        currents              - An NxN sparse matrix of currents in the network.  As many tasks in the network will
                                require that the currents be calculated, it is best that we do this once and share.
        rows_G
        cols_G
    """
    
    def __init__(self, G, external_voltages, I_threshold, G_OFF=1, G_ON=100):
        rnets.ResistorNetwork.__init__(self, G, external_voltages)
        self.I_threshold = I_threshold
        self.G_OFF = G_OFF
        self.G_ON = G_ON
        self.rows_G, self.cols_G = sparse.triu(self.G).nonzero()
        self.currents = None
            
    def calculate_currents(self):
        """
        calculates a matrix of currents running through each memristor.  Only the upper triangular portion
        of the matrix is calculated.
        """
        self.currents = sparse.lil_matrix(self.G.shape, dtype=float)
        self.currents[self.rows_G, self.cols_G] = np.multiply(self.G[self.rows_G, self.cols_G],
                                                  np.abs(self.voltages[self.rows_G] - self.voltages[self.cols_G]))
        self.currents = self.currents + self.currents.T
    
    def next_switching_factor(self):
        """
        Find the factor necessary to adjust the voltage to switch the next memristor
        """
        rows, cols = sparse.triu(self.G == self.G_OFF).nonzero()
        return np.min(np.divide(self.I_threshold[rows, cols], self.currents[rows, cols]))
        
    def switch_memristors(self):
        """
        Check the currents on each memristor.  Those that are in the low conductance state and that have a current
        above I_thresh will have their conductance switched to G_ON
        """
        switching_memristors = (self.currents > self.I_threshold).multiply(self.G == self.G_OFF).astype('bool')
        self.G[switching_memristors] = self.G_ON
        return switching_memristors.nnz / 2

    def switching_step(self):
        counter = []
        self.solve_voltages('spsolve')
        self.calculate_currents()
        num_switched = self.switch_memristors()
        while num_switched != 0:
            counter.append(num_switched)
            self.solve_voltages('spsolve')
            self.calculate_currents()
            num_switched = self.switch_memristors()
        return counter
    
    def new_voltage(self, prev_voltage):
        return prev_voltage * (self.next_switching_factor() + 1e-10)

    def update_voltage_busbar(self, voltage):
        """
        Using bus bar topology, this takes the previous voltage at the zeroth node and updates it so that the next memristor
        will flip
        """
        self.external_voltages[0:int(np.sqrt(self.G.shape[0]))] = voltage
