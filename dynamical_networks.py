# dynamical_nets.py
# By Forrest Sheldon
# This is a copy of the class I have defined in
# the dynamical_nets notebook.  It is copied here as a convenient way
# to import it into new notebooks.

import numpy as np
import scipy as sp
import resistor_networks as rnets


#==========================================================
# Dynamical Network
#==========================================================

class DynamicalNetwork(rnets.ResistorNetworkCC):
    """
    This class inherits from the resistor network class in the aim of expanding it to accomodate dynamic behavior in the
    resistances.  As such, it requires an update rule that alters the state of each resistor based on the current state
    of the system.
    
    """
    
    def __init__(self, state_variables, external_voltages, XtoG, update_rule, threshold_voltages=None, G_ON=1, G_OFF=1e-2):
        self.state_variables = state_variables
        self.rows, self.cols = state_variables.nonzero()
        self.external_voltages = external_voltages
        self.XtoG = XtoG
        self.G_ON = G_ON
        self.G_OFF = G_OFF
        self.update_rule = update_rule
        self.threshold_voltages = threshold_voltages
        rnets.ResistorNetworkCC.__init__(self, self.XtoG(self.state_variables), self.external_voltages)
        
    def update_conductance_matrix(self):
        self.G = self.XtoG(self.state_variables)
        
    def update_state_variables(self):
        
        # loop over each memristor. This could easily be done in parallel
        for i, j in zip(self.rows, self.cols):
            # Set the parameters of each memristor in the integrator
            voltage_drop = self.voltages[i] - self.voltages[j]
            
            # See if threshold voltages have been defined
            try:
                V_thresh = self.threshold_voltages[i, j]
            except TypeError:
                V_thresh = 1
            
            integrated_state_variable = self.update_rule(self.state_variables[i, j], voltage_drop, V_thresh, self.G_ON, self.G_OFF)
            
            # Check values and set to limits if we have strayed beyond them
            if integrated_state_variable >= self.G_ON:
                self.state_variables[i, j] = self.G_ON
            elif integrated_state_variable <= self.G_OFF:
                self.state_variables[i, j] = self.G_OFF
            else:
                self.state_variables[i, j] = integrated_state_variable
                
            self.update_conductance_matrix()

        
