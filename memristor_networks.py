# memristor_network.py
# By Forrest Sheldon
# This is a transcription of the classes I have created to solve
# and display resistor networks from my notebook
# Memristorr_Nets_V2.ipynb

import numpy as np
import scipy as sp
from scipy.integrate import ode
import resistor_network as rnets

#================================================================
# MemristorNetwork
#================================================================

class MemristorNetwork(rnets.ResistorNetworkCC):
    """
    This class inherits from my ResistorNetworkCC class in the aim of generalizing it to solve arbitrary
    memristor networks.  This class will store the memristor network at a moment in time and possesses an
    update(dt) method that integrates the network forward a time dt, assuming that the voltages in the
    network are constant. This requires:
    
    state_variables
    external_voltages
    XtoG
    xdot
    
    with optional arguments:
    
    jacobian
    memristor_params
    G_ON
    G_OFF
    """
    
    def __init__(self, state_variables, external_voltages, XtoG, xdot, jacobian=None, threshold_voltages=None,
                 G_ON=1, G_OFF=10**-2):
        self.state_variables = state_variables
        self.rows, self.cols = self.state_variables.nonzero()
        self.XtoG = XtoG
        self.G_ON = G_ON
        self.G_OFF = G_OFF
        self.threshold_voltages = threshold_voltages
        self.xdot=xdot
        self.integrator = ode(xdot, jac=jacobian)
        rnets.ResistorNetworkCC.__init__(self, self.XtoG(self.state_variables), external_voltages)
    
    def update_conductance_matrix(self):
        self.G = self.XtoG(self.state_variables)
    
    def integrate_state_variables(self, delta_t, method='scipyode'):
        """
        Integrate the state variables of the network from their current value to their value a time delta_t
        later assuming that the voltages are constant during this time.  This method is only intended to update
        the state variables infinitesimally.  The attribute xdot is a function of the state variables and the
        voltage drop across a memristor that gives the first derivative of the state variables. 
        """
    
        # loop over each memristor. This could easily be done in parallel
        for i, j in zip(self.rows, self.cols):
            # Set the parameters of each memristor in the integrator
            voltage_drop = self.voltages[i] - self.voltages[j]
            
            # See if threshold voltages have been defined
            try:
                V_thresh = self.threshold_voltages[i, j]
            except TypeError:
                V_thresh = 1
                
            if method == 'scipyode':
                self.integrator.set_initial_value(self.state_variables[i, j], 0).set_f_params(voltage_drop,
                                                  V_thresh).set_jac_params(voltage_drop, V_thresh)
                integrated_state_variable = self.integrator.integrate(delta_t)
            elif method == 'myrk4':
                integrated_state_variable = RK4_next(self.state_variables[i, j], voltage_drop, V_thresh,
                                                     0, delta_t, self.xdot)
            
            # Check values and set to limits if we have strayed beyond them
            if integrated_state_variable >= self.G_ON:
                self.state_variables[i, j] = self.G_ON
            elif integrated_state_variable <= self.G_OFF:
                self.state_variables[i, j] = self.G_OFF
            else:
                self.state_variables[i, j] = integrated_state_variable
            
            if not self.integrator.successful():   
                print "Integration was unsuccessful"
                
            
        self.update_conductance_matrix()

        
# I am making use of my RK4 integrator to provide a faster integration option.  The
# integrators provided by ode will offer a way to assess whether results given here
# are trustworthy

def RK4_next(current_x, V, V_T, time, delta_t, func_xdot):
    
    def xdot(x, t):
        return func_xdot(t, x, V, V_T)
    
    return current_x + 1. / 6. * delta_t * (k1(xdot, current_x, time) + 2 * k2(xdot, current_x, time, delta_t) +
                                            2 * k3(xdot, current_x, time, delta_t) + k4(xdot, current_x, time, delta_t))
    

    # Various intermediate derivatives used in RK4       
def k1(deriv, y, t):
    return deriv(y, t)

def k2(deriv, y, t, h):
    return deriv(y + 0.5 * h * k1(deriv, y, t), t + 0.5 * h)

def k3(deriv, y, t, h):
    return deriv(y + 0.5 * h * k2(deriv, y, t, h), t + 0.5 * h)

def k4(deriv, y, t, h):
    return deriv(y + h * k3(deriv, y, t, h), t + h)

#================================================================
# MemristorLattice2DCubic
#================================================================

class MemristorLattice2DCubic(rnets.ResistorLattice2DCubic):
    """
    This class inherits from my ResistorLattice2DCubic class in the aim of generalizing it to solve arbitrary
    memristor lattices.  It is the same code as the MemristorNetwork class, only inheriting the display
    methods from the resistor lattice class as well.This class will store the memristor network at a moment
    in time and possesses an update(dt) method that integrates the network forward a time dt, assuming that
    the voltages in the network are constant. This requires:
    
    state_variables
    external_voltages
    XtoG
    xdot
    
    with optional arguments:
    
    jacobian
    memristor_params
    G_ON
    G_OFF
    """
    
    def __init__(self, lattice_shape, state_variables, external_voltages, XtoG, xdot, 
                 jacobian=None, threshold_voltages=None, G_ON=1, G_OFF=10**-2):
        self.state_variables = state_variables
        self.rows, self.cols = self.state_variables.nonzero()
        self.XtoG = XtoG
        self.G_ON = G_ON
        self.G_OFF = G_OFF
        self.threshold_voltages = threshold_voltages
        self.xdot=xdot
        self.integrator = ode(xdot, jac=jacobian)
        rnets.ResistorLattice2DCubic.__init__(self, self.XtoG(self.state_variables), external_voltages, lattice_shape)
    
    def update_conductance_matrix(self):
        self.G = self.XtoG(self.state_variables)
    
    def integrate_state_variables(self, delta_t, method='scipyode'):
        """
        Integrate the state variables of the network from their current value to their value a time delta_t
        later assuming that the voltages are constant during this time.  This method is only intended to update
        the state variables infinitesimally.  The attribute xdot is a function of the state variables and the
        voltage drop across a memristor that gives the first derivative of the state variables. 
        """
    
        # loop over each memristor. This could easily be done in parallel
        for i, j in zip(self.rows, self.cols):
            # Set the parameters of each memristor in the integrator
            voltage_drop = self.voltages[i] - self.voltages[j]
            
            # See if threshold voltages have been defined
            try:
                V_thresh = self.threshold_voltages[i, j]
            except TypeError:
                V_thresh = 1
                
            if method == 'scipyode':
                self.integrator.set_initial_value(self.state_variables[i, j], 0).set_f_params(voltage_drop,
                                                  V_thresh).set_jac_params(voltage_drop, V_thresh)
                integrated_state_variable = self.integrator.integrate(delta_t)
            elif method == 'myrk4':
                integrated_state_variable = RK4_next(self.state_variables[i, j], voltage_drop, V_thresh,
                                                     0, delta_t, self.xdot)
            
            # Check values and set to limits if we have strayed beyond them
            if integrated_state_variable >= self.G_ON:
                self.state_variables[i, j] = self.G_ON
            elif integrated_state_variable <= self.G_OFF:
                self.state_variables[i, j] = self.G_OFF
            else:
                self.state_variables[i, j] = integrated_state_variable
            
            if not self.integrator.successful():   
                print "Integration was unsuccessful"
                
            
        self.update_conductance_matrix()

