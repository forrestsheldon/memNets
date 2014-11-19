# memristor_network.py
# By Forrest Sheldon
# This is a transcription of the classes I have created to solve
# and display resistor networks from my notebook
# Memristorr_Nets_V2.ipynb

import numpy as np
import scipy as sp
from scipy.integrate import ode
import resistor_networks as rnets

#================================================================
# MemristorNetwork
#================================================================

class MemristorNetwork(rnets.ResistorNetworkCC):
    """
    This class inherits from my ResistorNetworkCC class in the aim of generalizing it to solve arbitrary
    memristor networks.  This class will store the memristor network at a moment in time and possesses an
    update(dt) method that integrates the network forward a time dt, assuming that the voltages in the
    network are constant. This requires:
    
    state_variables   - A sparse NxN matrix in CSR format containing the values of state variables between
                        nodes i and j
    external_voltages - An Nx1 dense vector of external voltages.  Nodes not set to an external voltages contain a Nan.
                        The shape (N,) is preferred
    XtoG              - A function that returns a sparse matrix of conductances between nodes i and j given the state
                        variable matrix.  This matrix should be symmetrized, i.e. we must first calculate the conductance
                        of each bond and then symmetrize the resulting matrix.
    xdot              - A function giving the time derivative of the state variable as a function of (t, x, V, *args)
    
    with optional arguments:
    
    jacobian
    xdot_args
    x_min
    x_max
    """
    
    def __init__(self, state_variables, external_voltages, XtoM, xdot, jacobian=None, xdot_args=None,
                 x_min=0, x_max=1):
        self.state_variables = state_variables
        self.rows, self.cols = self.state_variables.nonzero()
        self.XtoG = XtoG
        self.xdot=xdot
        self.xdot_args = xdot_args
        self.x_min = x_min
        self.x_max = x_max
        self.integrator = ode(xdot, jac=jacobian)
        rnets.ResistorNetworkCC.__init__(self, self.XtoG(self.state_variables), external_voltages)
    
    def update_conductance_matrix(self):
        self.G = self.XtoG(self.state_variables)
    
    def integrate_state_variables(self, delta_t, method='scipyode'):
        """
        Integrate the state variables of the network from their current value to their value a time delta_t
        later assuming that the voltages are constant during this time.  This method is only intended to update
        the state variables infinitesimally.  The attribute xdot(t, x, V, *args) is a function of the state
        variables and the voltage drop across a memristor that gives the first derivative of the state variables. 
        """
    
        # loop over each memristor. This could easily be done in parallel
        for i, j in zip(self.rows, self.cols):
            # Set the parameters of each memristor in the integrator
            voltage_drop = self.voltages[i] - self.voltages[j]
            
            # See if threshold voltages have been defined
            try:
                xdot_arg = self.xdot_args[i, j]
            except TypeError:
                xdot_arg = 1
                
            if method == 'scipyode':
                self.integrator.set_initial_value(self.state_variables[i, j], 0).set_f_params(voltage_drop,
                                                  xdot_arg).set_jac_params(voltage_drop, xdot_arg)
                integrated_state_variable = self.integrator.integrate(delta_t)
            elif method == 'myrk4':
                integrated_state_variable = RK4_next(self.state_variables[i, j], voltage_drop, xdot_arg,
                                                     0, delta_t, self.xdot)
            
            # Check values and set to limits if we have strayed beyond them
            if integrated_state_variable >= self.x_max:
                self.state_variables[i, j] = self.x_max
            elif integrated_state_variable <= self.x_min:
                self.state_variables[i, j] = self.x_min
            else:
                self.state_variables[i, j] = integrated_state_variable
            
            if not self.integrator.successful():   
                print "Integration was unsuccessful"
                 
        self.update_conductance_matrix()

        
# I am making use of my RK4 integrator to provide a faster integration option.  The
# integrators provided by ode will offer a way to assess whether results given here
# are trustworthy

def RK4_next(current_x, V, xdot_arg, time, delta_t, func_xdot):
    
    def xdot(x, t):
        return func_xdot(t, x, V, xdot_arg)
    
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
