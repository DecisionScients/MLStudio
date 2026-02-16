#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : objectives.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, May 19th 2020, 1:06:30 pm                          #
# Last Modified : Tuesday, May 19th 2020, 1:06:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Objective functions and their gradients."""
from abc import ABC, abstractmethod

import numpy as np

from mlstudio.utils.data_manager import GradientScaler    
# --------------------------------------------------------------------------  #
#                        OBJECTIVE BASE CLASS                                 #
# --------------------------------------------------------------------------  #
class Benchmark(ABC):
    """Base class for objective functions."""

    def __init__(self, regularizer=None, gradient_scaler=GradientScaler()):
        self.regularizer = regularizer
        self.gradient_scaler = gradient_scaler

    @property
    def name(self):
        return "Benchmark Base Class"

    @property
    def density(self):
        """The linspace density per point of difference between min and max.""" 
        return 2

    @property
    def start(self):
        """Returns a good start point."""
        pass

    @property
    def minimum(self):
        """Returns a good start point."""
        pass    

    @property
    def range(self):
        """Returns x and y ranges for plotting."""
        pass
   
    
    @abstractmethod
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        pass
    
    @abstractmethod
    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        pass
# --------------------------------------------------------------------------  #
class Adjiman(Benchmark):
    """Base class for objective functions.""" 

    @property
    def name(self):
        return "Adjiman Objective"    

    @property   
    def start(self):
        return np.array([-4,0])

    @property
    def minimum(self):
        return np.array([2,0.10578])

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -4, 'max': 4}
        y = {'min': -5, 'max': 5}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        return np.cos(theta[0]) * np.sin(theta[1]) - (theta[0] / (theta[1]**2 + 1))

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = -(1/(theta[1]**2+1))*((theta[1]**2+1)*np.sin(theta[0])*np.sin(theta[1])+1)
        dfdy = 2*theta[0]*theta[1] /(theta[1]**2+1)**2 + np.cos(theta[0])*np.cos(theta[1])
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df

# --------------------------------------------------------------------------  #
class BartelsConn(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Bartels Conn Objective"
    
    @property
    def start(self):
        return np.array([-500,0])

    @property
    def minimum(self):
        return np.array([0,0])        

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -500, 'max': 500}
        y = {'min': -500, 'max': 500}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        a = abs(theta[0]**2 + theta[1]**2 + theta[0] * theta[1])
        b = abs(np.sin(theta[0]))
        c = abs(np.cos(theta[1]))
        return a + b + c

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        a = (2 * theta[0] + theta[1]) * np.sign(theta[0]**2 + theta[0] * theta[1] + theta[1]**2)
        b = np.cos(theta[0]) * np.sign(np.sin(theta[0]))
        dfdx = a + b
        a = (theta[0] + 2 * theta[1]) * np.sign(theta[0]**2 + theta[0] * theta[1] + theta[1]**2)
        b = np.sin(theta[1]) * np.sign(np.cos(theta[1]))
        dfdy = a - b
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        

# --------------------------------------------------------------------------  #
class Himmelblau(Benchmark):
    """Base class for objective functions."""   

    @property
    def name(self):
        return "Himmelblau Objective"     

    @property
    def start(self):
        return np.array([5,5])

    @property
    def minimum(self):
        return np.array([3,2])           

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -5, 'max': 5}
        y = {'min': -5, 'max': 5}
        return x, y    

    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        return (theta[0]**2 + theta[1] - 11)**2 + (theta[0]+theta[1]**2-7)**2

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = 4*theta[0]*(theta[0]**2+theta[1]-11)+2*theta[0]+2*theta[1]**2-14
        dfdy = 2*theta[0]**2 + 4*theta[1] * (theta[0]+theta[1]**2-7)+2*theta[1]-22
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        
        

# --------------------------------------------------------------------------  #
class Leon(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Leon Objective"    

    @property
    def start(self):
        return np.array([-1.5,1.5])

    @property
    def minimum(self):
        return np.array([1,1])  

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -1.5, 'max': 1.5}
        y = {'min': -2, 'max': 2}
        return x, y            

    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        return 100 * (theta[1] - theta[0]**2)**2 + (1 - theta[0])**2

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = -400*(-theta[0]**2+theta[1]) + 2*theta[0] - 2
        dfdy = -200*theta[0]**2 + 200 * theta[1]
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        
             

   

# --------------------------------------------------------------------------  #
class Rosenbrock(Benchmark):
    """Base class for objective functions."""

    @property
    def name(self):
        return "Rosenbrock Objective"    

    @property
    def start(self):
        return np.array([-5,10])

    @property
    def minimum(self):
        return np.array([1,1])  

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -10, 'max': 10}
        y = {'min': -10, 'max': 10}
        return x, y            

    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""
        a = 1
        b = 100
        return b*(theta[1]-theta[0]**2)**2+(a-theta[0])**2

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = -400*theta[0]*(-theta[0]**2+theta[1])+2*theta[0]-2
        dfdy = -200*theta[0]**2 + 200 * theta[1]
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        

# --------------------------------------------------------------------------  #
class Branin02(Benchmark):
    """Branin02 objective functions."""    

    @property
    def name(self):
        return "Branin02 Objective"    

    @property
    def start(self):
        return np.array([-5,-5])
        
    @property
    def minimum(self):
        return np.array([-3.2,12.53])           
    
    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -5, 'max': 15}
        y = {'min': -5, 'max': 15}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""        
        return (-1.275*theta[0]**2/np.pi**2 + 5*theta[0]/np.pi + theta[1]-6)**2\
            + (10-5/4*np.pi) * np.cos(theta[0]) * np.cos(theta[1]) \
                + np.log(theta[0]**2+theta[1]**2+1) + 10

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        # dfdx
        a = 1/(4*np.pi**4*(theta[0]**2+theta[1]**2+1))
        b = 8 * np.pi**4 * theta[0]
        c = 5*(theta[0] - 2 * np.pi)
        d = (theta[0]**2+theta[1]**2+1)
        e = (5 * theta[0]**2 - 20*np.pi*theta[0]+4*np.pi**2*(6-theta[1]))
        f = 5*np.pi**3 *  (-1+8*np.pi) * (theta[0]**2+theta[1]**2+1) \
            * np.sin(theta[0]) * np.cos(theta[1])
        dfdx = a * (b+c*d*e - f)
        # dfdy
        a = -(5*theta[0]**2/(2*(np.pi**2)))
        b = (10 * theta[0])/np.pi
        c = 2 * theta[1]
        d = (2*theta[1]) / (theta[0]**2+theta[1]**2+1)
        e = (10-(5/(np.pi*4)))
        f = np.sin(theta[1]) * np.cos(theta[0])
        dfdy = a + b + c + d - e * f - 12
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        


# --------------------------------------------------------------------------  #
class StyblinskiTank(Benchmark):
    """Styblinksi-Tank objective functions."""

    @property
    def name(self):
        return "Styblinski-Tank Objective"    

    @property
    def start(self):
        return np.array([-5,-4])

    @property
    def minimum(self):
        return np.array([-2.903534, -2.903534])  

    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -5, 'max': 5}
        y = {'min': -5, 'max': 5}
        return x, y            

    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""   
        a = 0
        for i in range(len(theta, **kwargs)):
            a += theta[i]**4 - 16 * theta[i]**2 + 5 * theta[i]

        return 1/2 * a

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = 2*theta[0]**3 - 16 * theta[0] + 5/2
        dfdy = 2*theta[1]**3 - 16*theta[1] + 5/2
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        
       

# --------------------------------------------------------------------------  #
class SumSquares(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Sum Squares Objective"    

    @property
    def start(self):
        return np.array([10,10])
        
    @property
    def minimum(self):
        return np.array([0,0])           
    
    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -10, 'max': 10}
        y = {'min': -10, 'max': 10}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""        
        return np.sum(theta[0]**2 + theta[1]**2)

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = 2 * theta[0]
        dfdy = 2 * theta[1]
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df                

# --------------------------------------------------------------------------  #
class ThreeHumpCamel(Benchmark):
    """Three hump camel objective functions."""    

    @property
    def name(self):
        return "Three Hump Camel Objective"    

    @property
    def start(self):
        return np.array([-1.5,1.5])
        
    @property
    def minimum(self):
        return np.array([0,0])           
    
    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -2, 'max': 2}
        y = {'min': -2, 'max': 2}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""        
        return 2*theta[0]**2 - 1.05*theta[0]**4 + ((theta[0]**6)/6)+theta[0]*theta[1]+theta[1]**2

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = theta[0]**5-((21*theta[0]**3)/5)+4*theta[0]+theta[1]
        dfdy = theta[0]+2*theta[1]
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        

# --------------------------------------------------------------------------  #
class Ursem01(Benchmark):
    """Ursem01 objective functions."""    

    @property
    def name(self):
        return "Ursem01 Objective"    

    @property
    def start(self):
        return np.array([0,2.5])
        
    @property
    def minimum(self):
        return np.array([1.69714,0])           
    
    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -2, 'max': 3}
        y = {'min': -2, 'max': 2.5}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""        
        return -np.sin(2*theta[0] - 0.5 * np.pi) - 3 * np.cos(theta[1]) - 0.5 * theta[0]

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        dfdx = -2 * np.sin(2 * theta[0]) - 0.5
        dfdy = 3 * np.sin(theta[1])
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        

# --------------------------------------------------------------------------  #
class Wikipedia(Benchmark):
    """Wikipedia objective functions."""    

    @property
    def name(self):
        return "Wikipedia Objective"    

    @property
    def start(self):
        return np.array([0,-1])
        
    @property
    def minimum(self):
        return np.array([-0.5,1])           
    
    @property
    def range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -1, 'max': 1}
        y = {'min': -1, 'max': 1}
        return x, y
    
    def __call__(self, theta, **kwargs):
        """Computes the objective function value"""        
        return np.sin(1/2*theta[0]**2 - 0.25 * theta[1]**2 + 3) * np.cos(2*theta[0]+1-np.exp(theta[1]))

    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        a = (theta[0] * np.cos(2*theta[0]-np.exp(theta[1])+1))
        b = np.cos((theta[0]**2)/2-(theta[1]**2)/4+3)
        c = -2*np.sin(2*theta[0])-np.exp(theta[1])+1
        d = np.sin((theta[0]**2)/2-(theta[1]**2)/4+3)
        e = -theta[1]/2*np.cos(2*theta[0]-np.exp(theta[1])+1)
        f = np.cos((theta[0]**2)/2-(theta[1]**2)/4+3)
        g = np.exp(theta[1])*np.sin(2*theta[0]-np.exp(theta[1])+1)
        h = np.sin((theta[0]**2)/2-(theta[1]**2)/4+3)
        dfdx = a * b + c * d
        dfdy = e * f + g * h
        # Package into gradient vector
        df = np.array([dfdx, dfdy])
        # Check gradient scale 
        df = self._check_gradient_scale(df)                
        return df        

          
