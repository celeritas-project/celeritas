#!/usr/bin/env python3
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""\
Calculate the outward normal of an involute.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from involute import Involute

class GetOutNorm:
    """
    Calculate outward normal given point on involute and  specified involute
    """
    
    def __init__(self, involute):
        self.involute = involute
        self.rb = involute.rb
        self.a0 = involute.a0
        self.sign = involute.sign
        
    def __call__(self, x, y):
        # Get t of point
        rxy = np.sqrt(x**2+y**2)
        t = np.sqrt((rxy/self.rb)**2-1)*self.sign
        
        # Calculate Norm
        angle = t+self.a0
        norm = np.array([
                         np.sin(angle),
                        -np.cos(angle)
                        ]) * self.sign
        
        return norm

rb = 1.0
a0 = np.pi*0.5
t = np.linspace(0,2*np.pi,500)
sign = 1
involuteP = Involute(rb, a0, t, sign)
involuteN = Involute(rb, a0, -t, -sign)

# Plot
plt.figure(figsize=(17.5,10))
plt.plot(involuteP.involuteX,involuteP.involuteY, color="tab:green")
plt.plot(involuteN.involuteX,involuteN.involuteY, color="tab:red")


# Positive Involute Test
normP = GetOutNorm(involuteP)
x = involuteP.involuteX[0]
y = involuteP.involuteY[0]
norm = normP(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

x = involuteP.involuteX[20]
y = involuteP.involuteY[20]
print(x,y)
norm = normP(x,y)
print(norm)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

x = involuteP.involuteX[50]
y = involuteP.involuteY[50]
norm = normP(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

x = involuteP.involuteX[100]
y = involuteP.involuteY[100]
norm = normP(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

x = involuteP.involuteX[200]
y = involuteP.involuteY[200]
norm = normP(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

x = involuteP.involuteX[400]
y = involuteP.involuteY[400]
norm = normP(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Green')

# Negative Involute Test
normN = GetOutNorm(involuteN)
x = involuteN.involuteX[0]
y = involuteN.involuteY[0]
norm = normN(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')

x = involuteN.involuteX[20]
y = involuteN.involuteY[20]
norm = normN(x,y)
print(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')
print(norm)

x = involuteN.involuteX[50]
y = involuteN.involuteY[50]
norm = normN(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')

x = involuteN.involuteX[100]
y = involuteN.involuteY[100]
norm = normN(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')

x = involuteN.involuteX[200]
y = involuteN.involuteY[200]
norm = normN(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')

x = involuteN.involuteX[400]
y = involuteN.involuteY[400]
norm = normN(x,y)
plt.arrow(x,y,norm[0],norm[1],width=0.05, color='Red')

plt.grid()
plt.gca().set_aspect("equal")

plt.xlabel('x')
plt.ylabel('y')