import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from involute import Involute

# Add Fuzzy Logic

class inOut:
    """
    Checks if Point is in out of involute.
    """
    
    def __init__(self, involute, rmin, rmax, tol=1e-7):
        """
        Initialize Involte where check is being performed on

        Args:
            involute (Involute): Involute object
            rmin (float): minimum radius
            rmax (float): maximum radius
        """
        
        self.involute = involute
        self.rb = involute.rb
        self.a0 = involute.a0
        self.sign = involute.sign
        self.tol = tol
        
        
        self.tmin = np.sqrt(rmin**2/self.rb**2-1) * self.sign
        self.tmax = np.sqrt(rmax**2/self.rb**2-1) * self.sign

            
        self.rmin = rmin
        self.rmax = rmax
            
    def isIn(self, x,y):
        """_summary_

        Args:
            x (float): x-coordinate of point
            y (_type_): y-coordinate of point

        Returns:
            bool: Returns true if poin is within bounded region
        """
        
        # First Check if Point is bellow rmin
        if x**2+y**2 < (self.rmin-self.tol)**2:
            return False
        
        # Second Check if Point is beyond rmax
        if x**2+y**2 > (self.rmax+self.tol)**2:
            return False
        
        # Find Location where Tangent from Point Intersects
        a = (x**2+y**2)
        b = -2*self.rb**2*x
        c = self.rb**4-y**2*self.rb**2
        
        xa = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        xb = (-b - np.sqrt(b**2-4*a*c))/(2*a)
        
        ya = np.sqrt(self.rb**2-xa**2)
        yb = -np.sqrt(self.rb**2-xa**2)
        yc = np.sqrt(self.rb**2-xb**2)
        yd = -np.sqrt(self.rb**2-xb**2)
        
        a = (x**2+y**2)
        b = -2*self.rb**2*y
        c = self.rb**4-x**2*self.rb**2
        
        yalpha = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        ybeta = (-b - np.sqrt(b**2-4*a*c))/(2*a)
        
        # Get First Tangent
        if math.isclose(yalpha, ya):
            point1 = np.array([xa,ya])
        else:
            point1 = np.array([xb,yc])
        
        # Get Seccond Tangent
        if math.isclose(ybeta, yb):
            point2 = np.array([xa,yb])
        else:
            point2 = np.array([xb,yd])
        
        # Determine which Tangent is used for Involute
        norm = np.array([-y,x])
        if np.sign(np.dot(point1,norm)) == self.sign:
            point = point1
        else:
            point = point2
        
        # Calculate angle of tangent   
        theta = np.arccos(point[0]/\
                          np.sqrt(point[0]**2+point[1]**2))
        if point[1] < 0:
            theta = (np.pi-theta)+np.pi
        
        # Set Interval Bounds
        tmin = np.max([0,self.tmin]) *self.sign
        tmax = np.min([2*np.pi,self.tmax])*self.sign
        
        while np.abs(tmax)<np.abs(tmin):
            tmax += 2*np.pi*self.sign
            tmax = np.min([np.abs(tmax),np.abs(self.tmax)])*self.sign
        
        while np.abs(theta) < np.abs(tmin):
            theta += 2*np.pi*self.sign
        
        rxy = np.sqrt(x**2+y**2)
        tPoint = np.sqrt(rxy**2/self.rb**2-1)*self.sign
        
        a1 = theta - tPoint
        
        # Check if Point is in First Interval
        if (a1 >= self.a0 or math.isclose(a1,self.a0)) and self.sign==1:
            return True
        if (a1 <= self.a0 or math.isclose(a1,self.a0)) and self.sign==-1:
            return True
        
        # Check if Point is in following Interval until theta exceeds tmax
        while np.abs(theta) < np.abs(self.tmax+self.a0):
            theta += 2*np.pi*self.sign
            tmin = copy.deepcopy(tmax)
            tmax += 2*np.pi*self.sign
            tmax = np.min([tmax,self.tmax])
            
            a1 = theta - tPoint

            # Check if Point is in Interval
            if (a1 >= self.a0 or math.isclose(a1,self.a0))\
                and (theta < tmax+self.a0 or math.isclose(theta, tmax+self.a0))\
                and self.sign==1:
                return True
            elif (a1 <= self.a0 or math.isclose(a1,self.a0))\
                and (theta > tmax+self.a0 or math.isclose(theta, tmax+self.a0))\
                and self.sign==-1:
                return True
            
        # Final on Involute Check
        if self.involute.IsOn(x, y):
            return True
        
        # If you get here Point is outside    
        return False
        
# Test Involute      
rb = 1.0
a0 = np.pi*0.5
t = np.linspace(0,4*np.pi,500)
sign = 1
involute = Involute(rb, a0, t, sign)

inout = inOut(involute, 2.0, 9.5)

# Plotting Information
tau = np.linspace(0,2*np.pi,500)
xmin = 2*np.cos(tau)
ymin = 2*np.sin(tau)
xmax = 9.5*np.cos(tau)
ymax = 9.5*np.sin(tau)
xlin = np.linspace(-3.2,-9.45,10)
theta = np.sqrt(9.5**2-1)+np.pi*0.5
ylin = -1/np.tan(theta)*(xlin-rb*np.cos(theta))+rb*np.sin(theta)

# Plot
plt.figure(figsize=(10,10))
plt.plot(involute.involuteX,involute.involuteY, color="black")
plt.plot(xmin,ymin, color="Blue", label="Rmin")
plt.plot(xmax,ymax, color="Purple", label="Rmax")
plt.plot(xlin,ylin, color ="Orange", label="ThetaMax")


# Origin Test
print("-------------------\n"
      "Testing Point (0,0)\nExpected Result: False")
x = 0 
y = 0
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red",label="Out of Bounds")

# Second in Minimum Circle Test
print("-------------------\n"
      "Testing Point (0,1)\nExpected Result: False")
x = 0 
y = 1
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")

# Outside Maximum Circle Test
print("-------------------\n"
      "Testing Point (5,9)\nExpected Result: False")
x = 5 
y = 9
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")

# Firts Interval after tmin Test
print("-------------------\n"
      "Testing Point (-2.5,0)\nExpected Result: True")
x = -2.5 
y = 0
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green", label="In Bounds")
else:
    plt.plot([x],[y], 'x', color="Red")

# Second Interval Test after tmin Test
print("-------------------\n"
      "Testing Point (5,2.5)\nExpected Result: True")
x = 5
y = 2.5
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")

# Outside but between rmin and rmax Test
print("-------------------\n"
      "Testing Point (-5,-5)\nExpected Result: False")
x = -5
y = -5
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")
    
# On Involute between rmin and t=2pi Test
print("-------------------\n"
      "Testing Point ({},{})\nExpected Result: True".format(\
                            involute.involuteX[100],involute.involuteY[100]))
x = involute.involuteX[100]
y = involute.involuteY[100]
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")
    
# On Involute between t=2pi and rmax Test
print("-------------------\n"
      "Testing Point ({},{})\nExpected Result: True".format(\
                            involute.involuteX[350],involute.involuteY[350]))
x = involute.involuteX[350]
y = involute.involuteY[350]
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")
    
# Fuzzy On Involute between t=2pi and rmax Test
print("-------------------\n"
      "Testing Point ({},{})\nExpected Result: True".format(\
                            involute.involuteX[300],involute.involuteY[300]))
x = involute.involuteX[300]+1e-8
y = involute.involuteY[300]+1e-8
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")
    
# Fuzzy Theta Max Test
print("-------------------")

x = -4
y = -1/np.tan(theta)*(x-rb*np.cos(theta))+rb*np.sin(theta)-1e-8
print("Testing Point ({},{})\nExpected Result: True".format(\
                            x,y))
print("Obtained Result : {}".format(inout.isIn(x, y)))
if inout.isIn(x, y):
    plt.plot([x],[y], 'o', color="Green")
else:
    plt.plot([x],[y], 'x', color="Red")

plt.legend()
plt.grid()

