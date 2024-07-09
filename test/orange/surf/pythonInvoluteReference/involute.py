import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton 
import copy
import math

class Involute:
    
    def __init__(self, rb, a0, t, sign, tol = 1e-8):
        # Initial Parameters
        self.rb = rb
        self.a0 = a0
        self.t = t
        self.sign = sign
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.tol = tol
        
        # Angle of Inolute
        angle = t + a0
        
        # Initialize Involute Curve for Plotting
        self.involuteX = rb * (np.cos(angle)+t*np.sin(angle))
        self.involuteY = rb * (np.sin(angle)-t*np.cos(angle))
        
    def IsOn(self, x, y):
        """IsOn checks if a point is on the Involute.

        Args:
            x (float): x coordinate of point
            y (float): y coordinate of point

        Returns:
            bool: Boolian describing whether a point is on a curve.
        """
        
        # Calulate Distance from Origin
        rxy = np.sqrt(x**2+y**2)
        
        if rxy < self.rb:
            return False
        
        rxyA = np.linspace(np.max(np.array([self.rb,rxy*0.99])),1.01*rxy,100000)
        rxy = np.append(rxyA, rxy)
        
        # Find Phase Angle at which Involute is same distance
        if self.sign == 1:
            t = np.sqrt(rxy**2/self.rb**2-1)
        elif self.sign == -1:
            t = -1*np.sqrt(rxy**2/self.rb**2-1)
        else:
            print("Warning: Sign of Involute not provided. Calculation done for" 
                  " Positive Involute.")
            t = np.sqrt(rxy**2/self.rb**2-1)
            
        # Evaluate Involute point at phase
        angle = t + self.a0
        xInvA = self.rb * (np.cos(angle)+t*np.sin(angle))
        yInvA = self.rb * (np.sin(angle)-t*np.cos(angle))
        
        # Check if point is on involute within tolerance
        for xInv, yInv in zip(xInvA,yInvA):
        
            if x > xInv-self.tol*10 and x < xInv+self.tol*10 and \
               y > yInv-self.tol*10 and y < yInv+self.tol*10:
                return True
        
        return False 
    
    def Distance(self,x,y,u,v, rmin, rmax):
        # Store to be used elsewhere
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        dir = np.array([u,v])
        distA = np.array([])
        
        # Test if Particle is on involute
        if self.IsOn(x,y):
            distA = np.append(distA, 0)
       
        # Line angle parameter
        try:
            beta = np.arctan(-v/u)
        except:
            beta = np.pi*0.5*np.sign(-v)
        
        # First Interval Bounds
        if self.sign == 1:
            tLower = 0
            tUpper = beta - self.a0
            while tUpper <= 0:
                tUpper += np.pi
        elif self.sign == -1:
            tLower = beta - self.a0 + 2 * np.pi
            while tLower >= 0:
                tLower -= np.pi
            tUpper = 0
        
        # Root Function
        def f(t):
            a = self.u * np.sin(t+self.a0) - self.v * np.cos(t+self.a0)
            b = t * (self.u * np.cos(t+self.a0) + self.v * np.sin(t+self.a0))
            c = self.rb * (a-b)
            return c + self.x * self.v - self.y * self.u
        
        # Point Function
        def point(t):
            angle = t + self.a0
            xInv = self.rb * (np.cos(angle)+t*np.sin(angle))
            yInv = self.rb * (np.sin(angle)-t*np.cos(angle))
            return xInv, yInv
        
        # Iterate on Roots until you find a root further from point
        smalldist = np.inf
        dist = np.inf
        newdist = 1e9
        distArray = np.array([1e9])
        
        # Find Absolute tmin and tmax
        if sign == 1:
            tmin = np.sqrt(rmin**2/self.rb**2-1)
            tmax = np.sqrt(rmax**2/self.rb**2-1)
        elif sign == -1:
            tmin = -np.sqrt(rmin**2/self.rb**2-1)
            tmax = -np.sqrt(rmax**2/self.rb**2-1)
        
        i = 1
        while (tLower<=tmax and sign == 1) or \
              (tUpper>=tmax and sign == -1):
                  
            dist = copy.deepcopy(newdist)
            
            
            ta = tLower
            tb = tUpper
            
            fta = f(ta)
            ftb = f(tb)
            
            xa, ya = point(ta)
            xb, yb = point(tb)
            
            if (np.sqrt(xa**2+ya**2) > rmax and sign == 1) or \
               (np.sqrt(xb**2+yb**2) > rmax and sign == -1):
                   break
            
            
            if np.sign(fta) != np.sign(ftb):
                ftc = 1
            
                # Iteration
                while np.abs(ftc) >= self.tol:
                    
                    # Regula Falsi : Buggs out sometimes
                    tc = (ta*ftb-tb*fta)/(ftb-fta)
                    
                    # # Illinois Falsi
                    # if sign == 1:
                    #     tc = (ta*ftb*0.5-tb*fta)/(ftb*0.5-fta)
                    # elif sign == -1:
                    #     tc = (ta*ftb-tb*fta*0.5)/(ftb-fta*0.5)                    
                    
                    # # Secant : Starts to translate sometimes
                    # tc = (fta*(tb-ta))/(fta-ftb)+ta
                                        
                    ftc = f(tc)
                    
                    if np.sign(ftc) == np.sign(ftb):
                        tb = copy.deepcopy(tc)
                        ftb = f(tb)
                    else:
                        ta  = copy.deepcopy(tc)
                        fta = f(ta)

                    
                t = tc
                xInv,yInv = point(t)
                
                if xInv**2+yInv**2 >= rmin**2 -self.tol*10 and \
                    xInv**2+yInv**2 < rmax**2 + self.tol*10:
                
                    dir2 = np.array([xInv-x,yInv-y])
                    
                    if np.dot(dir,dir2) >= 0:
                        newdist = np.sqrt((xInv-x)**2+(yInv-y)**2)
                        if newdist >= self.tol*10:
                            distA = np.append(distA, newdist)
            
                if self.sign == 1:
                    tLower = copy.deepcopy(tUpper)
                    tUpper += np.pi
                    # if tLower > tmax:
                    #     newdist = 2*dist
                elif self.sign == -1:
                    tUpper = copy.deepcopy(tLower)
                    tLower -= np.pi
                    # if tUpper < tmin:
                    #     newdist = 2 * dist
            else:
                if self.sign == 1:
                    tLower = copy.deepcopy(tUpper)
                    tUpper += np.pi*0.5/i
                    i+=1
                elif self.sign == -1:
                    tUpper = copy.deepcopy(tLower)
                    tLower -= np.pi*0.5/i
                    i+=1
                    
            if newdist <= distArray[-1]:
                distArray = np.append(distArray, newdist)
        
        return distA
                 
            
            
    # Root Function
    def f(self,t):
        a = self.u * np.sin(t+self.a0) - self.v * np.cos(t+self.a0)
        b = t * (self.u * np.cos(t+self.a0) + self.v * np.sin(t+self.a0))
        c = self.rb * (a-b)
        return c + self.x * self.v - self.y * self.u          
        
if False:                
    # First Test   
    rb = 1.0
    a0 = 0
    t = np.linspace(0,1.999*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
    
    #First Point (0,0) Direction (0,1)
    x = 0
    y = 0
    u = 0
    v = 1
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([2.9716938706909275])
    if math.isclose(dist[0],distTrue[0]):
        print("Test 1: Passed")
    else:
        print("Test 1: Failed")
        
    # Second Test   
    rb = 1.5
    a0 = 0
    t = np.linspace(0,1.999*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
    
    #Second Point (-1.5,1) Direction (0.2,0.9797958971)
    x = -1.5
    y = 1
    u = 0.2
    v = 0.9797958971
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([3.7273045229241015])
    if math.isclose(dist[0],distTrue[0]):
        print("Test 2: Passed")
    else:
        print("Test 2: Failed")
        
    # Third Test   
    rb = 1.1
    a0 = np.pi*0.5
    t = np.linspace(0,1.999*np.pi,500)
    sign = -1
    involute = Involute(rb, a0, t, sign)
    
    #Third Point (-0.2,1.1) Direction (1,0)
    x = -0.2
    y = 1.1
    u = 1
    v = 0
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([0.2, 2.764234602725404])
    if math.isclose(dist[0],distTrue[0]) and math.isclose(dist[1],distTrue[1]):
        print("Test 3: Passed")
    else:
        print("Test 3: Failed")
    
    # Fourth Test   
    rb = 1.1
    a0 = -np.pi*0.5
    t = np.linspace(0,1.999*np.pi,500)
    sign = -1
    involute = Involute(rb, a0, t, sign)
    
    #Fourth Point (-0.0001,-1.11) Direction (-0.2,0.9797958971)
    x = -0.0001
    y = -1.11
    u = -0.1
    v = 0.9949874371
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([0.0036178033243678843, 6.0284475628795926])
    if math.isclose(dist[0],distTrue[0],rel_tol=1e-7) and math.isclose(dist[1],distTrue[1]):
        print("Test 4: Passed")
    else:
        print("Test 4: Failed")
        
    # Fifth Test   
    rb = 1.1
    a0 = -np.pi*0.5
    t = np.linspace(0,1.999*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
    
    #Fifth Point (0.0058102462574510716,-1.1342955336941216) 
    #      Direction (0.7071067812,0.7071067812)
    x = 0.0058102462574510716
    y = -1.1342955336941216
    u = 0.7071067812
    v = 0.7071067812
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([0, 4.652832754892369])
    if math.isclose(dist[0],distTrue[0]) and math.isclose(dist[1],distTrue[1]):
        print("Test 5: Passed")
    else:
        print("Test 5: Failed")
        
    # Sixth Test   
    rb = 1.1
    a0 = -np.pi*0.5
    t = np.linspace(0,1.999*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
    
    #Sixth Point ( -6.865305298657132,2.723377881785643) 
    #      Direction (0.7071067812,0.7071067812)
    x = -6.865305298657132
    y = -0.30468305643505367
    u = 0.9933558377574788
    v = -0.11508335932330707
    tmin = 0
    tmax = 1.999*np.pi
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([0, 6.911224915264738, 9.167603472624553])
    if math.isclose(dist[0],distTrue[0]) and math.isclose(dist[1],distTrue[1]) \
       and math.isclose(dist[2],distTrue[2]):
        print("Test 6: Passed")
    else:
        print("Test 6: Failed")
        
    # Seventh Test   
    rb = 3
    a0 = np.pi
    t = np.linspace(0.5, 4,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
        
    #Seventh Point ( 0,-2) 
    #      Direction (1,0)
    x = 0
    y = -2
    u = 1
    v = 0
    tmin = 0.5
    tmax = 4
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    if dist.size == 0:
        print("Test 7: Passed")
    else:
        print("Test 7: Failed")
        
    # Eighth Test   
    rb = 3
    a0 = np.pi
    t = np.linspace(2, 4,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
        
    #Eighth Point ( -4.101853006408607,-5.443541628262038) 
    #      Direction (0,1)
    x = -4.101853006408607
    y = -5.443541628262038
    u = 0
    v = 1
    tmin = 2
    tmax = 4
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([0])
    if math.isclose(dist[0],distTrue[0]):
        print("Test 8: Passed")
    else:
        print("Test 8: Failed")
        
    # Ninth Test   
    rb = 0.5
    a0 = -np.pi * 0.4 + np.pi
    t = np.linspace(2.5*np.pi, 3.5*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
        
    #Ninth Point ( -4,2) 
    #      Direction (0.894427191,-0.4472135955)
    x = 4
    y = 2
    u = -0.894427191
    v = -0.4472135955
    tmin = 2
    tmax = 4
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    distTrue = np.array([6.0371012183803652])
    if math.isclose(dist[0],distTrue[0]):
        print("Test 9: Passed")
    else:
        print("Test 9: Failed")
        
    # Tenth Test   
    rb = 0.75
    a0 = -2*np.pi
    t = np.linspace(2.5*np.pi, 3.5*np.pi,500)
    sign = -1
    involute = Involute(rb, a0, t, sign)
        
    #Tenth Point ( -7,-1) 
    #      Direction (0.894427191,-0.4472135955)
    x = -7
    y = -1
    u = 0.894427191
    v = -0.4472135955
    tmin = 2
    tmax = 4
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    if dist.size == 0:
        print("Test 10: Passed")
    else:
        print("Test 10: Failed")
        
    # Elventh Test   
    rb = 0.25
    a0 = -2*np.pi
    t = np.linspace(2.5*np.pi, 3.5*np.pi,500)
    sign = 1
    involute = Involute(rb, a0, t, sign)
    
    plt.plot(involute.involuteX, involute.involuteY)
    
    #Elventh Point (-2,1) 
    #      Direction (0.894427191,-0.4472135955)
    x = -2
    y = 1
    u = 0.4472135955
    v = -0.894427191
    tmin = 2
    tmax = 4
    rmin = rb*np.sqrt(1+tmin**2)
    rmax = rb*np.sqrt(1+tmax**2)
    dist  = involute.Distance(x, y, u, v, rmin, rmax)
    if dist.size == 0:
        print("Test 11: Passed")
    else:
        print("Test 11: Failed")       
        
                
# Test Involute      
rb = 1.0
a0 = 0
t = np.linspace(0,4*np.pi,500)
sign = 1
involute = Involute(rb, a0, t, sign)

# # IsOn test
# onBool = involute.IsOn(involute.involuteX[4], involute.involuteY[4])
# print("On Involute test.\nExpeccted Result : True\nResult : {}".format(onBool))
# offBool = involute.IsOn(involute.involuteX[4], involute.involuteY[10])
# print("Off Involute test.\nExpeccted Result : False\nResult : {}".format(offBool))

# # Falsi Iteration Tests
# print("\nFalsi Iteration Tests\n---------------")

# # On Invoute Test : Passed
# print("On Involute Test")

# x = involute.involuteX[4]
# y = involute.involuteY[4]
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)

# distPIOn = involute.Distance(x,y,u,v, rb, 8.0)
# print("Particle On Involute Test.\nExpected Result : 0.0.\n"
#       "Result : {}".format(distPIOn))

# # In Circle of Involute Test :
# print("In Circle of Involute Test")

# x = 0.01
# y = 0.01
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)


# distIn = involute.Distance(x, y, u, v, rb, 8.0)
# x1 = x + distIn*u
# y1 = y + distIn*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.05)
# plt.plot([x1],[y1], 'o')

# # Out of Circle of Involute Test :
# print("Out of Circle of Involute Test")
# x = -2
# y = 2
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)
# u = -0.0669427622799394
# v = -0.9977568173549763

# distOut = involute.Distance(x, y, u, v, rb, 5.0)
# x1 = x + distOut*u
# y1 = y + distOut*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.05)
# plt.plot([x1],[y1],"*")

# # Random Point Test :
# # Extending Range for this test
# # Test Involute      
# rb = 1.0
# a0 = 0
# t = np.linspace(0,8*np.pi,500)
# sign = 1
# involute = Involute(rb, a0, t, sign)
# print("Random Point Test")
# x = np.random.rand()*20-10
# y = np.random.rand()*20-10
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)

# distOut = involute.Distance(x, y, u, v, rb, 25.0)
# x1 = x + distOut*u
# y1 = y + distOut*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.15)
# plt.plot([x1],[y1],"*")

# # Negative Random Point Test :
# # Extending Range for this test
# # Test Involute      
# rb = 1.0
# a0 = 0
# t = np.linspace(0,-8*np.pi,500)
# sign = -1
# involute = Involute(rb, a0, t, sign)
# print("Random Point Test")
# x = np.random.rand()*20-10
# y = np.random.rand()*20-10
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)

# distOut = involute.Distance(x, y, u, v, rb, 25.0)
# x1 = x + distOut*u
# y1 = y + distOut*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.15)
# plt.plot([x1],[y1],"*")

# # Random Point Test Higher rmin:
# # Extending Range for this test
# # Test Involute      
# rb = 1.0
# a0 = 0
# tmin = np.sqrt(8**2-1)
# t = np.linspace(tmin,8*np.pi,500)
# sign = 1
# involute = Involute(rb, a0, t, sign)
# t = np.linspace(0,8*np.pi,500)
# involute1 = Involute(rb, a0, t, sign)
# print("Random Point Test")
# x = np.random.rand()*20-10
# y = np.random.rand()*20-10
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)

# distOut = involute.Distance(x, y, u, v, 8, 25.0)
# x1 = x + distOut*u
# y1 = y + distOut*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute1.involuteX,involute1.involuteY)
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.15)
# plt.plot([x1],[y1],"*")

# # Negative Random Point Test Higher rmin:
# # Extending Range for this test
# # Test Involute      
# rb = 1.0
# a0 = 0
# tmin = np.sqrt(8**2-1)
# t = np.linspace(-tmin,-8*np.pi,500)
# sign = -1
# involute = Involute(rb, a0, t, sign)
# t = np.linspace(0,-8*np.pi,500)
# involute1 = Involute(rb, a0, t, sign)
# print("Random Point Test")
# x = np.random.rand()*20-10
# y = np.random.rand()*20-10
# theta = np.random.rand()*2*np.pi
# u = np.cos(theta)
# v = np.sin(theta)

# distOut = involute.Distance(x, y, u, v, 8, 25.0)
# x1 = x + distOut*u
# y1 = y + distOut*v
# print("Is On Involute?")
# print(involute.IsOn(x1, y1))
# print("Check Figure For Correct Root")
# plt.figure(figsize=(10,10))
# plt.plot(involute1.involuteX,involute1.involuteY)
# plt.plot(involute.involuteX,involute.involuteY)
# plt.arrow(x,y,u,v,width=0.15)
# plt.plot([x1],[y1],"*")

# plt.grid()