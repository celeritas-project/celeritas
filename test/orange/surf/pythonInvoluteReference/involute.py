import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton 
import copy

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
        
        # Test if Particle is on involute
        if self.IsOn(x,y):
            return 0
       
        # Line angle parameter
        beta = np.arctan(-v/u)
        
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
                
                if xInv**2+yInv**2 >= rmin**2:
                
                    dir2 = np.array([xInv-x,yInv-y])
                    
                    if np.dot(dir,dir2) >= 0:
                        newdist = np.sqrt((xInv-x)**2+(yInv-y)**2)
            
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
                    
        dist = distArray[-1]
        if dist >= 1e9:
            print("No Solution Found For Given Bounds")
            print("Bounds : rmin = {}, rmax = {} ".format(rmin,rmax))
            print("Particle Position : x = {}, y = {}".format(x,y))
            print("Particle Directiom : u = {}, v = {}".format(u,v))
            
            return 0
        
        return dist
                 
            
            
    # Root Function
    def f(self,t):
        a = self.u * np.sin(t+self.a0) - self.v * np.cos(t+self.a0)
        b = t * (self.u * np.cos(t+self.a0) + self.v * np.sin(t+self.a0))
        c = self.rb * (a-b)
        return c + self.x * self.v - self.y * self.u          
        
                
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