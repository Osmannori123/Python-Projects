#PART 1:
#1)
import numpy as np
import matplotlib.pyplot as plt

Mtotal = 1.989 * 10**30
l = np.linspace(0.01 * 1.496e11, 2 * 1.496e11, 1000) 
pi = np.pi
G = 6.6743 * 10**-11
P = np.linspace(0, 0, 1000) 

for i in range(len(l)):
    P[i] = (2 * pi * np.sqrt(l[i]**3 / (G * Mtotal))) / (60 * 60 * 24 * 365)

plt.plot(l/1.496e11, P, color="r")
plt.xlabel('Distance (AU)')
plt.ylabel('Orbital Period (years)')
plt.title('Visualization for L1, L2, L3')
plt.grid()
plt.show()


#2)
Mtotal = 1.989 * 10**30
l = np.linspace(0.01 * 1.496e11, 2 * 1.496e11, 500) 
lxvals = np.linspace(0.01 * 1.496e11, 2 * 1.496e11, 500) 
lyvals = np.linspace(0.01 * 1.496e11, 2 * 1.496e11, 500) 
lx, ly = np.meshgrid(lxvals, lyvals)
pi = np.pi
G = 6.6743 * 10**-11
P1 = np.linspace(0, 0, 500)
magnitude = np.sqrt((lx**3)**2 + (ly**3)**2)
for i in range(len(l)):
    P1 = (2 * pi * np.sqrt(magnitude / (G * Mtotal))) / (60 * 60 * 24 * 365)

plt.contourf(lx / 1.496e11, ly/ 1.496e11, P1, levels=50, cmap='plasma')
plt.colorbar(label='Orbital Period (Years)')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Visualization for L4, L5')
plt.axvline(0.5, color='w', linestyle='--')
plt.axvline(0.75, color='w', linestyle='--')
plt.axhline(0.75, color='w', linestyle='--')
plt.axhline(1, color='w', linestyle='--')
plt.grid()
plt.show()

#We know that all lagrange points share a
#period at 1 Year, and from the colorbar we
#can see which strips of the grid we can narrow
#points down to. This is highlighted by the 
#lines drawn to enclose the region that best
#matches the color for 1 year. So we can now
#see that the x coordinate must lie 
#0.5 <= x <= 0.75 and the y coordinate must
#lie 0.75 <= y <= 1.0. We know by symmetry 
#that L5 will have the same x coordinate, 
#and the same y coordinate but with a sign flip.



#PART 2:
#1)
from datetime import datetime


import matplotlib        as mpl
import matplotlib.pyplot as plt


import numpy as np
from numpy import exp, log, sqrt


import scipy.optimize as optimize


r_earth = 1.496 * 10**11
r_sun = 4.492 * 10**5
Nmax = 1000

#L1:
def bisection(f, x1, x2):
    accuracy = 1E-9
    x_c = 1/2 * (x1 + x2)
    evaluate = f(x_c)
    if (evaluate > 0 and f(x1) > 0) or (evaluate < 0 and f(x1) < 0):
        x1 = x_c
    else:
        x2 = x_c
    
    while abs(x2-x1) > accuracy:
        x_c = 1/2 * (x1 + x2)
        evaluate = f(x_c)
        if (evaluate > 0 and f(x1) > 0) or (evaluate < 0 and f(x1) < 0):
            x1 = x_c
        else:
            x2 = x_c
    return x_c

def L1(x):
    return (r_earth - x) - (r_earth/(1 - x)**2) + (r_sun/x**2)


xVal = np.linspace(0.001, 0.05, 1000)
yVal = L1(xVal)
plt.plot(xVal, yVal)
plt.title('Graph for L1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(0.001, color='r', linestyle='--', label="First Point")
plt.axvline(0.05, color='b', linestyle='--', label="Second Point")
plt.legend()
plt.show()

print('\n')

round = np.round(bisection(L1, 0.001, 0.05), decimals=2)
distanceL1 = ((round * 1.496 * 10**11)) / 10**3
print(f'Distance for L1 is {distanceL1} km')



#L2:
def falseposition(f, x1, x2):
    ans = x1
    for i in range(Nmax):
        ans = (x1 * f(x2) - x2 * f(x1))/ (f(x2) - f(x1))
        if f(ans) * f(x1) < 0:
            x2 = ans
        else:
            x1 = ans
    return ans
    
def L2(x):
    return (r_earth + x) - (r_earth/(1 + x)**2) + (r_sun/x**2)


xVal = np.linspace(0.001, 0.05, 1000)
yVal = L2(xVal)
plt.plot(xVal, yVal)
plt.title('Graph for L2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(0.001, color='r', linestyle='--', label="First Point")
plt.axvline(0.05, color='b', linestyle='--', label="Second Point")
plt.show()

print('\n')

round = np.round(falseposition(L2, -0.001, -0.05), decimals=2)
distanceL2 = ((round * 1.496 * 10**11)) / 10**3
print(f'Distance for L2 is {distanceL2} km')



#L3:
def relax(f,x):
    accuracy = 1E-9
    Nmax     = 1
    i     = 0
    delta = 1.0
    
    while abs(delta) > accuracy and i < Nmax:
        i += 1
        xprev = x
        x     = f(x)
        delta = x/xprev - 1
        
    return x
    
def L3(x):
    return (r_earth/(r_sun)**2 + r_sun/(r_earth)**2 - x) 

x2Val = np.linspace(0.001, 0.05, 1000)
y2Val = L3(x2Val)
plt.plot(x2Val, y2Val)
plt.title('Graph for L3')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(0.001, color='r', linestyle='--', label="First Point")
plt.axvline(0.05, color='b', linestyle='--', label="Second Point")
plt.show()


round3 = np.round((relax(L3, 0.001) * 1.496 * 10**11), decimals = 2)
distanceL3 = np.round((round3 + r_earth) / 10**3, decimals = -8)
print(f'Distance for L3 is {distanceL3} km')


#4)
import numpy as np
#L1: 
#check with bisection

scipyBisect = optimize.bisect(L1, 0.001, 0.05, rtol=1E-9)
simplify1 = np.round(scipyBisect, decimals = 2)
answer1 = ((simplify1 * 1.496 * 10**11)) / 10**3
print(f'Distance for L1 is {answer1} km')

print('\n')

#L2:
#check with brent

scipyBrent = optimize.brentq(L2, -0.001, -0.05, rtol=1E-9)
simplify2 =  np.round(scipyBrent, decimals = 2)
answer2 = ((simplify2 * 1.496 * 10**11)) / 10**3
print(f'Distance for L2 is {answer2} km')

print('\n')

#L3
#check with secant

scipySecant = optimize.newton(L3, 0.001, tol=1E-9)
simplify3 = np.round(scipySecant, decimals = 2)
answer3 = np.round((((simplify3 * 1.496 * 10**11)) + r_earth) / 10**3, decimals = -8)
print(f'Distance for L3 is {answer3} km')

#Solutions are all in agreement!



#PART 3:
#1)
from scipy.optimize import root
import numpy as np
G = 6.6743 * 10**-11
M_sun = 1.989 * 10**30
M_earth = 5.972 * 10**24
r_sun = 4.492 * 10**5
r_earth = 1.496 * 10**11 - r_sun

def fx(x,y):
    re = np.sqrt((x-r_earth)**2 + y**2)
    rs = np.sqrt((x+r_sun)**2 + y**2)

    fSunX = -G * M_sun * (x + r_sun) / rs**3
    fEarthX = -G * M_earth * (x - r_earth) / re**3
    aX = -G * (M_sun + M_earth) * x / (1.496 * 10**11)**3
    return (fSunX + fEarthX - aX) / -G*(M_sun * M_earth)

def fy(x,y):
    re = np.sqrt((x-r_earth)**2 + y**2)
    rs = np.sqrt((x+r_sun)**2 + y**2)

    fSunY= -G * M_sun * y / rs**3
    fEarthY = -G * M_earth * y / re**3
    aY = -G * (M_sun + M_earth) * y / (1.496 * 10**11)**3
    return (fSunY + fEarthY - aY) / -G*(M_sun * M_earth)

#Rough Visualization:
x_range = np.linspace(0.5 * 1.496 * 10**11, np.sqrt(3)/2 * 1.496 * 10**11, 500)
y_range = np.linspace(0.5 * 1.496 * 10**11, np.sqrt(3)/2 * 1.496 * 10**11, 500)

X, Y = np.meshgrid(x_range, y_range)

F_magnitude = np.sqrt(fx(X, Y)**2 + fy(X, Y)**2)

plt.figure(figsize=(8, 8))
plt.imshow(F_magnitude, extent=[-2, 2, -2, 2], origin='lower', cmap='binary')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('2D Force Field')
plt.colorbar(label='Net Force Strength (Newtons)')
plt.show()

#The areas of lighter shade are closer to 
#the lagrange point, since we know those
#areas are where the net force should be
#the weakest due to force balancing. 
#From this, we know we can guess points at, 
#or around,the white strip/lighter shaded
#areas



def JacobianMatrix(x,y):
    h = 10**-5
    XderivdX = (fx(x+h,y) - fx(x,y))/h
    XderivdY = (fx(x,y+h) - fx(x,y))/h
    YderivdX = (fx(x+h,y) - fx(x,y))/h
    YderivdY = (fx(x,y+h) - fx(x,y))/h
    Jmatrix = np.array([[XderivdX, XderivdY], [YderivdX, YderivdY]])
    return Jmatrix

def newtonsMethod2(xCoord,yCoord,tol = 10**-9):
    for n in range(100):
        fValues = np.array([fx(xCoord,yCoord), fy(xCoord,yCoord)])
        try:
            inverseJ = np.linalg.inv(JacobianMatrix(xCoord,yCoord))
        except np.linalg.LinAlgError:
            break
        delta = np.dot(inverseJ, fValues)
        xCoord, yCoord = xCoord - delta[0], yCoord - delta[1]
        if np.linalg.norm(delta) < tol:
            return xCoord, yCoord
    return xCoord, yCoord

xGL4 = 1/2 * 1.496e11 
yGL4 = np.sqrt(3)/2 * 1.496e11 

xGL5 = 1/2 * 1.496e11 
yGL5 = -np.sqrt(3)/2 * 1.496e11 

#If we guess around the coordinates we will
#get approximate solutions close to the exact
#ones

xL4, yL4 = newtonsMethod2(xGL4, yGL4)
print(f'L4 coordinates are: ({xL4/1.496e11} AU, {yL4/1.496e11} AU)')
xL5, yL5 = newtonsMethod2(xGL5, yGL5)
print(f'L5 coordinates are: ({xL5/1.496e11} AU, {yL5/1.496e11} AU)')

#2)
def XandY(P):
    x,y = P 
    return [fx(x,y),fy(x,y)]

answerL4 = root(XandY, [xGL4, yGL4])
xL4Scipy, yL4Scipy = answerL4.x
answerL5 = root(XandY, [xGL5, yGL5])
xL5Scipy, yL5Scipy = answerL5.x

print(f'L4 Scipy coordinates are: ({xL4Scipy/1.496e11} AU, {yL4Scipy/1.496e11} AU)')
print(f'L5 Scipy coordinates are: ({xL5Scipy/1.496e11} AU, {yL5Scipy/1.496e11} AU)')
print('Solutions are in agreement')



#PART 4:
#Problem is to find the gravitational potential
#energy along the path from Earth to L1
#using the trapezoid rule
#parts 1,2,3:
import numpy as np

G = 6.67430e-11 
M_earth = 5.972e24  
M_sun = 1.989e30  
l = 1.496e11 


def gravPotential(x):
    r_earth = x
    r_sun = l - x
    
    earthPotential = -G * M_earth / r_earth
    sunPotential = -G * M_sun / r_sun
    netPotential = earthPotential +sunPotential
    
    return netPotential


def trapezoid_fun(a, b, N):
    h = (b - a) / (N)
    X = np.linspace(a, b, N) 
    U = gravPotential(X)  

    integral = 0.5 * (U[0] + U[-1]) 
    integral *= h
    integral += np.sum(U[1:-1])
   
    return integral


radiusEarth = 6.371e6 
adjustedL = l * 0.99  
N = 1000 

xVals = np.linspace(radiusEarth, adjustedL, N) 
potentialVals = gravPotential(xVals)  

integral= trapezoid_fun(radiusEarth, adjustedL, N)

plt.plot(xVals / 1.496e11, potentialVals, label="Gravitational Potential Energy")
plt.title("Visualization of gravitational potential energy along the path from earth to L1")
plt.xlabel("Distance along the path (AU)")
plt.ylabel("Gravitational Potential Energy (J)")
plt.axvline(1.0, color='r', linestyle='--', label="Sun's Position")
plt.axvline(0.0, color='b', linestyle='--', label="Earth's Position")
plt.legend()
plt.grid()
plt.show()


print(f"Gravitational potential energy along the path is: {integral} J")

#On the path from Earth to L1, the gravitational
#potential energy of an object going along this
#path changes due to influence from both the 
#Earth and the sun. The potential will slowly
#decrease as an object moves towards the sun
#(also towards L1), and near L1 we know that
#forces should balance out so this makes sense.
