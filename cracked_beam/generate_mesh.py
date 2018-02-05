from __future__ import print_function
from dolfin import *
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pylab as plt
from fenics import *
from mshr import *
from math import pi, sin, cos, sqrt
 
L = 1
W = 0.2
w=L/100
b1 = Box(Point(0, 0, 0), Point(L, W, W))
b2 = Box(Point(L/2-w, 0, W/2), Point(L/2+w, W, W))
#geometry = b1 - s2

# Parameters
R = W/4
r = 0.08
t = W
x = W/2+R*cos(float(t) / 180 * pi)
y = W/2
z = R*sin(t)
 
# Create geometry
s1 = Sphere(Point(x+L-3/2*W, y, z), r)
s2 = Sphere(Point(x, y, z), r)
#b1 = Box(Point(-2, -2, -0.03), Point(2, 2, 0.03))
geometry = b1 - s2 -s1

#r1 = Rectangle(Point(0,0),Point(L,W))
#r2 = Rectangle(Point(L/2-w,W/2),Point(L/2+w,W))
#geometry = r1-r2

# Create mesh
mesh = generate_mesh(geometry,10)

File("results/test.pvd") << mesh