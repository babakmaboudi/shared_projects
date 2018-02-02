from __future__ import print_function
from dolfin import *
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pylab as plt
from fenics import *
from mshr import *
from math import pi, sin, cos, sqrt
 
# Parameters
R = 1.1
r = 0.4
t = 10
x = R*cos(float(t) / 180 * pi)
y = 0
z = R*sin(t)
 
# Create geometry
s1 = Sphere(Point(0, 0, 0), 1)
s2 = Sphere(Point(x, y, z), r)
b1 = Box(Point(-2, -2, -0.03), Point(2, 2, 0.03))
geometry = s1 - s2 - b1

L = 1
W = 0.2
w=L/500
b1 = Box(Point(0, 0, 0), Point(L, W, W))
b2 = Box(Point(L/2-w, 0, W/2), Point(L/2+w, W, W))
geometry = b1 - b2
 
# Create mesh
mesh = generate_mesh(geometry,32)

File("cracked_beam.pvd") << mesh