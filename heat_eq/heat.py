from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

g = Expression("0",degree=2)
u = TrialFunction(V)
v = TestFunction(V)
coef = np.zeros(33*33)
u0 = Function(V)
u0.vector().set_local( coef )
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=2)

dt = 0.002

a = u*v*dx + dt*inner(grad(u), grad(v))*dx
L = u0*v*dx + dt*f*v*dx

A = assemble(a)
#b = assemble(L)

temp = Function(V)

vtkfile = File('results/solution.pvd')

#vtkfile << (u0,0.0)

for i in range(0,500):
	print(i)
	b = assemble(L)
	bc.apply(A,b)
	solve(A,temp.vector(),b)
	u0.assign(temp)

	vtkfile << (temp,i*dt)

plot(u0)
plt.show()
