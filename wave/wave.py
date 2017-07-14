from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt


class PeriodicBoundary(SubDomain):

# periodic boundary conditions
	def inside(self, x, on_boundary):
		return bool( (near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0], 0) and near(x[1], 1)) or (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

	def map(self, x, y):
		if near(x[0], 1) and near(x[1], 1):
			y[0] = x[0] - 1.
			y[1] = x[1] - 1.
		elif near(x[0], 1):
			y[0] = x[0] - 1.
			y[1] = x[1]
		else:
			y[0] = x[0]
			y[1] = x[1] - 1.

pbc = PeriodicBoundary()

mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)

q = TrialFunction(V)
p = TrialFunction(V)
v = TestFunction(V)
q_new = Function(V)
p_new = Function(V)

f1 = Expression("1*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=2)
q0 = project(f1,V)

f2 = Expression("0",degree=2)
p0 = project(f2,V)

dt = 0.01
c = 0.01
theta = 0.5

# Crank-Nicolson discretization
aq = q*v*dx + pow(dt,2)*pow(theta,2)*inner(grad(q),grad(v))*dx
Lq = q0*v*dx - pow(dt,2)*theta*(1-theta)*inner(grad(q0),grad(v))*dx + dt*p0*v*dx #add source term if necessary

ap = p*v*dx
Lp = p0*v*dx - dt*theta*inner(grad(q_new),grad(v))*dx - dt*(1-theta)*inner(grad(q0),grad(v))*dx #add source term if necessary

Aq = assemble(aq)
Ap = assemble(ap)

vtkfile = File('results/solution.pvd')

for i in range(0,500):
	print(i)
	bq = assemble(Lq)
	solve(Aq,q_new.vector(),bq)

	bp = assemble(Lp)
	solve(Ap,p_new.vector(),bp)

	q0.assign(q_new)
	p0.assign(p_new)

	vtkfile << (q_new,i*dt)
