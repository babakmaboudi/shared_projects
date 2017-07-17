from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

class PeriodicBoundary(SubDomain):
	def inside(self,x,on_boundary):
		return bool( x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary )

	def map(self,x,y):
		y[0] = x[0] - 1.0

pbc = PeriodicBoundary()

mesh = UnitIntervalMesh(500)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)

q = TrialFunction(V)
p = TrialFunction(V)
v = TestFunction(V)
q_new = Function(V)
p_new = Function(V)

f1 = Expression("1*exp(-(pow(x[0]-0.5, 2)) / 0.02)",degree=2)
q0 = project(f1,V)
p0 = Function(V)

dt = 0.01
c = 0.01

aq = q*v*dx
Lq = q0*v*dx + dt*p0*v*dx

ap = p*v*dx
Lp = p0*v*dx - dt*c*inner( grad(q_new) , grad(v) )*dx

Aq = assemble(aq)
Ap = assemble(ap)

energy = c*q0.dx(0)*q0.dx(0)*dx + p0*p0*dx

vtkfile = File('results/solution.pvd')

for i in range(0,500):
#	print(i)
	bq = assemble(Lq)
	solve(Aq,q_new.vector(),bq)

	bp = assemble(Lp)
	solve(Ap,p_new.vector(),bp)

	q0.assign(q_new)
	p0.assign(p_new)
#	e = assemble(energy)
#	print(e)
	vtkfile << (q0,i*dt)

#plot(q_new)
#plt.show()
