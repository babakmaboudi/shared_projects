from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pylab as plt

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 2)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=2)

a = inner(grad(u), grad(v))*dx
L = f*v*dx

a_a = assemble(a)
#anp = np.matrix( M.array() )

L_a = assemble(L)

bc.apply(a_a,L_a)

anp = np.matrix( a_a.array() )
Lnp = np.matrix( L_a.array() )
Lnp = np.transpose(Lnp)

coef = np.linalg.solve(anp,Lnp)

u = Function(V)
u.vector().set_local( coef )

file = File("poisson.pvd")
file << u
