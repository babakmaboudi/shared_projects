from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)
V1d = FunctionSpace(mesh, 'P', 1)

tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

q = TrialFunction(V)
p = TrialFunction(V)
d = q.geometric_dimension()
v = TestFunction(V)
q_new = Function(V)
p_new = Function(V)

f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))

q0 = Function(V)
p0 = Function(V)

dt = 0.1
theta = 0.5

aq = inner(q,v)*dx + pow(theta,2)*pow(dt,2)*inner( sigma(q) , epsilon(v) )*dx
Lq = inner(q0,v)*dx - pow(dt,2)*theta*(1-theta)*inner( sigma(q0) , epsilon(v) )*dx + dt*inner(p0,v)*dx + pow(dt,2)*pow(theta,2)*inner(f,v)*dx + pow(dt,2)*theta*(1-theta)*inner(f,v)*dx

ap = inner(p,v)*dx
Lp = inner(p0,v)*dx - dt*theta*inner( sigma(q_new) , epsilon(v) )*dx - dt*(1-theta)*inner( sigma(q0) , epsilon(v) )*dx + dt*theta*inner(f,v)*dx + dt*(1-theta)*inner(f,v)*dx #+ dt*theta*inner(T, v)*ds + dt*(1-theta)*inner(T,v)*ds


Aq = assemble(aq)
Ap = assemble(ap)

energy = 0.5*inner(p0,p0)*dx + 0.5*inner(sigma(q0),epsilon(q0))*dx - inner( f,q0 )*dx

vtkfile = File('results_imp/solution.pvd')
vtkfile2 = File('results_imp/energy.pvd')

e_vec = np.array([0])

for i in range(0,400):
	print(i)
	bq = assemble(Lq)
	bc.apply(Aq,bq)
	solve(Aq,q_new.vector(),bq)

	bp = assemble(Lp)
	bc.apply(Ap,bp)
	solve(Ap,p_new.vector(),bp)
#	solve(ap == Lp , p_new)

	q0.assign(q_new)
	p0.assign(p_new)

#	magnitude = sqrt(dot(q_new, q_new))
#	magnitude = project(magnitude, V1d)

#	vtkfile << (magnitude,i*dt)
#	mat = np.matrix( q0.vector() )
#	print( np.max( np.abs(mat) ) )

	E = assemble(energy)
	e_vec = np.append(e_vec,E)
#	print(e_vec)
#	
#	if np.mod(i,10) == 0:
#		vtkfile << (q_new,i*dt)

plt.plot(e_vec)
plt.show()
