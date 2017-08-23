from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

import numpy as np

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < DOLFIN_EPS

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u,lambda_,mu,d):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
  
def energy(q0,p0,f,lambda_,mu,d):
    return 0.5*inner(p0,p0)*dx + 0.5*inner(sigma(q0,lambda_,mu,d),epsilon(q0))*dx - inner( f,q0 )*dx

class Wave:
	def __init__( self ):
		# physical parameters
		self.L = 1
		self.W = 0.2
		self.mu = 1
		self.rho = 1
		self.delta = self.W/self.L
		self.gamma = 0.4*self.delta**2
		self.beta = 1.25
		self.lambda_ = self.beta
		self.g = self.gamma

		# numerical parameters
		self.MAX_ITER = 500
		self.dt = 0.01
		self.theta = 0.5;

	def initiate_fem( self ):
		#define mesh
		self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)

		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)

		#define functions
		self.q = TrialFunction(self.V)
		self.p = TrialFunction(self.V)
		self.d = self.q.geometric_dimension()
		self.v = TestFunction(self.V)
		self.q_new = Function(self.V)
		self.p_new = Function(self.V)

		self.q0 = Function(self.V)
		self.p0 = Function(self.V)

		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))
		
		#initialize the energy vector
		self.e_vec = assemble(energy(self.q0,self.p0,self.f,self.lambda_,self.mu,self.d))

		#define the weak form with the strommer-verlet scheme 
		#self.aq = inner(self.q,self.v)*dx
		#self.Lq = inner(self.q0,self.v)*dx + self.dt/2*inner(self.p0,self.v)*dx
		#self.ap = inner(self.p,self.v)*dx
		#self.Lp = inner(self.p0,self.v)*dx -self.dt*inner(sigma(self.q_new,self.lambda_,self.mu,self.d),epsilon(self.v))*dx + self.dt*inner(self.f,self.v)*dx
		
		#define the weak form of implicit-midpoint scheme
		self.aq = inner(self.q,self.v)*dx + pow(self.theta,2)*pow(self.dt,2)*inner( sigma(self.q,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx
		self.Lq = inner(self.q0,self.v)*dx - pow(self.dt,2)*self.theta*(1-self.theta)*inner( sigma(self.q0,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx + self.dt*inner(self.p0,self.v)*dx + pow(self.dt,2)*pow(self.theta,2)*inner(self.f,self.v)*dx + pow(self.dt,2)*self.theta*(1-self.theta)*inner(self.f,self.v)*dx
		self.ap = inner(self.p,self.v)*dx
		self.Lp = inner(self.p0,self.v)*dx - self.dt*self.theta*inner( sigma(self.q_new,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx - self.dt*(1-self.theta)*inner( sigma(self.q0,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx + self.dt*self.theta*inner(self.f,self.v)*dx + self.dt*(1-self.theta)*inner(self.f,self.v)*dx #+ dt*theta*inner(T, v)*ds + dt*(1-theta)*inner(T,v)*ds

		self.Aq = assemble(self.aq)
		self.Ap = assemble(self.ap)
		self.Aq_mat = np.matrix( self.Aq.array() )
		self.Ap_mat = np.matrix( self.Ap.array() )

	def apply_bc_q( self ):
		self.bq = assemble(self.Lq)
		self.bc.apply(self.Aq,self.bq)
		self.bq_mat = np.matrix( self.bq.array() )
		self.bq_mat = np.transpose( self.bq_mat )

	def apply_bc_p( self ):
		self.bp = assemble(self.Lp)
		self.bc.apply(self.Ap,self.bp)
		self.bp_mat = np.matrix( self.bp.array() )
		self.bp_mat = np.transpose( self.bp_mat )

	def strommer_verlet( self ):
		#store initial data
		self.snap_Q = np.zeros((528,1))
		self.snap_P = np.zeros((528,1))

		#loop over time steps
		for i in range(0,self.MAX_ITER):
			print(i)
			self.apply_bc_q()
			coef = np.linalg.solve(self.Aq_mat,self.bq_mat)
			self.q_new.vector().set_local( coef )

			self.apply_bc_p()
			coef = np.linalg.solve(self.Ap_mat,self.bp_mat)
			self.p_new.vector().set_local( coef )

#			if np.mod(i,50) == 0:
			self.snap_P = np.c_[self.snap_P,coef]

			self.p0.assign( self.p_new )
			self.q0.assign( self.q_new )

			self.apply_bc_q()
			coef = np.linalg.solve(self.Aq_mat,self.bq_mat)
			self.q_new.vector().set_local( coef )

			self.q0.assign( self.q_new )
			
#			if np.mod(i,50) == 0:
			self.snap_Q = np.c_[self.snap_Q,coef]
			
	def implicit_midpoint( self ):
		#store initial data
		self.snap_Q = np.zeros((528,1))
		self.snap_P = np.zeros((528,1))

		#loop over time steps
		for i in range(0,self.MAX_ITER):
			print(i)
			self.apply_bc_q()
			coef = np.linalg.solve(self.Aq_mat,self.bq_mat)
			self.q_new.vector().set_local( coef )

			self.apply_bc_p()
			coef = np.linalg.solve(self.Ap_mat,self.bp_mat)
			self.p_new.vector().set_local( coef )

			self.p0.assign( self.p_new )
			self.q0.assign( self.q_new )

#			if np.mod(i,50) == 0:
			self.snap_P = np.c_[self.snap_P,coef]
#			if np.mod(i,50) == 0:
			self.snap_Q = np.c_[self.snap_Q,coef]

	def save_vtk_result( self ):
		f = Function(self.V)

		vtkfile = File('results/solution.pvd')
		for i in range(0,self.MAX_ITER+1):
			coef = self.snap_Q[:,i]
			f.vector().set_local( coef )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def save_snapshots( self ):
		self.snap_Q.dump("snap_Q.dat")
		self.snap_Q.dump("snap_P.dat")
