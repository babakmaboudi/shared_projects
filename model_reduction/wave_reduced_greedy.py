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

class Wave_Reduced:
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
		self.MAX_ITER = 5000
		self.dt = 0.01

		# load reduced basis
		self.Phi = np.load( "RB.dat" )
		
		self.N = np.int_( self.Phi.shape[0]/2 );
		self.K = np.int_( self.Phi.shape[1]/2 );

	def initiate_fem( self ):
		#define mesh
		self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
#		self.mesh = UnitIntervalMesh(3)
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

		#define the weak form with the symplecti Euler
		#mass matrix
		self.aq = inner(self.q,self.v)*dx
		self.ap = inner(self.p,self.v)*dx
		self.Aq = assemble(self.aq)
		self.Ap = assemble(self.ap)

		#stiffness matrix
		self.kq = inner(self.p,self.v)*dx
		self.kp = -inner(sigma(self.q,self.lambda_,self.mu,self.d),epsilon(self.v))*dx
		self.Kq = assemble(self.kq)
		self.Kp = assemble(self.kp)

		#force vector
		self.Lq = inner(self.T,self.v)*dx
		self.bq = assemble(self.Lq)
		self.bc.apply( self.Aq , self.bq )

		self.Lp = inner(self.f,self.v)*dx
		self.bp = assemble(self.Lp)
		self.bc.apply( self.Ap , self.bp )

		#construct matrices
		self.Aq_mat = np.matrix( self.Aq.array() )
		self.Ap_mat = np.matrix( self.Ap.array() )
		self.Kp_mat = np.matrix( self.Kp.array() )
		self.A_inv = np.linalg.inv( self.Ap_mat )
		self.AinvK = self.A_inv*self.Kp_mat

		temp1 = np.concatenate( ( np.zeros([self.N,self.N]) , np.eye(self.N) ) , 1 )
		temp2 = np.concatenate( ( self.AinvK , np.zeros([self.N,self.N]) ) , 1 )
		self.L = np.concatenate( (temp1,temp2), 0 )

#		self.bc.apply(self.Ap,self.bp)
		self.c = np.matrix( self.bp.array() )
		self.c = np.transpose( self.c )
		self.cp = self.A_inv*self.c
		self.c_vec = np.concatenate( (np.zeros([self.N,1]),self.cp) )

		
		#construct reduced matrices
		self.Lr = np.transpose( self.Phi )*self.L*self.Phi
		self.cr = np.transpose( self.Phi )*self.c_vec

		self.L11 = self.Lr[0:self.K,0:self.K]
		self.L12 = self.Lr[0:self.K,self.K:2*self.K]
		self.L21 = self.Lr[self.K:2*self.K,0:self.K]
		self.L22 = self.Lr[self.K:2*self.K,self.K:2*self.K]
		self.L1 = self.Lr[0:self.K,:]
		self.L2 = self.Lr[self.K:2*self.K,:]
		
		self.c1 = self.cr[0:self.K,:]
		self.c2 = self.cr[self.K:2*self.K,:]

	def symplectic_euler( self ):
		vqr0 = np.zeros(self.c1.shape)
		vpr0 = np.zeros(self.c2.shape)
		
		inv_factor = np.eye(self.K) - self.dt*self.L11
		inv_factor = np.linalg.inv( inv_factor )

		print(inv_factor.shape)
		print(self.L1.shape)
		print(self.c1.shape)
		c1_factor = self.dt*inv_factor*self.c1

		self.snap = np.zeros( [2*self.K,1] )

		for i in range(0,self.MAX_ITER):
			print(i)

			vqr0 = inv_factor*vqr0 + self.dt*inv_factor*self.L12*vpr0 + c1_factor
			vpr0 = vpr0 + self.dt*self.L21*vqr0 + self.dt*self.L22*vpr0 + self.dt*self.c2
			temp = np.concatenate( (vqr0,vpr0),0 )
			self.snap = np.concatenate( (self.snap,temp),1 )

	def save_vtk_result( self ):
		f = Function(self.V)

		vtkfile = File('results_reduced/solution.pvd')
		for i in range(0,self.MAX_ITER+1):
			coef = self.Phi*self.snap[:,i]
			coef = coef[0:self.N,:]
			f.vector().set_local( coef )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)
