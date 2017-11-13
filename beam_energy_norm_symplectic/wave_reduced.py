from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

import numpy as np
import scipy.linalg as scla

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < DOLFIN_EPS

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u,lambda_,mu,d):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

def construct_Jk(K):
	Jk = np.zeros([2*K,2*K])
	for i in range(0,K):
		Jk[i,K+i] = 1
		Jk[K+i,i] = -1
	return Jk

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
		self.MAX_ITER = 100000
		self.dt = 0.0001

		#load reduced basis and weight matrix
		self.Phi = np.load( "RB.dat" )
		self.X_mat = np.load( "X_mat.dat" )
#		self.X_mat = np.load( "X_mat_eye.dat" )
		self.X_mat = np.matrix(self.X_mat)

		self.N = np.int_( self.Phi.shape[0]/2 );
		self.K = np.int_( self.Phi.shape[1]/2 );

	def initiate_fem( self ):
		#define mesh
		self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)

		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)

		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)

		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))

		#define functions
		q = TrialFunction(self.V)
		p = TrialFunction(self.V)
		self.d = q.geometric_dimension()
		v = TestFunction(self.V)

		#define mass and stifness matrices
		kq = inner(p,v)*dx
		Lq = inner(self.T,v)*dx
		aq = inner(q,v)*dx
		Kq, self.bq = assemble_system( kq, Lq, self.bc )
		self.Aq, dummy = assemble_system( aq, Lq, self.bc )
		Kq_mat = np.matrix( Kq.array() ) 
		Aq_mat = np.matrix( self.Aq.array() )


		kp = -inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(v))*dx
		Lp = inner(self.f,v)*dx
		ap = inner(p,v)*dx
		Kp, self.bp = assemble_system( kp, Lp, self.bc )
		self.Ap, dummy = assemble_system( ap, Lp, self.bc )
		Kp_mat = np.matrix( Kp.array() )
		Ap_mat = np.matrix( self.Ap.array() )

		A_inv = np.linalg.inv( Ap_mat )
		self.L = A_inv*Kp_mat

#		self.bc.apply(self.Ap,self.bp)
		c = np.matrix( self.bp.array() )
		c = np.transpose( c )
		self.cp = A_inv*c

		#defining large vectors and matrices
		temp1 = np.concatenate( ( np.zeros([self.N,self.N]) , np.eye(self.N) ) , 1 )
		temp2 = np.concatenate( ( self.L , np.zeros([self.N,self.N]) ) , 1 )
		L_mat = np.concatenate( (temp1,temp2), 0 )
		c_vec = np.concatenate( (np.zeros([self.N,1]),self.cp) )

		#construct reduced matrices
#		self.X_mat = np.matrix(self.X_mat)
#		proj_mat = self.X_mat*self.X_mat*self.Phi
#		self.Lr = np.transpose( proj_mat )*L_mat*proj_mat
#		self.cr = np.transpose( proj_mat )*c_vec

#		self.Phi = np.matrix(self.Phi)
#		self.Lr = np.transpose(self.Phi)*L_mat*self.Phi
#		self.cr = np.transpose(self.Phi)*c_vec

#		self.Phi = np.matrix(self.Phi)
#		self.Lr = np.transpose(self.Phi)*self.X_mat*self.X_mat*L_mat*self.Phi
#		self.cr = np.transpose(self.Phi)*self.X_mat*self.X_mat*c_vec

		Jk = construct_Jk(self.K)
		Jk = np.matrix(Jk)
		Jn = construct_Jk(self.N)
		Jn = np.matrix(Jn)

		self.Phi = np.matrix(self.Phi)
		self.X_mat = np.matrix(self.X_mat)
		Phi_cross = np.transpose(Jk)*np.transpose(self.Phi)*self.X_mat*self.X_mat*Jn

		self.Lr = Phi_cross*self.X_mat*self.X_mat*L_mat*self.Phi
		self.cr = Phi_cross*self.X_mat*self.X_mat*c_vec

#		Aplus = np.transpose(Jk)*np.transpose(self.Phi)*self.X_mat*self.X_mat*Jn*self.X_mat
#		S = Aplus*self.X_mat*Jn*self.X_mat*np.transpose(Aplus)
#		Lin = np.transpose(self.Phi)*np.transpose(Jn)*L_mat*self.Phi

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

		c1_factor = self.dt*inv_factor*self.c1

		self.snap = np.zeros( [2*self.K,1] )

		for i in range(0,self.MAX_ITER):
			print(i)

			vqr0 = inv_factor*vqr0 + self.dt*inv_factor*self.L12*vpr0 + c1_factor
			vpr0 = vpr0 + self.dt*self.L21*vqr0 + self.dt*self.L22*vpr0 + self.dt*self.c2
			temp = np.concatenate( (vqr0,vpr0),0 )
			self.snap = np.concatenate( (self.snap,temp),1 )

	def mid_point( self ):
		vtkfile = File('results_reduced/solution.pvd')
		f = Function(self.V)

		K = self.cr.shape[0]
		x0 = np.zeros([K,1])

		inv_factor = np.linalg.inv( np.eye(K) - self.dt/2*self.Lr )
		L = inv_factor*( np.eye(K) + self.dt/2*self.Lr )

		c_factor = self.dt*inv_factor*self.cr
		

		for i in range(0,self.MAX_ITER):
#			print(i)

			x0 = L*x0 + c_factor
#			x0 = self.dt*self.Lr*x0 + self.dt*self.cr

			print(np.linalg.norm(x0))
#			f.vector().set_local( x0[0:N,:] )
			if np.mod(i,1000) == 0:
				temp = self.Phi*x0
				temp = temp[0:528,:]
				f.vector().set_local( temp )
				vtkfile << (f,i*self.dt)

	def save_vtk_result( self ):
		f = Function(self.V)

		vtkfile = File('results_reduced/solution.pvd')
		for i in range(0,self.MAX_ITER+1):
			coef = self.Phi*self.snap[:,i]
			coef = coef[0:self.N,:]
			f.vector().set_local( coef )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)


