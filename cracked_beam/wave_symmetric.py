from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
from mshr import *

import numpy as np
import scipy.linalg as scla

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < DOLFIN_EPS

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u,lambda_,mu,d):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

class Wave:
	def __init__( self ):
		# physical parameters
		self.L = 1
		self.W = 0.2
		self.w = self.L/100
		self.mu = 1
		self.rho = 1
		self.delta = self.W/self.L
		self.gamma = 0.4*self.delta**2
		self.beta = 1.25
		self.lambda_ = self.beta
		self.g = self.gamma

		# numerical parameters
		self.MAX_ITER = 1000
		self.dt = 0.01

	def initiate_fem( self ):
		#define mesh
		#self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
		
		b1 = Box(Point(0, 0, 0), Point(self.L, self.W, self.W))
		b2 = Box(Point(self.L/2-self.w, 0, self.W/2), Point(self.L/2+self.w, self.W, self.W))
		geometry = b1 - b2
		
		# Create and store mesh
		self.mesh = generate_mesh(geometry,10)
		File('results/cracked_beam.pvd') << self.mesh
		
		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)
		
		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)
		
		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))

		#define functions
		q = TrialFunction(self.V)
#		p = TrialFunction(self.V)
		self.d = q.geometric_dimension()
		v = TestFunction(self.V)

		aq = inner(q,v)*dx
		kq = -inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(v))*dx
		Lq = inner(self.f,v)*dx

		Kq, bq = assemble_system( kq, Lq, self.bc )
		self.Aq, dummy = assemble_system( aq, Lq, self.bc )

		#define the mass and stiffness matrices
		M = np.matrix( self.Aq.array() )
		self.M_inv = np.linalg.inv(M)
		self.K = np.matrix( Kq.array() )

		#define the force term
		c = np.matrix( bq.array() )
		self.cp = np.transpose( c )

	def symplectic_euler( self ):

		vq0 = np.zeros(self.cp.shape)
		vp0 = np.zeros(self.cp.shape)

		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		self.snap_Q = np.zeros( self.cp.shape )
		self.snap_P = np.zeros( self.cp.shape )

		for i in range(0,self.MAX_ITER):
			print(i)

			vq0 = vq0 + self.dt*self.M_inv*vp0
			
			vp0 = vp0 + self.dt*self.K*vq0 + self.dt*self.cp

			self.snap_Q = np.concatenate( (self.snap_Q,vq0) , 1 )
			self.snap_P = np.concatenate( (self.snap_P,vp0) , 1 )
		
			f.vector().set_local( vq0 )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def mid_point( self ):
		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		N = self.cp.shape[0]
		x0 = np.zeros([2*N,1])

		temp1 = np.concatenate([np.zeros([N,N]),self.M_inv],1)
		temp2 = np.concatenate([self.K,np.zeros([N,N])],1)

		mat = np.concatenate([temp1,temp2],0)

		inv_factor = np.linalg.inv( np.eye(2*N) - self.dt/2*mat )
		L = inv_factor*( np.eye(2*N) + self.dt/2*mat )

		c_factor = self.dt*inv_factor*np.concatenate([np.zeros([N,1]),self.cp],0)

#		energy_mat = np.concatenate([-temp2,temp1],0)
		

		self.snap_Q = np.zeros( self.cp.shape )
		self.snap_P = np.zeros( self.cp.shape )

		for i in range(0,self.MAX_ITER):
			print(i)

			x0 = L*x0 + c_factor
#			print(np.transpose(x0)*energy_mat*x0 - np.transpose(x0) * np.concatenate([np.zeros([N,1]),self.cp],0))

			self.snap_Q = np.concatenate( (self.snap_Q,x0[0:N,:]) , 1 )
			self.snap_P = np.concatenate( (self.snap_P,x0[N:2*N,:]) , 1 )

#			print(np.linalg.norm(x0))
			f.vector().set_local( x0[0:N,:] )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def generate_X_matrix( self ):
		q = TrialFunction(self.V)
		f = Function(self.V)
		v = v = TestFunction(self.V)
		k = inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(v))*dx
		L = inner(f,v)*dx

		K, dummy = assemble_system( k, L, self.bc )
		Kq_mat = np.matrix( K.array() )

		p = TrialFunction(self.V)
		k = inner(p,v)*dx

		K, dummy = assemble_system( k, L, self.bc )
		Kp_mat = np.matrix( K.array() )

		temp1 = np.concatenate( [Kq_mat,np.zeros(Kq_mat.shape)] , 1 )
		temp2 = np.concatenate( [np.zeros(Kp_mat.shape),self.M_inv] , 1 )
		X_mat = np.concatenate( [temp1,temp2] , 0 )

#		N = self.cp.shape[0]
#
#		temp1 = np.concatenate([np.zeros([N,N]),self.M_inv],1)
#		temp2 = np.concatenate([self.K,np.zeros([N,N])],1)
#
#		X_mat = np.concatenate([-temp2,temp1],0)
		Xsqrt = scla.sqrtm(X_mat)
		Xsqrt = np.matrix( np.real(Xsqrt) )
		Xsqrt.dump("X_mat.dat")

		N = self.cp.shape[0]
		temp = np.eye(2*N)
		temp.dump("X_mat_eye.dat")
		


#		X_mat = np.concatenate( (Kmat,np.zeros(Kmat.shape)) , 1 )
#		temp = np.concatenate( (np.zeros(Kmat.shape),np.eye(Kmat.shape[0])) , 1 )
#		X_mat = np.concatenate( (X_mat,temp) , 0 )
#
#		Xsqrt = scla.sqrtm(X_mat)
#		Xsqrt = np.matrix( Xsqrt )

	def save_snapshots( self ):
		self.snap_Q.dump("snap_Q.dat")
		self.snap_P.dump("snap_P.dat")
