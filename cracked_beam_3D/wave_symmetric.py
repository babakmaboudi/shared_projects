from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import mshr as mshr

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
		self.w = self.L/40
		self.mu = 1
		self.rho = 1
		self.delta = self.W/self.L
		self.gamma = 0.4*self.delta**2
		self.beta = 1.25
		self.lambda_ = self.beta
		self.g = self.gamma

		# numerical parameters
		self.MAX_ITER = 1000
		self.dt = 0.00006

	def initiate_fem( self ):
		#define mesh
		#self.mesh= Mesh('mesh.xml')
#		self.mesh = refine(self.mesh)

		# Parameters
		R = self.W/4
		r = 0.08
		t = self.W
		x = self.W/2+R*cos(float(t) / 180 * pi)
		y = self.W/2
		z = R*sin(t)
		
		# Create geometry
		s1 = mshr.Sphere(Point(x+self.L-3/2*self.W, y, z), r)
		s2 = mshr.Sphere(Point(x, y, z), r)

		b1 = mshr.Box(Point(0, 0, 0), Point(self.L, self.W, self.W))
		b2 = mshr.Box(Point(self.L/2-self.w, 0, self.W/2), Point(self.L/2+self.w, self.W, self.W))
		geometry = b1 - s1 -s2
		#geometry2 = b1 - b2
		
		# Create and store mesh
		self.mesh = mshr.generate_mesh(geometry,10) # use geometry1 or geometry2
		
		File('results/cracked_beam.pvd') << self.mesh
		File('results/cracked_beam.xml') << self.mesh
		
		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)
		
		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)
		
		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))
		
		##define function space
		#self.V = VectorFunctionSpace(self.mesh, 'P', 1)

		##define dirichlet boundary
		#self.bc = DirichletBC(self.V, Constant((0, 0)), clamped_boundary)

		##define right hand side function
		#self.f = Constant((0, -self.rho*self.g))
		#self.T = Constant((0, 0))	

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

	def stormer_verlet( self ):
		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		N = self.cp.shape[0]
		q0 = np.zeros([N,1])
		p0 = np.zeros([N,1])

		self.snap_Q = np.zeros( self.cp.shape )
		self.snap_P = np.zeros( self.cp.shape )

		for i in range(0,self.MAX_ITER):
			print(i)
			q0 = q0 + self.dt/2*self.M_inv*p0
			p0 = p0 + self.dt*self.K*q0 + self.dt*self.cp
			q0 = q0 + self.dt/2*self.M_inv*p0

			f.vector().set_local( q0 )

			if np.mod(i,10) == 0:
				self.snap_Q = np.concatenate((self.snap_Q,q0),1)
				self.snap_P = np.concatenate((self.snap_P,p0),1)

			if np.mod(i,125) == 0:
				vtkfile << (f,i*self.dt)

		print( np.linalg.cond(self.M_inv) )
		print( np.linalg.cond(self.K) )
		print( N )

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

		Xsqrt = scla.sqrtm(X_mat)
		Xsqrt = np.matrix( np.real(Xsqrt) )
		Xsqrt.dump("X_mat.dat")

		N = self.cp.shape[0]
		temp = np.eye(2*N)
		temp.dump("X_mat_eye.dat")

	def save_snapshots( self ):
		self.snap_Q.dump("snap_Q.dat")
		self.snap_P.dump("snap_P.dat")
