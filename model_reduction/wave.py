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
		self.MAX_ITER = 2000
		self.dt = 0.01

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

#		temp1 = np.concatenate( (self.Aq_mat , np.zeros(self.Aq_mat.shape)), 1 )
#		temp2 = np.concatenate( (np.zeros(self.Ap_mat.shape) , self.Ap_mat), 1 )
#		self.At = np.concatenate( (temp1,temp2) , 0 )

		#stiffness matrix
		self.kq = inner(self.p,self.v)*dx
		self.kp = -inner(sigma(self.q,self.lambda_,self.mu,self.d),epsilon(self.v))*dx
		self.Kq = assemble(self.kq)
		self.Kp = assemble(self.kp)
		self.Kq_mat = np.matrix( self.Kq.array() )
		self.Kp_mat = np.matrix( self.Kp.array() )

		temp1 = np.concatenate( (np.zeros(self.Kq_mat.shape),self.Kq_mat) , 1 )
		temp2 = np.concatenate( (self.Kp_mat,np.zeros(self.Kp_mat.shape)) , 1 )
		self.K = np.concatenate( (temp1,temp2) , 0 )

		#force vector
		self.Lq = inner(self.T,self.v)*dx
		self.bq = assemble(self.Lq)
		self.bc.apply( self.Aq , self.bq )
		self.Aq_mat = np.matrix( self.Aq.array() )

		self.Lp = inner(self.f,self.v)*dx
		self.bp = assemble(self.Lp)
		self.bc.apply( self.Ap , self.bp )
		self.Ap_mat = np.matrix( self.Ap.array() )

		self.A_inv = np.linalg.inv( self.Ap_mat )
		self.L = self.A_inv*self.Kp_mat
		
#		self.bc.apply(self.Ap,self.bp)
		self.c = np.matrix( self.bp.array() )
		self.c = np.transpose( self.c )
		self.cp = self.A_inv*self.c

	def apply_bc_q( self ):
		self.bq.set_local( self.q_rhs )
		self.bc.apply( self.Aq , self.bq )
		self.q_rhs = np.matrix( self.bq.array() )
		self.q_rhs = np.transpose( self.q_rhs )

	def apply_bc_p( self ):
		self.bp.set_local( self.p_rhs )
		self.bc.apply( self.Ap , self.bp )
		self.p_rhs = np.matrix( self.bp.array() )
		self.p_rhs = np.transpose( self.p_rhs )


	def symplectic_euler( self ):

		vq0 = np.zeros(self.c.shape)
		vp0 = np.zeros(self.c.shape)

		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		self.snap_Q = np.zeros( self.c.shape )
		self.snap_P = np.zeros( self.c.shape )


		for i in range(0,self.MAX_ITER):
			print(i)

			self.q_rhs = vq0 + self.dt*vp0
			self.apply_bc_q()		
			vq0 = self.q_rhs
			
			self.p_rhs = vp0 + self.dt*self.L*vq0 + self.dt*self.cp
			self.apply_bc_p()
			vp0 = self.p_rhs

#			self.snap_Q = np.concatenate( (self.snap_Q,vq0) , 1 )
#			self.snap_P = np.concatenate( (self.snap_P,vp0) , 1 )

			f.vector().set_local( vq0 )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def stormer_verlet( self ):
		vq0 = np.zeros(self.c.shape)
		vp0 = np.zeros(self.c.shape)

		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		for i in range(0,self.MAX_ITER):
			print(i)

			self.p_rhs = vp0 + self.dt/2*self.L*vq0 + self.dt/2*self.cp
			self.apply_bc_p()
			vp0 = self.p_rhs

			self.q_rhs = vq0 + self.dt*vp0
			self.apply_bc_q()		
			vq0 = self.q_rhs

			self.p_rhs = vp0 + self.dt/2*self.L*vq0 + self.dt/2*self.cp
			self.apply_bc_p()
			vp0 = self.p_rhs

			f.vector().set_local( vq0 )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def crank_nicolson_implicit( self ):
		N = self.L.shape[0]

		x = np.zeros([2*N,1])

		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		temp1 = self.dt/2*np.eye(N)
		temp2 = self.dt/2*self.L

		temp3 = np.concatenate( (np.eye(N),-temp1) , 1 )
		temp4 = np.concatenate( (-temp2,np.eye(N)) , 1 )

		mat1 = np.concatenate( (temp3,temp4) , 0 )
		mat1 = np.matrix(mat1)

		temp3 = np.concatenate( (np.eye(N),temp1) , 1 )
		temp4 = np.concatenate( (temp2,np.eye(N)) , 1 )

		mat2 = np.concatenate( (temp3,temp4) , 0 )
		mat2 = np.matrix(mat2)

		vec = np.concatenate( (np.zeros([N,1]),self.dt*self.cp) , o )

		for i in range(0,self.MAX_ITER):
			print(i)

			x = np.linalg.solve(mat1,mat2*x+vec)
	
			self.q_rhs = x[0:N]
			self.apply_bc_q
			self.p_rhs = x[N:2*N]
			self.apply_bc_p

			x = np.concatenate( (self.q_rhs,self.p_rhs) , 0 )

			f.vector().set_local(x[0:N])
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def crank_nicolson( self ):
		vq0 = np.zeros(self.c.shape)
		vp0 = np.zeros(self.c.shape)

		vtkfile = File('results/solution.pvd')
		f = Function(self.V)

		N = self.L.shape[0]

		mat1 = np.eye(N) - pow(self.dt,2)/4*self.L
		mat2 = np.eye(N) + pow(self.dt,2)/4*self.L

		e_vec = np.zeros( [self.MAX_ITER,1] )

		for i in range(0,self.MAX_ITER):
			print(i)

			b = self.dt*vp0 + mat2*vq0 + pow(self.dt,2)/2*self.cp
			self.q_rhs = np.linalg.solve(mat1,b)
			self.apply_bc_q()

			self.p_rhs = vp0 + self.dt/2*self.L*(self.q_rhs + vq0) + self.dt*self.cp
			self.apply_bc_p()

			vq0 = self.q_rhs
			vp0 = self.p_rhs

			e = self.compute_energy()
			e_vec[i] = e

			f.vector().set_local(vq0)
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

		plt.plot(e_vec)
		plt.show()

	def compute_energy( self ):
		q0 = Function( self.V )
		q0.vector().set_local( self.q_rhs )
		p0 = Function( self.V )
		p0.vector().set_local( self.p_rhs )

		energy = 0.5*inner(p0,p0)*dx + 0.5*inner(sigma(q0,self.lambda_,self.mu,self.d),epsilon(q0))*dx - inner( self.f,q0 )*dx
		E = assemble(energy)
		return E

	def save_snapshots( self ):
		self.snap_Q.dump("snap_Q.dat")
		self.snap_Q.dump("snap_P.dat")
