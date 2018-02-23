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
		self.w = self.L/40
		self.mu = 1
		self.rho = 1
		self.delta = self.W/self.L
		self.gamma = 0.4*self.delta**2
		self.beta = 1.25
		self.lambda_ = self.beta
		self.g = self.gamma

		# numerical parameters
		self.MAX_ITER = 8000
		self.dt = 0.0001

		#load reduced basis
		self.Phi = np.load( "RB.dat" )
		self.X = np.load( "X_mat_red.dat" )
		self.Phi = np.matrix(self.Phi)
		self.TJJ = np.load("JJ_trans.dat")
		self.TJJ = np.matrix(self.TJJ)

		self.N = np.int_( self.Phi.shape[0]/2 );
		self.K = np.int_( self.Phi.shape[1]/2 );

	def initiate_fem( self ):
		## Parameters
		#R = self.W/4
		#r = 0.08
		#t = self.W
		#x = self.W/2+R*cos(float(t) / 180 * pi)
		#y = self.W/2
		#z = R*sin(t)
		
		## Create geometry
		#s1 = mshr.Sphere(Point(x+self.L-3/2*self.W, y, z), r)
		#s2 = mshr.Sphere(Point(x, y, z), r)

		#b1 = mshr.Box(Point(0, 0, 0), Point(self.L, self.W, self.W))
		#b2 = mshr.Box(Point(self.L/2-self.w, 0, self.W/2), Point(self.L/2+self.w, self.W, self.W))
		#geometry = b1 - s1 -s2
		#geometry2 = b1 - b2
		
		# Create and store mesh
		#self.mesh = mshr.generate_mesh(geometry,10) # use geometry1 or geometry2
		self.mesh= Mesh('meshes/cracked_beam_size_field.xml')
		
		#File('results/cracked_beam.pvd') << self.mesh
		#File('results/cracked_beam.xml') << self.mesh
		
		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)
		
		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)
		
		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))
		
		##define mesh
##		self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
		#self.mesh = Mesh('results/cracked_beam.xml')

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
		self.Kmat = np.matrix( Kq.array() )

		#define the force term
		c = np.matrix( bq.array() )
		self.cp = np.transpose( c )

		#define reduced matrices
		temp1 = np.concatenate([np.zeros([self.N,self.N]),self.M_inv],1)
		temp2 = np.concatenate([self.Kmat,np.zeros([self.N,self.N])],1)

		mat = np.concatenate([temp1,temp2],0)
		
		Jk = construct_Jk(self.K)
		Jk = np.matrix(Jk)
		Jn = construct_Jk(self.N)
		Jn = np.matrix(Jn)

		X = self.X
		Jtn = X*Jn*X
		A = self.Phi
		A_plus = np.transpose(Jk)*np.transpose(A)*X*Jtn
		J = A_plus*X*Jn*X*np.transpose(A_plus)

#		TJJ = self.compute_JJ_transformation(J)
		TJJ = self.TJJ
		T_inv = np.linalg.inv(TJJ)

		J = TJJ*J*np.transpose(TJJ)

		K = -Jn*mat
		K = np.transpose(T_inv)*np.transpose(A)*K*A*T_inv

		K = 0.5*(K+np.transpose(K))
		self.Lr = Jk*K

		f_term = np.concatenate([np.zeros([self.N,1]),self.cp],0)
		self.f_termr = TJJ*A_plus*X*f_term
		self.TJJ = TJJ

	def test( self ):
		q0r = np.zeros([2*self.K,1])

		for i in range(0,1000):
			q0r = self.dt/100*self.Lr*q0r + self.dt/100*self.f_termr
			print(np.max(np.abs(q0r)))

	def mid_point( self ):
		vtkfile = File('results_reduced/solution.pvd')
		f = Function(self.V)

		K = self.f_termr.shape[0]
		x0 = np.zeros([K,1])

		inv_factor = np.linalg.inv( np.eye(K) - self.dt/2*self.Lr )
		L = inv_factor*( np.eye(K) + self.dt/2*self.Lr )

		c_factor = self.dt*inv_factor*self.f_termr
		
		self.snap_x0 = np.zeros(x0.shape)
		self.energy = self.compute_energy(x0)

		for i in range(0,self.MAX_ITER):
			print(i)

			x0 = L*x0 + c_factor

			#print( self.compute_energy( x0 ) )
			full = self.Phi*np.linalg.inv(self.TJJ)*x0
			f.vector().set_local( full[0:self.N,:] )
			if np.mod(i,10) == 0:
				self.snap_x0 = np.concatenate((self.snap_x0,x0),1)
				Energy = self.compute_energy(x0)
				self.energy = np.concatenate((self.energy,Energy),0)

			if np.mod(i,125) == 0:
				vtkfile << (f,i*self.dt)

	def symplectic_euler( self ):
		vtkfile = File('results_reduced/solution.pvd')
		f = Function(self.V)
#		q0 = np.zeros([self.K,1])
#		p0 = np.zeros([self.K,1]
		x0 = np.zeros( [2*self.K,1] )
		self.dt = self.dt

		L1 = self.Lr[0:2*self.K,0:self.K]
		L1 = np.concatenate( (L1,np.zeros([2*self.K,self.K])) , 1 )
		L2 = self.Lr[0:2*self.K,self.K:2*self.K]
		L2 = np.concatenate( (np.zeros([2*self.K,self.K]),L2) , 1 )
		inv_factor = np.linalg.inv( np.eye(2*self.K) - self.dt*L2 )
		factor = np.eye(2*self.K) + self.dt*L1

		for i in range(0,100000):
			print(i)
			x0 = inv_factor*factor*x0 + self.dt*inv_factor*self.f_termr

			print( self.compute_energy( x0 ) )
			if np.mod(i,1000) == 0:
				full = self.Phi*np.linalg.inv(self.TJJ)*x0
				f.vector().set_local( full[0:self.N,:] )
				vtkfile << (f,i*self.dt)

	def symplectic_euler2( self ):
		vtkfile = File('results/solution.pvd')
		f = Function(self.V)
		self.dt = self.dt/100

		q0 = np.zeros([self.K,1])
		p0 = np.zeros([self.K,1])

		L11 = self.Lr[0:self.K,0:self.K]
		L12 = self.Lr[0:self.K,self.K:2*self.K]
		L21 = self.Lr[self.K:2*self.K,0:self.K]
		L22 = self.Lr[self.K:2*self.K,self.K:2*self.K]

		if2 = np.eye(self.K) - self.dt*L22
		if2 = np.linalg.inv(if2)

		f1 = self.f_termr[0:self.K]
		f2 = self.f_termr[self.K:2*self.K]

		for i in range(0,100000):
			p0 = if2*p0 + self.dt*if2*L21*q0 + self.dt*if2*f2
			q0 = q0 + self.dt*L11*q0 + self.dt*L12*p0 + self.dt*f1

			print( np.max( q0 ) )		

	def stormer_verlet( self ):
		vtkfile = File('results_reduced/solution.pvd')
		f = Function(self.V)
		self.dt = self.dt/2

		q0 = np.zeros([self.K,1])
		p0 = np.zeros([self.K,1])

		L11 = self.Lr[0:self.K,0:self.K]
		L12 = self.Lr[0:self.K,self.K:2*self.K]
		L21 = self.Lr[self.K:2*self.K,0:self.K]
		L22 = self.Lr[self.K:2*self.K,self.K:2*self.K]

		if1 = np.eye(self.K) - self.dt/2*L11
		if1 = np.linalg.inv(if1)

		if2 = np.eye(self.K) - self.dt/2*L22
		if2 = np.linalg.inv(if2)

		f1 = self.f_termr[0:self.K]
		f2 = self.f_termr[self.K:2*self.K]

		term1 = np.eye(self.K) + self.dt/2*L11

		self.snap_Q = np.zeros( self.cp.shape )
		self.snap_P = np.zeros( self.cp.shape )
		
		x0 = np.concatenate((q0,p0),0)
		self.snap_x0 = np.zeros(x0.shape)
		self.energy = self.compute_energy(x0)

		for i in range(0,self.MAX_ITER):
			print(i)
			p0 = if2*p0 + self.dt/2*if2*L21*q0 + self.dt/2*if2*f2
#			q0 = q0 + self.dt/2*L11*q0 + self.dt/2*L12*p0 + self.dt/2*f1
			q0 = if1*q0 + self.dt/2*if1*L11*q0 + + self.dt*if1*L12*p0 + self.dt*if1*f1
			p0 = p0 + self.dt/2*L21*q0 + self.dt/2*L22*p0 + self.dt/2*f2
			#print(np.max(np.abs(q0)))
			#if np.mod(i,250) == 0:
			x0 = np.concatenate((q0,p0),0)
			full = self.Phi*np.linalg.inv(self.TJJ)*x0
			f.vector().set_local( full[0:self.N,:] )
			
			if np.mod(i,10) == 0:
				self.snap_x0 = np.concatenate((self.snap_x0,x0),1)
				Energy = self.compute_energy(x0)
				self.energy = np.concatenate((self.energy,Energy),0)

			if np.mod(i,125) == 0:
				vtkfile << (f,i*self.dt)

	def compute_JJ_transformation(self,J):
		N = int(J.shape[0]/2)
		Q = np.eye(2*N)
		T = np.eye(2*N)
		T = np.matrix(T)

		temp = np.arange(64).reshape(8,8)

		for i in range(0,N):
			print(i)
			p = N+i

			idxc = np.argmax(np.abs( J[i:2*N,i:2*N] ), axis = 0)
			vc = np.max(np.abs( J[i:2*N,i:2*N] ), axis = 0)
			idxt = np.argmax(vc)

			ii = idxc[0,idxt]+i
			jj = idxt+i
			
			if(jj>ii):
				temp = jj
				jj = ii
				ii = temp

			Q = np.eye(2*N)
			Q = np.matrix(Q)
			Q[ii,ii] = 0
			Q[p,p] = 0
			Q[p,ii] = 1
			Q[ii,p] = 1
			T = Q*T
			J = Q*J*np.transpose(Q)

			Q = np.eye(2*N)
			Q = np.matrix(Q)
			Q[jj,jj] = 0
			Q[i,i] = 0
			Q[i,jj] = 1
			Q[jj,i] = 1
			T = Q*T
			J = Q*J*np.transpose(Q)

			Q = np.eye(2*N)
			Q[p,p] = -1/J[p,i]
			T = Q*T
			J = Q*J*np.transpose(Q)

			for j in range(i+1,2*N):
				if(j!=p):
					Q = np.eye(2*N)
					Q[j,p] = -J[j,i]/J[p,i]
					T = Q*T
					J = Q*J*np.transpose(Q)

			for j in range(i+1,2*N):
				if(j!=p):
					Q = np.eye(2*N)
					Q[j,i] = -J[j,p]
					T = Q*T
					J = Q*J*np.transpose(Q)

		return T

	def compute_energy( self , x0 ):
		Jk = construct_Jk(self.K)
		E = np.transpose(x0)*Jk*self.Lr*x0+np.transpose(x0)*Jk*self.f_termr 
		return E

	def save_snapshots( self ):
		self.snap_x0.dump("snap_x0.dat")
		#self.snap_Q.dump("snap_Q150.dat")
		#self.snap_P.dump("snap_P150.dat")
		
	def compute_error(self):
		Phi = np.matrix(np.load( "RB.dat" ))
		X = np.matrix(np.load( "X_mat.dat" ))
		
		z = np.concatenate((np.matrix(np.load("snap_Q.dat")),np.matrix(np.load("snap_P.dat"))),0)
		y = np.matrix(np.load("snap_x0.dat"))
		
		self.l2err = np.linalg.norm(z - Phi*y, axis=0)
		self.werr = np.linalg.norm(X*(z - Phi*y), axis=0)
		
		self.l2err.dump("l2err.dat")
		self.werr.dump("werr.dat")
		self.energy.dump("energy.dat")
		
		fig = plt.figure(1)
		plt.subplot(311)
		plt.plot(self.l2err)

		plt.subplot(312)
		plt.plot(self.werr)
		
		plt.subplot(313)
		plt.plot(self.energy)
		fig.savefig('errors-energy',dpi = fig.dpi)
		
		plt.show()
		#print(self.l2err.shape)
		#plt.plot(self.werr)
		#plt.show()
		#,range(0,self.MAX_ITER),self.werr)
		