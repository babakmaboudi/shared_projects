from fenics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import mshr as mshr
import warnings

import numpy as np
import scipy.linalg as scla

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < DOLFIN_EPS

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u,lambda_,mu,d):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

def construct_J(K):
	return np.block([[np.zeros([K,K]),np.identity(K)],[-np.identity(K),np.zeros([K,K])]])

class wave:
	def __init__( self, MAX_ITER,dt,resultsfilename,meshfilename, simtype, reduced_sim, snapgap ):
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
		self.MAX_ITER = MAX_ITER
		self.dt = dt
		
		self.resultsfilename = resultsfilename
		self.meshfilename = meshfilename
		self.simtype = simtype
		self.reduced_sim = reduced_sim
		self.snapgap = snapgap

		# TODO load reduced basis if reduced simulation
		if simtype == "reduced":
			self.Phi = np.load( "RB.dat" )
			
			if self.reduced_sim == "Xweighted":
				self.X = np.load( "X_mat_red.dat" )
			else:
				self.X = np.eye(self.Phi.shape[0])
				
			self.Phi = np.matrix(self.Phi)
			self.TJJ = np.load("JJ_trans.dat")
			self.TJJ = np.matrix(self.TJJ)

			self.N = np.int_( self.Phi.shape[0]/2 );
			self.K = np.int_( self.Phi.shape[1]/2 );

	def initiate_fem( self ):
		#load mesh
		self.mesh= Mesh(self.meshfilename)
#		self.mesh = refine(self.mesh)

		#define function space
		self.V = VectorFunctionSpace(self.mesh, 'P', 1)
		
		#define dirichlet boundary
		self.bc = DirichletBC(self.V, Constant((0, 0, 0)), clamped_boundary)
		
		#define external force and traction
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))
		
		#define trial and test functions
		q = TrialFunction(self.V)
#		p = TrialFunction(self.V)
		v = TestFunction(self.V)
		self.d = q.geometric_dimension()

		aq = inner(q,v)*dx
		kq = inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(v))*dx
		Lq = inner(self.f,v)*dx

		Kq, bq = assemble_system( kq, Lq, self.bc )
		self.Aq, dummy = assemble_system( aq, Lq, self.bc )

		#define the mass and stiffness matrices
		M = np.matrix( self.Aq.array() )
		self.M_inv = np.linalg.inv(M)
		self.Kmat = np.matrix( Kq.array() )
		#self.K = np.matrix( Kq.array() )
		
		self.N = np.int_( M.shape[0] );

		#define the force term
		c = np.matrix( bq.array() )
		self.cp = np.transpose( c )

		# TODO define reduced matrices
		if self.simtype == "reduced":
			#temp1 = np.concatenate([np.zeros([self.N,self.N]),self.M_inv],1)
			#temp2 = np.concatenate([self.Kmat,np.zeros([self.N,self.N])],1)

			#mat = np.concatenate([temp1,temp2],0)
			
			mat = np.block([[np.zeros([self.N,self.N]),self.M_inv],[self.Kmat,np.zeros([self.N,self.N])]])
			
			Jk = np.matrix(construct_J(self.K))
			Jn = np.matrix(construct_J(self.N))
			
			#import pdb; pdb.set_trace()
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
			
	def test( self ):
		q0r = np.zeros([2*self.K,1])

		for i in range(0,1000):
			q0r = self.dt/100*self.Lr*q0r + self.dt/100*self.f_termr
			print(np.max(np.abs(q0r)))

	def mid_point( self ):
		vtkfile = File(self.resultsfilename)
		f = Function(self.V)
		
		if self.simtype == "reduced":
			A = self.Phi
			Atilde = self.X*A
			Jk = construct_J(self.K)
			Jn = construct_J(self.N)
			bJn = self.X*Jn*self.X
			A_plus = np.transpose(Jk)*np.transpose(A)*bJn
			Atilde_plus = np.transpose(Jk)*np.transpose(Atilde)*bJn
			self.bJk = Atilde_plus*bJn*np.transpose(Atilde_plus)
			
			x0 = Atilde_plus*self.X*np.zeros([2*self.N,1])
			self.Lmat = np.transpose(A)*np.block([[self.Kmat,np.zeros([self.N,self.N])],[np.zeros([self.N,self.N]),self.M_inv]])*A

			inv_factor = np.eye(2*self.K) - self.dt/2*self.bJk*self.Lmat
			factor = np.eye(2*self.K) + self.dt/2*self.bJk*self.Lmat
			#L = inv_factor*factor

			c_factor = self.dt*np.transpose(A)*np.concatenate([np.zeros([self.N,1]),self.cp],0)
		else:
			x0 = np.zeros([2*self.N,1])
			self.Lmat = np.block([[self.Kmat,np.zeros([self.N,self.N])],[np.zeros([self.N,self.N]),self.M_inv]])

			inv_factor = np.eye(2*self.N) - self.dt/2*np.matrix(construct_J(self.N))*self.Lmat
			factor = np.eye(2*self.N) + self.dt/2*np.matrix(construct_J(self.N))*self.Lmat
			#L = inv_factor*factor

			c_factor = self.dt*np.concatenate([np.zeros([self.N,1]),self.cp],0)
		
		self.snap_x0 = np.zeros(x0.shape)

		for i in range(0,self.MAX_ITER):
			print(i)
			x0 = np.linalg.solve(inv_factor,factor*x0 +c_factor)
			#x0 = L*x0 + c_factor
			#print( self.compute_energy( x0 ) )
			if self.simtype == "reduced":
				full = self.Phi*np.linalg.inv(self.TJJ)*x0
			else:
				full = x0
				
			f.vector().set_local( full[:self.N,:] )
			if np.mod(i,self.MAX_ITER/self.snapgap) == 0:
				self.snap_x0 = np.concatenate((self.snap_x0,x0),1)
				vtkfile << (f,i*self.dt)

		self.energy = self.compute_energy()

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
		vtkfile = File('results/solution.pvd')
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
		#N = self.cp.shape[0]
		#q0 = np.zeros([N,1])
		#p0 = np.zeros([N,1])

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
			q0 = q0 + self.dt/2*self.M_inv*p0
			p0 = p0 + self.dt*self.K*q0 + self.dt*self.cp
			q0 = q0 + self.dt/2*self.M_inv*p0
			#with warnings.catch_warnings():
				#warnings.simplefilter("error", category=RuntimeWarning)

			f.vector().set_local( q0 )
			
			if np.mod(i,10) == 0:
				self.snap_x0 = np.concatenate((self.snap_x0,x0),1)
				Energy = self.compute_energy(x0)
				self.energy = np.concatenate((self.energy,Energy),0)

			if np.mod(i,125) == 0:
				vtkfile << (f,i*self.dt)

#			if(np.mod(i,100) == 0):
#				x0 = np.concatenate((q0,p0),0)
#				full = self.Phi*np.linalg.inv(self.TJJ)*x0
#				self.snap_Q = np.concatenate( (self.snap_Q,full[0:self.N,:]) , 1 )
#				self.snap_P = np.concatenate( (self.snap_P,full[self.N:2*self.N,:]) , 1 )


			



#		q0 = np.zeros([N,1])
#		p0 = np.zeros([N,1])
#
#		self.snap_Q = np.zeros( self.cp.shape )
#		self.snap_P = np.zeros( self.cp.shape )
#
#		for i in range(0,self.MAX_ITER):
#			print(i)
#			q0 = q0 + self.dt/2*self.M_inv*p0
#			p0 = p0 + self.dt*self.K*q0 + self.dt*self.cp
#			q0 = q0 + self.dt/2*self.M_inv*p0
#
#			self.snap_Q = np.concatenate((self.snap_Q,q0),1)
#			self.snap_P = np.concatenate((self.snap_P,p0),1)
#
#			f.vector().set_local( q0 )
#			if np.mod(i,10) == 0:
#				vtkfile << (f,i*self.dt)
#

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
		print( np.linalg.cond(self.M_inv) )
		print( np.linalg.cond(self.K) )
		print( N )

	def generate_X_matrix( self ):
		#q = TrialFunction(self.V)
		#f = Function(self.V)
		#v = v = TestFunction(self.V)
		#k = inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(v))*dx
		#L = inner(f,v)*dx

			##Q = np.eye(2*N)
			##Q = np.matrix(Q)
			##Q[jj,jj] = 0
			##Q[i,i] = 0
			##Q[i,jj] = 1
			##Q[jj,i] = 1
			##T = Q*T
			##J = Q*J*np.transpose(Q)
		#K, dummy = assemble_system( k, L, self.bc )
		#Kq_mat = np.matrix( K.array() )

			##Q = np.eye(2*N)
			##Q[p,p] = -1/J[p,i]
			##T = Q*T
			##J = Q*J*np.transpose(Q)
		#p = TrialFunction(self.V)
		#k = inner(p,v)*dx

			##for j in range(i+1,2*N):
				##if(j!=p):
					##Q = np.eye(2*N)
					##Q[j,p] = -J[j,i]/J[p,i]
					##T = Q*T
					##J = Q*J*np.transpose(Q)
		#K, dummy = assemble_system( k, L, self.bc )
		#Kp_mat = np.matrix( K.array() )

			##for j in range(i+1,2*N):
				##if(j!=p):
					##Q = np.eye(2*N)
					##Q[j,i] = -J[j,p]
					##T = Q*T
					##J = Q*J*np.transpose(Q)
		#temp1 = np.concatenate( [Kq_mat,np.zeros(Kq_mat.shape)] , 1 )
		#temp2 = np.concatenate( [np.zeros(Kp_mat.shape),self.M_inv] , 1 )
		X_mat = self.Lmat

			#return T
		Xsqrt = scla.sqrtm(X_mat)
		Xsqrt = np.matrix( np.real(Xsqrt) )
		Xsqrt.dump("X_mat.dat")

		temp = np.eye(2*self.N)
		temp.dump("X_mat_eye.dat")
		
	def compute_energy( self ):
		Jn = construct_J(self.N)
		
		E = np.zeros(self.snap_x0.shape[1])
		
		if self.simtype == "full":
			for i in range(0,self.snap_x0.shape[1]):
				E[i] = 0.5*np.transpose(self.snap_x0[:,i])*self.Lmat*self.snap_x0[:,i] +np.transpose(self.snap_x0[:,i])*np.transpose(Jn)*np.concatenate([np.zeros([self.N,1]),self.cp],0)
			return E
		else:
			for i in range(0,self.snap_x0.shape[1]):
				E[i] = 0.5*np.transpose(self.snap_x0[:,i])*self.Lmat*self.snap_x0[:,i] +np.transpose(self.Phi*self.bJk*self.snap_x0[:,i])*np.concatenate([np.zeros([self.N,1]),self.cp],0)
			return E
		  
	def save_snapshots( self ):
		if self.simtype == "full":
			self.snap_x0[:self.N,:].dump("snap_Q.dat")
			self.snap_x0[self.N:,:].dump("snap_P.dat")

			self.snap_x0.dump("snap_x0.dat")
			#self.snap_Q.dump("snap_Q150.dat")
			#self.snap_P.dump("snap_P150.dat")
		else:
			self.snap_x0.dump("snap_y0.dat")
		
	def compute_error(self):
		Phi = np.matrix(np.load( "RB.dat" ))
		X = np.matrix(np.load( "X_mat.dat" ))
		
		z = np.matrix(np.load("snap_x0.dat"))
		y = np.matrix(np.load("snap_y0.dat"))
		
		self.l2err = np.linalg.norm(z - Phi*y, axis=0)
		self.werr = np.linalg.norm(X*(z - Phi*y), axis=0)
		
		self.l2err.dump("l2err.dat")
		self.werr.dump("werr.dat")
		self.energy.dump("energy.dat")
		
		fig = plt.figure(1)
		plt.subplot(311)
		plt.plot(self.l2err,label="l2err")

		plt.subplot(312)
		plt.plot(self.werr, label="werr")
		
		plt.subplot(313)
		plt.plot(self.energy,label="energy")
		
		#plt.legend()
		if self.reduced_sim == "Xweighted":
			fig.savefig('errors-energy',dpi = fig.dpi)
			plt.show()
			
		#print(self.l2err.shape)
		#plt.plot(self.werr)
		#plt.show()
		#,range(0,self.MAX_ITER),self.werr)
		