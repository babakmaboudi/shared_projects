from fenics import*
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < DOLFIN_EPS

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u,lambda_,mu,d):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

class Beam:
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

		self.alpha_diss = 1.0# dissipation parameter

		# numerical parameters
		self.MAX_ITER = 300
		self.dt = 0.01

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
		self.f = TrialFunction(self.V)
		self.f_hist = TrialFunction(self.V)

		self.d = self.q.geometric_dimension()
		self.v = TestFunction(self.V)
		

		#define the right hand side fucntion
		self.f_rhs = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))

		#defining the weak forms
		self.aq = inner(self.q,self.v)*dx
		self.ap = inner(self.p,self.v)*dx
		self.af = inner(self.f,self.v)*dx
		self.Aq = assemble(self.aq)
		self.Ap = assemble(self.ap)
		self.Af = assemble(self.af)


		#define stiffness matrices
		self.kq = inner(self.f,self.v)*dx
		self.kp = -inner(sigma(self.q,self.lambda_,self.mu,self.d),epsilon(self.v))*dx
		self.Kq = assemble(self.kq)
		self.Kp = assemble(self.kp)
		self.Kq = assemble(self.kq)
		self.Kp = assemble(self.kp)
		self.Kq_mat = np.matrix( self.Kq.array() )
		self.Kp_mat = np.matrix( self.Kp.array() )

		#force vector
		self.Lq = inner(self.T,self.v)*dx
		self.bq = assemble(self.Lq)
		self.bc.apply( self.Aq , self.bq )
		self.Aq_mat = np.matrix( self.Aq.array() )

		self.Lp = inner(self.f_rhs,self.v)*dx
		self.bp = assemble(self.Lp)
		self.bc.apply( self.Ap , self.bp )
		self.Ap_mat = np.matrix( self.Ap.array() )

		self.A_inv = np.linalg.inv( self.Ap_mat )
		self.L = self.A_inv*self.Kp_mat

		self.c = np.matrix( self.bp.array() )
		self.c = np.transpose( self.c )
		self.cp = self.A_inv*self.c

		#define extension variables
		self.F_hist = np.zeros(self.c.shape)
		self.theta = np.zeros(self.c.shape)
		self.phi = np.zeros(self.c.shape)

	def apply_bc_q( self ):
		self.bq.set_local( self.q_new )
		self.bc.apply( self.Aq , self.bq )
		self.q_new = np.matrix( self.bq.array() )
		self.q_new = np.transpose( self.q_new )

	def apply_bc_p( self ):
		self.bp.set_local( self.p_new )
		self.bc.apply( self.Ap , self.bp )
		self.p_new = np.matrix( self.bp.array() )
		self.p_new = np.transpose( self.p_new )

	def symplectic_euler( self ):
		vq0 = np.zeros(self.c.shape)
		vp0 = np.zeros(self.c.shape)
		self.f_vec_hist = np.zeros(self.c.shape)
		self.f_vec = np.zeros(self.c.shape)

		vtkfile = File('results/solution.pvd')
		out_func = Function(self.V)

		self.snap_Q = np.zeros( self.c.shape )
		self.snap_P = np.zeros( self.c.shape )

		E_wave = np.zeros([1,self.MAX_ITER+1])

		print('Computing snapshots ...')

		for i in range(0,self.MAX_ITER):

			self.p_new = vp0 + self.dt*self.L*vq0 + self.dt*self.cp

			self.apply_bc_p()
			vp0 = self.p_new

			self.compute_f_vector()

			self.q_new = vq0 + self.dt*self.f_vec
			self.apply_bc_q()
			vq0 = self.q_new

			self.snap_Q = np.concatenate( [self.snap_Q,vq0] , 1 )
			self.snap_P = np.concatenate( [self.snap_P,vp0] , 1 )
			
	def compute_f_vector( self ):
		self.f_vec = (self.p_new -\
		self.f_vec_hist)/(1 + self.alpha_diss*self.dt)

		self.f_vec_hist += self.alpha_diss*self.dt*self.f_vec
		
		self.F_hist = np.concatenate([self.F_hist,self.f_vec],1)

	def energy_wave( self , q_vec , p_vec , Tphi_vec ):
		q = Function( self.V )
		q.vector().set_local( q_vec )
		p = Function( self.V )
		p.vector().set_local( p_vec - Tphi_vec )

		energy = 0.5*inner(p,p)*dx + 0.5*inner(sigma(q,self.lambda_,self.mu,self.d),epsilon(q))*dx - inner( self.f_rhs,q )*dx
		E = assemble(energy)
		return E

	def energy_string( self ):
		t = Function(self.V)
		dpdx = Function(self.V)

		energy = 0.5*inner(t,t)*dx + 0.5*inner(dpdx,dpdx)*dx

		E = 0
		num_t = self.theta.shape[1]
		for i in range(0,num_t):
			t.vector().set_local( self.theta[:,i] )
			dpdx.vector().set_local( self.dphidx[:,i] )
			
			E += assemble(energy)*self.dt
		return E

	def compute_energy( self ):
		print('Computing the System Energy...')
		num_t = self.F_hist.shape[1]
		E_wave = np.zeros([num_t,1])
		E_string = np.zeros([num_t,1])	
		for i in range(1,num_t):
			print(i)
			f = self.F_hist[:,0:i]
			self.compute_theta(f)
			self.compute_dphidx(f)
			self.compute_Tphi(f)

			E_wave[i] = self.energy_wave(self.snap_Q[:,i],self.snap_P[:,i],self.Tphi)
			E_string[i] = self.energy_string()
			
			

		plt.plot(E_wave)
		plt.plot(2*E_string)
		plt.plot(E_wave+2*E_string)
		plt.show()

	def compute_theta( self , f ):
		N = f.shape[1]
		self.theta = np.sqrt(2)/2*f[:,N-1]
		for i in range(1,N):
			temp = np.sqrt(2)/2*f[:,N-i-1]
			self.theta = np.concatenate( [self.theta,temp] , 1 )

	def compute_dphidx( self , f ):
		N = f.shape[1]
		self.dphidx = np.sqrt(2)/2*f[:,N-1]
		for i in range(1,N):
			temp = -np.sqrt(2)/2*f[:,N-i-1]
			self.dphidx = np.concatenate( [self.dphidx,temp] , 1 )		

	def compute_Tphi( self , f ):
		self.Tphi = np.sum(f,1)*self.dt

	def save_snapshots( self ):
		self.snap_Q.dump("snap_Q.dat")
		self.snap_Q.dump("snap_P.dat")

