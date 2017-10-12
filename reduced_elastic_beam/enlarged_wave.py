from fenics import *
from scipy.linalg import sqrtm
from scipy.ndimage.interpolation import shift
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
		self.damping = 1; # damping flag and scaling: 0 == no damping, nonzero == scaling factor
		
		# numerical parameters
		self.MAX_ITER = 10
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
		self.r = TrialFunction(self.V)
		self.d = self.q.geometric_dimension()
		self.v = TestFunction(self.V)
		self.q_new = Function(self.V)
		self.p_new = Function(self.V)
		self.r_new = Function(self.V)
		self.r_vec = self.r_new

		self.q0 = Function(self.V)
		self.p0 = Function(self.V)
		self.r0 = Function(self.V)
		
		#define right hand side function
		self.f = Constant((0, 0, -self.rho*self.g))
		self.T = Constant((0, 0, 0))
		
		#initialize the energy vector
		self.e_vec = np.array([0])
		
		#define the weak form of the enlarged system for implicit-midpoint scheme
		self.aq = inner(self.q,self.v)*dx \
		+ pow(self.theta,2)*pow(self.dt,2)*inner( sigma(self.q,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx
		
		self.Lq = inner(self.q0,self.v)*dx \
		- pow(self.dt,2)*self.theta*(1-self.theta)*inner( sigma(self.q0,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx \
	       	+ self.dt*inner(self.r0,self.v)*dx \
		+ pow(self.dt,2)*pow(self.theta,2)*inner(self.f,self.v)*dx \
	       	+ pow(self.dt,2)*self.theta*(1-self.theta)*inner(self.f,self.v)*dx
		
		self.ap = (1 + self.damping*self.dt*self.theta)*inner(self.p,self.v)*dx

		self.Lp = (1 - self.damping*self.dt*(1-self.theta))*inner(self.p0,self.v)*dx \
		- self.dt*self.theta*inner( sigma(self.q_new,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx \
		- self.dt*(1-self.theta)*inner( sigma(self.q0,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx \
		+ self.dt*self.theta*inner(self.f,self.v)*dx + self.dt*(1-self.theta)*inner(self.f,self.v)*dx

		self.ar = inner(self.r,self.v)*dx

		self.Lr = inv(1+self.dt)*(inner(self.p_new,self.v)*dx-self.dt*inner(self.r_vec,self.v)*dx)

		self.Aq = assemble(self.aq)
		self.Ap = assemble(self.ap)
		self.Ar = assemble(self.ar)
		self.Aq_mat = np.matrix( self.Aq.array() )
		self.Ap_mat = np.matrix( self.Ap.array() )
		self.Ar_mat = np.matrix( self.Ar.array() )
		
		#assemble mass, stiffness, and damping matrices
		self.k = inner( sigma(self.q,self.lambda_,self.mu,self.d) , epsilon(self.v) )*dx 	# stiffness matrix integrand
		self.m = inner(self.q,self.v)*dx                                			# mass matrix integrand

		self.K = assemble(self.k) 	# assemble stiffness matrix
		self.M = assemble(self.m) 	# assemble mass matrix
		self.D = self.M 		# damping matrix
		self.K_mat = np.matrix( self.K.array() )
		self.M_mat = np.matrix( self.M.array() )
		self.D_mat = self.damping*np.matrix( self.D.array() )
		print(sqrt(self.D_mat.size))

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
		
	def apply_bc_r( self ):
		self.br = assemble(self.Lr)
		self.bc.apply(self.Ar,self.br)
		self.br_mat = np.matrix( self.br.array() )
		self.br_mat = np.transpose( self.br_mat )

	def strommer_verlet( self ): # outdated !
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
		self.snap_R = np.zeros((528,1))

		#loop over time steps
		for i in range(0,self.MAX_ITER):
			print(i)
			self.apply_bc_q()
			coef = np.linalg.solve(self.Aq_mat,self.bq_mat)
			self.q_new.vector().set_local( coef )

#			if np.mod(i,50) == 0:
			self.snap_Q = np.c_[self.snap_Q,coef]

			self.apply_bc_p()
			coef = np.linalg.solve(self.Ap_mat,self.bp_mat)
			self.p_new.vector().set_local( coef )

#			if np.mod(i,50) == 0:
			self.snap_P = np.c_[self.snap_P,coef]
			
			self.apply_bc_r()
			coef = np.linalg.solve(self.Ar_mat,self.br_mat)
			self.r_new.vector().set_local( coef )

#			if np.mod(i,50) == 0:
			self.snap_R = np.c_[self.snap_R,coef]
			
			self.q0.assign( self.q_new )
			self.p0.assign( self.p_new )
			self.r0.assign( self.r_new )

			self.E = assemble(energy(self.q0, self.damping*self.r0 +(not self.damping)*self.p0, self.f, self.lambda_, self.mu, self.d))
			self.e_vec = np.append(self.e_vec,self.E)
			self.r_vec += self.r_new
			
	def plot_energy( self ):
		self.energy_vec = self.e_vec + self.damping*self.strings_energy()
		self.energy_vec.dump("results/energy_vec.dat")
		plt.plot(range(0,self.MAX_ITER+1),self.energy_vec,'r--',range(0,self.MAX_ITER+1),self.e_vec, 'b-.',range(0,self.MAX_ITER+1), self.energy_vec -self.e_vec, 'k.')
		plt.show()

	def save_vtk_result( self ):
		f = Function(self.V)

		vtkfile = File('results/solution.pvd')
		for i in range(0,self.MAX_ITER+1):
			coef = self.snap_Q[:,i]
			f.vector().set_local( coef )
			if np.mod(i,10) == 0:
				vtkfile << (f,i*self.dt)

	def save_snapshots( self ):
		self.snap_Q.dump("results/snap_Q.dat")
		self.snap_P.dump("results/snap_P.dat")
		self.snap_R.dump("results/snap_R.dat")
		
	def strings_energy( self ):
		#store initial data
		#self.phi = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1))
		#self.phi_s = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1,528))
		#self.phi_t = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1,528))
		
		## evaluate sqrtm beforehand
		#self.sqrtD = sqrtm(self.D_mat/2)
		
		## evaluate strings' displacements
		#for i in range(0,527):
			#self.phi[:,0] = self.snap_Q[i,:]
			#for j in range(1,self.MAX_ITER+1):
				#self.phi[:, j] = shift(self.phi[:,0],j,cval=0)
			#[self.phi_s[:, :, i], self.phi_t[:, :, i]] = np.gradient(self.phi)
			
		#return [np.trapz([pow(np.linalg.norm(self.sqrtD * self.phi_s[i, j, :], 2) ,2) for i in range(0,self.MAX_ITER+1)] + [pow(np.linalg.norm(self.sqrtD * self.phi_t[i, j, :], 2) ,2) for i in range(0,self.MAX_ITER+1)]) for j in range(0,self.MAX_ITER+1)]
		
		#return np.trapz([np.norm(self.sqrtD * np.reshape(self.phi_s[i, j, :], (527)), 2) **elpow** 2 for i in range(0,527)] + [np.norm(self.sqrtD * np.reshape(self.phi_t[i, j, :], (527)), 2) **elpow** 2 for i in range(0,527)] for j in range(0,self.MAX_ITER))
		#store initial data
		  self.phi_mat = np.zeros((528,self.MAX_ITER+1))
		  self.phi = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1))
		  self.phi_s = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1,528))
		  self.phi_t = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1,528))
		  
		  # evaluate sqrtm beforehand
		  self.sqrtD = sqrtm(self.D_mat/2)
		  
		  for i in range(0,self.MAX_ITER+1):
			  self.phi_mat[:,i] = np.reshape(self.sqrtD*self.snap_Q[:,i],528)
			  		  
		  self.X = TensorFunctionSpace(self.mesh, 'P', 1)		  
		  self.temp = Function(self.X)
		  #self.temp.vector().array()[:] = np.zeros((self.MAX_ITER+1,self.MAX_ITER+1,528))
		  
		  # evaluate strings' displacements
		  for i in range(0,527):
			  self.phi[:, 0] = self.phi_mat[i, :]
			  for j in range(1,self.MAX_ITER+1):
				  self.phi[:, j] = shift(self.phi[:,0],j,cval=0)
			  [self.phi_s[:, :, i], self.phi_t[:, :, i]] = np.gradient(self.phi) #TODO: already have the time derivative of q(t)
			  
		  for i in range(0,self.MAX_ITER+1):
			  for j in range(0,self.MAX_ITER+1):
				  self.Phi_s = self.temp
				  self.Phi_s.vector()[i,j,:] = self.phi_s[i,j,:]
				  
				  self.temp.vector().set_local( self.phi_t[i,j,:] )
				  self.temp.update()
				  self.Phi_t[i,j,:] = Function(self.V)
				  self.Phi_t[i,j,:] = self.temp
						  

		  #return [np.trapz([pow(np.linalg.norm(self.phi_s[i, j, :], 2) ,2) for j in range(0,self.MAX_ITER+1)] + [pow(np.linalg.norm(self.phi_t[i, j, :], 2) ,2) for j in range(0,self.MAX_ITER+1)]) for i in range(0,self.MAX_ITER+1)]
		  #Phi_s = Function(self.V).vector().set_local(self.phi_s[i, j, :])
		  [np.trapz([assemble(inner(self.Phi_s[i, j, :], self.Phi_s[i, j, :])*dx) for j in range(0,self.MAX_ITER+1)] + [assemble(inner(self.Phi_t[i, j, :], self.Phi_t[i, j, :])*dx) for j in range(0,self.MAX_ITER+1)]) for i in range(0,self.MAX_ITER+1)]
		  #temp = Function(self.V)
		  #[([temp[i, j, :].vector().set_local( self.phi_s[i, j, :] ) for j in range(0, self.MAX_ITER+1)]) for i in range(0, self.MAX_ITER+1)]
		  #temp.update()
		  #[np.trapz([assemble(inner(temp[i,j,:],temp[i,j,:])*dx) for j in range(0, self.MAX_ITER+1)]) for i in range(0, self.MAX_ITER+1)]
		  return 0

		  return [np.trapz([pow(np.linalg.norm(self.phi_s[i, j, :], 2) ,2) for i in range(0,self.MAX_ITER+1)] + [pow(np.linalg.norm(self.phi_t[i, j, :], 2) ,2) for i in range(0,self.MAX_ITER+1)]) for j in range(0,self.MAX_ITER+1)]

