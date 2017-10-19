import numpy as np
import matplotlib.pyplot as plt

class Mor:
	def __init__( self ):
		self.snap_Q = np.load("snap_Q.dat")
		self.snap_P = np.load("snap_P.dat")
		self.X = np.load("X_mat.dat")
#		self.X = np.load("X_mat_eye.dat")

	def set_basis_size(self,k):
		self.rb_size = k

	def POD_energy(self):
		snaps = np.append(self.snap_Q,self.snap_P,0)
		snaps = self.X*snaps;

		U,s,V = np.linalg.svd(snaps, full_matrices=True)
#		plt.semilogy(s)
#		plt.show()
		
		self.RB = np.linalg.inv(self.X)*U[:,0:self.rb_size]

#		snaps = np.append(self.snap_Q,self.snap_P,0)
#		temp = self.X*snaps - self.X*RB*np.transpose(RB)*self.X*self.X*snaps
#		for i in range(0,snaps.shape[1]):
#			temp = self.X*snaps[:,i] - self.X*self.RB*np.transpose(self.RB)*self.X*self.X*snaps[:,i]
#			print( np.linalg.norm(snaps[:,i]) )

	def PSD(self):
		snaps = np.append(self.snap_Q,self.snap_P,1)
		
		U,s,V = np.linalg.svd(snaps, full_matrices=True)
#		plt.semilogy(s)
#		plt.show()

		self.RB = U[:,0:self.rb_size]

	def initiate_greedy(self):
		self.N = self.snap_Q.shape[0]
		self.ns = self.snap_Q.shape[1]

		self.Jn = np.zeros([2*self.N,2*self.N])
		for i in range(0,self.N):
			self.Jn[i,self.N+i] = 1
			self.Jn[self.N+i,i] = -1

#		self.X = np.load("X_mat.dat")

	def construct_Jk(self,K):
		self.K = K
		self.Jk = np.zeros([2*self.K,2*self.K])
		for i in range(0,self.K):
			self.Jk[i,self.K+i] = 1
			self.Jk[self.K+i,i] = -1

	def symplectic_proj(self):
		temp = np.dot( np.transpose(self.Jk) , np.transpose(self.A) )
		self.A_plus = np.dot(temp,self.Jn)
		self.P = np.dot(self.A,self.A_plus)

	def greedy(self,MAX_ITER):
		idx = np.random.random_sample(500)
		idx = idx*self.ns
		idx = np.floor(idx)
		idx = np.squeeze(np.asarray(idx))
		idx = idx.astype(int)

		snaps = np.append(self.snap_Q,self.snap_P,0)
		snaps = snaps[:,idx]
		ns = 500

		E = np.matrix(snaps[:,1]).reshape([2*self.N,1])
		E = E/np.linalg.norm(E)
		F = np.dot(np.transpose(self.Jn),E)

		K = 1

		for it in range(0,MAX_ITER):
			er = np.zeros(self.ns)
			for i in range(0,ns):
				self.A = np.append(E,F,1)
				self.construct_Jk(K)
				self.symplectic_proj()

				vec = np.matrix(snaps[:,i]).reshape(2*self.N,1)
				er[i] = self.porj_error(vec)

			max_idx = np.argmax(er)
			print( [ it , er[max_idx] ] )
			vec = np.matrix(snaps[:,max_idx]).reshape(2*self.N,1)
			
			vec = self.symplectic_QR(vec,E,F)
			vec = self.symplectic_QR(vec,E,F)

			E = np.append(E,vec,1)
			temp = np.dot( np.transpose(self.Jn) , vec )
			F = np.append( F , temp , 1 )
			K += 1

		self.RB = np.concatenate( (E,F),1 )
		print(self.RB.shape)

	def greedy_energy(self,MAX_ITER):
		idx = np.random.random_sample(500)
		idx = idx*self.ns
		idx = np.floor(idx)
		idx = np.squeeze(np.asarray(idx))
		idx = idx.astype(int)

		snaps = np.append(self.snap_Q,self.snap_P,0)
		snaps = snaps[:,idx]
		snaps = np.matrix(snaps)
#		self.X = np.matrix(self.X)
#		snaps = self.X*snaps
		ns = 500

		E = np.matrix(snaps[:,1]).reshape([2*self.N,1])
		E = E/np.linalg.norm(E)
		F = np.dot(np.transpose(self.Jn),E)

		K = 1

		for it in range(0,MAX_ITER):
			er = np.zeros(self.ns)
			for i in range(0,ns):
				self.A = np.append(E,F,1)
				self.construct_Jk(K)
				self.symplectic_proj()

				vec = np.matrix(snaps[:,i]).reshape(2*self.N,1)
				er[i] = self.porj_error(vec)

			max_idx = np.argmax(er)
			print( [ it , er[max_idx] ] )
			vec = np.matrix(snaps[:,max_idx]).reshape(2*self.N,1)
			
			vec = self.symplectic_QR(vec,E,F)
			vec = self.symplectic_QR(vec,E,F)

			E = np.append(E,vec,1)
			temp = np.dot( np.transpose(self.Jn) , vec )
			F = np.append( F , temp , 1 )
			K += 1

#		E = np.linalg.inv(self.X)*E
#		F = np.transpose(self.Jn)*E

		self.RB = np.concatenate( (E,F),1 )

#		temp = np.transpose( self.RB )*self.Jn*self.RB
#		print(temp)

	def porj_error(self,vec):
		return np.linalg.norm( vec - np.dot(self.P,vec) )
		
	def symplectic_QR(self,v,E,F):
		vec = v
		for i in range(E.shape[1]):
			e = np.matrix(E[:,i]).reshape([2*self.N,1])
			f = np.matrix(F[:,i]).reshape([2*self.N,1])
			vec = self.J2_orthogonalize(vec,e,f)
		vec = vec/np.linalg.norm(vec)
		return vec

	def J2_orthogonalize(self,v,e,f):
		temp = np.dot(-np.transpose( v ),self.Jn)
		alpha = np.dot(temp,f)

		temp = np.dot(np.transpose( v ),self.Jn)
		beta = np.dot(temp,e)

		res = v + alpha[0,0]*e + beta[0,0]*f
		return res

	def save_basis(self):
		self.RB.dump("RB.dat")
