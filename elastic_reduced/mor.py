import numpy as np
import matplotlib.pyplot as plt

class Mor:
	def __init__( self ):
		self.snap_Q = mat2 = np.load("snap_Q.dat")
		self.snap_P = mat2 = np.load("snap_Q.dat")
		print(self.snap_Q.shape)

	def set_basis_size(self,k):
		self.rb_size = k

	def POD(self):
		snaps = np.append(self.snap_Q,self.snap_P,0)

		U,s,V = np.linalg.svd(snaps, full_matrices=True)
		plt.semilogy(s)
		plt.show()
		
		self.RB = U[:,0:self.rb_size]

	def PSD(self):
		snaps = np.append(self.snap_Q,self.snap_P,1)
		
		U,s,V = np.linalg.svd(snaps, full_matrices=True)
		plt.semilogy(s)
		plt.show()

		self.RB = U[:,0:self.rb_size]

	def save_basis(self):
		self.RB.dump("RB.dat")
