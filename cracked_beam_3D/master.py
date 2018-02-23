#symulate full system
import wave as wv

wave = wv.wave(80000,0.00001,'results/solution.pvd','meshes/cracked_beam.xml',"full","dontcare",80)
print('Initializing FEM...')
wave.initiate_fem()
print('Time integrating...')
wave.mid_point()
print('Genearting the weight matrix...')
wave.generate_X_matrix()
print('Saving snapshots...')
wave.save_snapshots()
print('complete.')

# genearate reduced basis
print('Generateing reduced basis...')
import mor as mr

mor = mr.Mor()
#mor.set_basis_size(100)
#mor.POD_energy()
mor.initiate_greedy()
mor.greedy(100,100)
mor.save_basis()
print('complete.')

# simulate reduced system
print('simulating Iweighted reduced system...')
import wave as wv

wave = wv.wave(80000,0.00001,'results_reduced/solution.pvd','meshes/cracked_beam.xml',"reduced","Iweighted",80)
wave.initiate_fem()
#wave.stormer_verlet()
#wave.test()
wave.mid_point()
#wave.symplectic_euler()
#wave.symplectic_euler2()
wave.save_snapshots()
wave.compute_error()
print('complete.')

print('simulating Xweighted reduced system...')
import wave as wv

wave = wv.wave(80000,0.00001,'results_reduced/solution.pvd','meshes/cracked_beam.xml',"reduced","Xweighted",80)
wave.initiate_fem()
#wave.stormer_verlet()
#wave.test()
wave.mid_point()
#wave.symplectic_euler()
#wave.symplectic_euler2()
wave.save_snapshots()
wave.compute_error()
print('complete.')