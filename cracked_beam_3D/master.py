# symulate full system
import wave_symmetric as wv

wave = wv.Wave()
print('Initializing FEM...')
wave.initiate_fem()
print('Time integrating...')
wave.stormer_verlet()
print('Generating the weight matrix...')
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
mor.greedy(99)
mor.save_basis()
print('complete.')


# simulate reduced system
print('simulating reduced system...')
import wave_reduced_hamil as wvr

wave = wvr.Wave_Reduced()
wave.initiate_fem()
#wave.stormer_verlet()
#wave.test()
wave.mid_point()
#wave.symplectic_euler()
#wave.symplectic_euler2()
wave.save_snapshots()
wave.compute_error()
print('complete.')