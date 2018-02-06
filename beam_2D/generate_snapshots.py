import wave_symmetric as wv

wave = wv.Wave()
print('Initializing FEM...')
wave.initiate_fem()
print('Time integrating...')
wave.stormer_verlet()
#print('Generating the weight matrix...')
#wave.generate_X_matrix()
print('Saving snapshots...')
wave.save_snapshots()
