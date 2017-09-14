import wave as wv

wave = wv.Wave()
wave.initiate_fem()
wave.crank_nicolson()
#wave.symplectic_euler()
#wave.stormer_verlet()
#wave.save_snapshots()
