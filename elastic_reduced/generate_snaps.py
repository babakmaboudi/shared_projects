import wave as wv

wave = wv.Wave()
wave.initiate_fem()
wave.strommer_verlet()
wave.save_snapshots()
wave.save_vtk_result()
