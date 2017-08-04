import wave_reduced as wvr

wave = wvr.Wave_Reduced()
wave.initiate_fem()
wave.strommer_verlet()
wave.save_vtk_result()
