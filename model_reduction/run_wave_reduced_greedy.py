import wave_reduced_greedy as wvr

wave = wvr.Wave_Reduced()
wave.initiate_fem()
wave.symplectic_euler()
wave.save_vtk_result()
