import wave_reduced as wvr

wave = wvr.Wave_Reduced()
wave.initiate_fem()
wave.symplectic_euler()
wave.compute_energy()