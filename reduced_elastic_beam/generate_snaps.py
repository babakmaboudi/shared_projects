#import wave as wv
import enlarged_wave as wv

wave = wv.Wave()
wave.initiate_fem()
wave.implicit_midpoint()
#wave.save_snapshots()
wave.save_vtk_result()
wave.plot_energy()