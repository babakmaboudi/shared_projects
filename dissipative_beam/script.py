import beam as bm

beam = bm.Beam()
beam.initiate_fem()
beam.symplectic_euler()
beam.compute_energy()
