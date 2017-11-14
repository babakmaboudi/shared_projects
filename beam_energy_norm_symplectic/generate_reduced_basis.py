import mor as mr

mor = mr.Mor()
#mor.set_basis_size(100)
#mor.POD_energy()
mor.initiate_greedy()
mor.greedy(99)
mor.save_basis()
