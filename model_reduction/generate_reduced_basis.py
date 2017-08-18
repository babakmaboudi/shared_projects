import mor as mr

mor = mr.Mor()
mor.set_basis_size(100)
#mor.PSD()
mor.initiate_greedy()
mor.greedy(49)
mor.save_basis()
