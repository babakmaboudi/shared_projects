import mor as mr

mor = mr.Mor(1)
#mor.set_basis_size(100)
#mor.POD_energy()
mor.initiate_greedy()
mor.greedy(99,1) # 0 == weighted norm, else Euclidean norm
mor.save_basis()
