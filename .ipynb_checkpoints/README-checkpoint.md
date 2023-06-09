# Coding_Practice
 
`function?` or `help(function)` gives all information you could need about the method.

Use default settings 

`M, U = choose_default_device(m, r, path)`

Input initialization

r_, phi_ = input_state(r, m, n) 
A = set_input(r_, phi_) 


There are two options for defining the interferometer matrix
`U = get_random_interferometer(m, n_BS)`

or 
`U = import_interferometer(path, file_name)`

After 
`M = set_device_parameters(r, A, U)`
Or 
`M = import_complex_matrix(path, "/GBS_matrix.dat")`

## Compute minors 

`compute_minors()`
compute minors for all samples in `input`

## Compute moments 


there is a class `MomentUtility()` which compute moments from minors up to 4th order for one sample. 
To use it one needs to crete an instanse of the class

`moments =  MomentUtility(id_ = 1, n_moments = 4, path=path) `

where `id_` is the id of a sample, `n_moments` number of moments for computation. (

With export
`moments.export_moments()`

`m1_, m2_, m3_, m4_ = moments.get_moments()`

