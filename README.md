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