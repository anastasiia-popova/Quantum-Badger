# Cheat Sheet

## Preparation 

Use default settings 

```
M, U = choose_default_device(m, r, path)
```

Input initialization

```
r_, phi_ = input_state(r, m, n) 
A = set_input(r_, phi_) 
```


There are two options for defining the interferometer matrix
```
U = get_random_interferometer(m, n_BS)
```

or 
```
U = import_interferometer(path, file_name)
```

After you can initialize the device

```
M = set_device_parameters(r, A, U)
```

or import previously generated matrix

```
M = import_input(path, "/GBS_matrix.dat")
```

## Get DataFrame for basis states

```
df_basis = get_basis_df(M)
```
Find probability of a specific sample in DataFrame

```
sample = [0,1,0,0,1,1,1,1,1]
df_basis["probability_exact"].loc[str(sample)] 
```

## Generate, export or import samples

```
batch_size = 10 
samples = uniform_sampling_tr(batch_size,n,m)

export_samples(samples, path, "/samples.dat")

samples = import_samples(path, "/samples.dat")
```

## Compute submatrices according to samples

```
submatriсes_export(M, samples, path)
```

## Compute minors 

```
compute_minors()
```

Computes minors for all samples in `input` directory. 

## Compute moments 

there is a class `MomentUtility()` which computes moments from minors up to 4th order for one sample. 
To use it one needs to create an instance of the class

```
moments =  MomentUtility(id_ = 1, n_moments = 4, path=path)
```

where `id_` is the id of a sample, and `n_moments` number of moments for computation. 

With export

```
moments.export_moments()

m1_, m2_, m3_, m4_ = moments.get_moments()
```

## Get approximate probabilities

```
cumulants = CumulantUtility(id_ = id_, n_moments = 4)

probability_approx_2,probability_approx_3, probability_approx_4 = cumulants.prob_approx()
```

### Compute probabilities for all samples

```
dict_probabilities = get_approx_probabilities(path) #if you already have submatrices for samples
```

`dict_probabilities` is a dictionary containing unique samples and probabilities for samples. 


If you have only samples and want to compute probabilities for them. 

```
dict_probabilities = compute_probabilities(samples)
```

