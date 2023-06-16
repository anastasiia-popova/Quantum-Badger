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
`M = import_input(path, "/GBS_matrix.dat")`

## Get DataFrame for basis states

```
df_basis = get_basis_df(M)
```
```
# Find probability of a specific sample in DataFrame
sample = [0]*m
df_basis["probability_exact"].loc[str(sample)] 

```
## Generate, export and imput samples

```
batch_size = 10 
samples = uniform_sampling_tr(batch_size,n,m)

export_samples(samples, path, "/samples.dat")

samples = import_samples(path, "/samples.dat")
```

## Compute submatrices according to samples

`submatriсes_export(M, samples, path)`

## Compute minors 

`compute_minors()`
compute minors for all samples in `input`

## Compute moments 


there is a class `MomentUtility()` which compute moments from minors up to 4th order for one sample. 
To use it one needs to crete an instanse of the class

`moments =  MomentUtility(id_ = 1, n_moments = 4, path=path) `

where `id_` is the id of a sample, `n_moments` number of moments for computation. (

With export

```
moments.export_moments()

m1_, m2_, m3_, m4_ = moments.get_moments()
```

## Get approximate probabilitie

```
cumulants = CumulantUtility(id_ = id_, n_moments = 4)

probability_approx_2,probability_approx_3, probability_approx_4 = cumulants.prob_approx()
```

### For all samples

```
dict_probabilities = get_approx_probabilities(path) #if you already have submatrices for samples
```
`dict_probabilities` is a dictionary containing unique samples and probabilities for them. 


If you have only samples and want to compute probabilities for them. 

```
dict_probabilities = compute_probabilities(samples)
```

## Generate DataFrame with probabilities of all basis states 

```
df_basis = get_basis_df(M)
df_basis.info()
df_basis.head()
```

### Find probability of a specific sample in DataFrame
```
sample = [0]*m 

# or sample ='00000' without convert_list_to_str() method

df_basis["probability_exact"].loc[convert_list_to_str(sample)] 
```

### Total probability mass function of a small Gaussian Boson Sampling device

```
pmf_values = [sum(df_basis["probability_exact"][df_basis["n_clicks"] == n]) for n in range(m+1)]

plt.vlines(
    range(m+1), 
    0,
    pmf_values, 
    color = colors[0],
    linestyles='dashed'
)

plt.plot(
    range(m+1), 
    pmf_values,
    'o-',
    color = colors[1]
)

plt.yscale('log')
plt.xlabel("Number of clicks", fontsize=12)
plt.ylabel("Probability mass function", fontsize=12)
plt.title("PMF for Gaussian Boson Sampling", fontsize=17);
plt.show()

```
#### Total probability mass function of sectors 

```
sample = [1]*(m-2) + [0]*2 # just for example
n_clicked = sum(sample)

P_sectors = prob_sectors_exact(M, sample=sample)

for nu in range(n_clicked,n_clicked*10, n_clicked):
    plt.plot(
        range(n_clicked+1),
        [P_sectors[j,nu] for j in range(n_clicked+1)],
        '--' ,
        label = 'k='+str(nu)
)
plt.yscale('log')
plt.ylim(10**(-11), 10**(-1))
plt.legend(prop={'size':10}, loc='lower left')
plt.xlabel("Number of clicks", fontsize=12)
plt.ylabel("Probability mass function", fontsize=12)
plt.title("PMF for sectors", fontsize=17);
plt.show()
```

#### Probability mass function of sectors for a sample

```
sample = [1]*(m-2) + [0]*2 # just for example
n_clicked = sum(sample)
nu_max = 10*n_clicked

P_sectors = prob_sectors_exact(M, sample=sample)

plt.plot(
        range(nu_max),
        [P_sectors[n_clicked,nu] for nu in range(nu_max)],
        '-' 
)
plt.yscale('log')
#plt.legend(prop={'size':10}, loc='lower left')
plt.xlabel("Sectors", fontsize=12)
plt.ylabel("Probability mass function", fontsize=12)
plt.title(f"PMF of sectors for {n_clicked} clicked detectors", fontsize=17);
plt.show()
```
