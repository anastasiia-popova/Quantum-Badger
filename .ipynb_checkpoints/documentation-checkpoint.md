<p align="right">
<img src="images/qb_image_2.png" alt="Drawing" style="width: 300px;"/> 
<p align="right">
    
# Project Overview     
---     
**Quantum Badger** is a python module that solves Gaussian Boson Sampling problem with threshold detection. The main focus of this project is to demonstrate and validate the method, proposed by the [A.S. Popova and A.N. Rubtsov](https://arxiv.org/pdf/2106.01445.pdf). At the current stage, this module is designed to compute probabilities of samples rather than generate samples from a distribution. We believe that our module could be valuable for testing the output of GBS devices and serving as a benchmark for other software.
    
# Installation 
--- 

To use the Quantum Budger on your local machine you need to download the repository and to have `Python 3` and the `g++` compiler installed. 
    
## Codes and Resourses Used 

* **Editor**: To run `demo.ipynb` file you need to install [Jupyter](https://jupyter.org/) enviroment. We used Jupyter Lab. 
    
* **Python version**: 3.9.13 
    
## Python Packages Used 
    
To reproduce the current project, you must the following Python libraries:

**General Purpose**: `shutil`, `subprocess`

**Data Manipulation**: `pandas`, `datetime`

**Modelling**: `math`, `random`, `scipy`, `numpy`
    
**Validation**: `strawberryfields`, `thewalrus`

**Visualization**: `matplotlib`

# Data 
    
All necessary data to reproduce basic functionality is included in the `/data` directory of the repository.
    
Each time when you initiate simulation all input and output data is generated in the folder with the current date. The stucture of the generated data will be the following 
    
```
├── data
│   ├── min_hour-day_month_year
│   │   ├── input
│   │   │   ├── samples_ids.dat
│   │   │   ├── Submatrix_0.dat
│   │   │   ├── ...
│   │   │   ├── Submatrix_max_id.dat
│   │   ├── output
│   │   │   ├── samples_probabilities.csv
│   │   │   ├── Cumulants_0.dat
│   │   │   ├── Moments_0.dat
│   │   │   ├── Minors_0-1_0.dat
│   │   │   ├── Minors_2_0.dat
│   │   │   ├── Minors_3_0.dat
│   │   │   ├── Minors_4_0.dat
│   │   │   ├── Result_0.dat
│   │   │   ├── ...
│   │   │   ├── Cumulants_max_id.dat
│   │   │   ├── Moments_max_id.dat
│   │   │   ├── Minors_0-1_max_id.dat
│   │   │   ├── Minors_2_max_id.dat
│   │   │   ├── Minors_3_max_id.dat
│   │   │   ├── Minors_4_max_id.dat
│   │   │   └── Result_max_id.dat
│   │   ├── matrix_U.dat
│   │   ├── parameters_of_interferometer.dat
│   │   ├── initial_state.dat
│   │   ├── GBS_matrix.dat
│   │   └── samples.dat

```

To use your own interferometer matrix, you need to replace `data/min_hour-day_month_year/matrix_U.dat`  file.   
    
# Usage 
--- 
One can find the possible use cases of the **Quantum Badger** module in `demo.ipynb`. 

It is possible to use our module in two ways: to simulate your GBS device or to investigate how our method performs in general. For simulation, you need to specify the parameters of the setup and obtain samples. After this, you can compute approximate probabilities and evaluate the results using four metrics: **Relative Weighted Error**, **Total Variation Distance**, **Fidelity**, and **Cross Entropy**.
 
    

* `function?` or `help(function)` gives all information you could need about the method.


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


