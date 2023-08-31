<p align="center">
<img src="images/qb_image_2.png" alt="Drawing" style="width: 300px;"/> 
<p align="center">

Quantum devices seek to **Quantum Advantage**: a result beyond the capabilities of classical computers. A particular type of optical quantum computer - Gaussian Boson Sampler - is one of the most promising candidates for achieving this in the nearest future. Fast simulation methods play a crucial role in assessing performance and enhancing these devices.   
    
    
# Project Overview     
  
**Quantum Badger** is a Python module that solves the Gaussian Boson Sampling problem with threshold detection. The main focus of this project is to demonstrate and validate the simulation method, proposed by the [A.S. Popova and A.N. Rubtsov](https://arxiv.org/pdf/2106.01445.pdf). At the current stage, this module is designed to compute probabilities of samples rather than generate samples from a distribution. We believe that our module could be valuable for testing the output of GBS devices and serving as a benchmark for other software.
    
# Installation 

To use the Quantum Budger on your local machine, you need to download the repository and have `Python 3`, as well as the `g++` compiler, installed.
We also recommend creating a virtual environment with the packages listed below.
## Codes and Resources Used

* **Editor**: To run `tutorial.ipynb` file you need to install [Jupyter](https://jupyter.org/) enviroment. We used Jupyter Lab. 
    
* **Python version**: 3.9.13 
    
## Python Packages Used 
    
To reproduce the current project, you must the following Python libraries:

**General Purpose**: `shutil`, `subprocess`

**Data Manipulation**: `pandas==1.5.3`, `jinja2==3.1.2`, `datetime`

**Modelling**: `math`, `random`, `scipy==1.11.1`, `numpy==1.24.3`, `joblib==1.2.0`
    
**Validation**: `strawberryfields==0.23.0`, `thewalrus==0.19.0`

**Visualization**: `matplotlib==3.7.2`

## Data 
    
All necessary data to reproduce basic functionality is generated in `tutorial.ipynb` and `method_details.ipynb`, and it is stored in the `/data` directory of the repository.
    
Each time when you initiate simulation all input and output data is generated in the folder with the current date and time. The stucture of the generated data will be the following 
    
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

To use your own interferometer matrix, you need to replace `matrix_U.dat`  file.   
    
# Usage 

One can find the possible use cases of the **Quantum Badger** module in `tutorial.ipynb`. 

It is possible to use our module in two ways: to simulate your GBS device or to investigate how our method performs in general. For simulation, you need to specify the parameters of the setup and obtain samples. After this, you can compute approximate probabilities and evaluate the results using 5 metrics: **Relative Weighted Error**, **Mean Absolute Percentage Error**, **Total Variation Distance**, **Cosine Similarity**, and **Fidelity**. 

The details about the method can be found in `method_details.ipynb`. 
    
**Note**
    
`function?` or `help(function)` gives all information you could need about the method.


# Results and Evaluation 

Because we are developing an algorithm for computing probabilities, we do not compare our results with a device output directly. Instead, we evaluate the performance in two ways:
 
**For devices having less than 30 modes**

* We compare the exact and approximated probabilities of all possible samples;
    
* We compare the exact probabilities of uniformly generated samples for an ensemble of slightly different interferometers. It enables to consider of stochastic measurement errors in the reconstruction of an interferometer matrix. 
    
**For larger devices**  
        
* We compare the frequency of occurrence of samples of small mid-sized GBS devices and the probabilities for these samples, computed approximately (we suppose here that a large part of output samples should have large probability). 
    
# Future work  

Potential future work include:
    
* Considering devices with photon losses; 

* Implementation of the Markov Chain Monte Carlo sampling with the approximate evaluation of probabilities, developed here. 
 
# Acknowledgments 
 
We appreciate the Xanadu team for the free and open-source libraries that enabled us to benchmark our method.
    
# Support 
    
We would really welcome feedback, suggestions, comments, or criticism. If you have questions, you also can contact us at ppva.nastya@proton.me. 

# Licence

For this github repository, the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) is used. 
