# Created by Anastasia Popova ppv.nastya@proton.me
import numpy as np
import random
from math import * 
import os
import pandas as pd

import scipy.optimize as opt

import shutil
import subprocess
old_settings = np.seterr(all='ignore') 

from datetime import datetime

# Compiling c++ file
cmd = "cpp/Minors.cpp"
subprocess.call(["g++", cmd])


def return_path(filename='tutorial.ipynb'):
    """Uses os to return the correct path of a directory.
    Default is a directory containing 'tutorial.ipynb' file. 
    """
    absolute_path = os.path.abspath(filename)
    directory_name = os.path.dirname(absolute_path).replace("\\" , "/" )
    #full_path = os.path.join(directory_name, filename)
    return directory_name #full_path

def create_path(filename='tutorial.ipynb'):
    # Directory name is a time now
    #now = datetime.now().strftime("%M_%H-%d_%m_%Y") 
    
    directory = datetime.now().strftime("%M_%H-%d_%m_%Y")
    # Parent directory path
    parent_dir = os.path.join(return_path(filename), "data")

    # Join path
    path = os.path.join(parent_dir, directory)
    
    try:
        os.makedirs(os.path.join(path, "input"))
        os.makedirs(os.path.join(path, "output"))
    
    except:
        print("error")
        path = parent_dir
    
    return path
    
    



# Methods
def input_state(r, m, n):

    """
    Returns a list of squeezing parameters and a list of input random phases.
    Fills n modes out of m with equal squeezing parameter r.
    
    Args:
        r (float): The squeezing parameter.
        m (int): The total number of modes.
        n (int): The number of filled inputs.
        
    Returns:
        list: A list of squeezing parameters with length m, where only half of the modes have the value of r.
        list: A list of input random phases with length m.
    """

    r_ = [r if i < n else 0 for i in range(m)]
    phi_ = [np.round(np.pi * random.random(), 5) if i < n else 0 for i in range(m)]

    return r_, phi_


def input_matrix(r, phi, m, n):

    """
    Returns a diagonal input matrix A.
    
    Args:
        r (list): A list of squeezing parameters with length n.
        phi (list): A list of input random phases with length n.
        m (int): The size of the square diagonal matrix A.
        n (int): The number of non-zero diagonal elements in A.
        
    Returns:
        np.ndarray: A diagonal input matrix A of shape (m, m) with complex128 dtype, where the diagonal elements are 
                    calculated as -exp(1j * phi[i]) * tanh(r[i]) / 2 for i in the range [0, n).
    """
    
    A = np.zeros((m, m), dtype=np.complex128)

    for i in range(n):
        A[i, i] = -np.exp(1j * phi[i]) * np.tanh(r[i]) / 2

    return A

def set_input(r, phi, path=None):
    
    """
    Returns a diagonal input matrix A.
    
    Args:
        r (list): A list of squeezing parameters with length m.
        phi (list): A list of input random phases with length m.
        path (str, optional): A path to the directory
        
    Returns:
        np.ndarray: A diagonal input matrix A of shape (m, m) with complex128 dtype, where the diagonal elements are 
                    calculated as -exp(1j * phi[i]) * tanh(r[i]) / 2 for i in the range [0, m).
    """
    
    
    m = len(r)
    A = np.zeros((m, m), dtype=np.complex128)

    for i in range(m):
        A[i, i] = -np.exp(1j * phi[i]) * np.tanh(r[i]) / 2
        
    if path is not None:
        with open(path + r"/initial_state.dat", "w") as ouf:

            ouf.write("N\tr\tphi\tA_real\tA_imag\n")

            for k in range(A.shape[0]):
                ouf.write(
                    f"{k}\t{r[k]}\t{phi[k]}\t{A[k, k].real}\t{A[k, k].imag}\n"
                )  
        print("Data were exported to " + path + r"/initial_state.dat")

    return A
    

# Interferometer random matrix


def ps_1(phi, ind, i, m):

    """
    Returns a matrix of phase shifter for the first random channel of a circuit.
    
    Args:
        phi (list): A list of input random phases.
        ind (list): A list of indices specifying the locations in the matrix to modify.
        i (int): The index of the random phase to use from phi.
        m (int): The size of the square matrix U.
        
    Returns:
        U (np.ndarray): A complex128 matrix U of shape (m, m), where the (ind[0], ind[0]) and (ind[1], ind[1]) diagonal 
                    elements are modified with exp(1j * phi[i]) and 1 respectively, and all other elements are set 
                    to identity (1 on the diagonal and 0 off the diagonal).
    """

    U = np.eye(m, dtype=np.complex128)

    U[ind[0], ind[0]] = np.exp(1j * phi[i])
    U[ind[1], ind[1]] = 1

    return U


def ps_2(psi, ind, i, m):

    """
    Returns a matrix of phase shifter for the second random channel of a circuit.
    
    Args:
        psi (list): A list of input random phases.
        ind (list): A list of indices specifying the locations in the matrix to modify.
        i (int): The index of the random phase to use from psi.
        m (int): The size of the square matrix U.
        
    Returns:
        np.ndarray: A complex128 matrix U of shape (m, m), where the (ind[0], ind[0]) and (ind[1], ind[1]) diagonal 
                    elements are modified with 1 and exp(1j * psi[i]) respectively, and all other elements are set 
                    to identity (1 on the diagonal and 0 off the diagonal).
    """

    U = np.eye(m, dtype=np.complex128)

    U[ind[0], ind[0]] = 1
    U[ind[1], ind[1]] = np.exp(1j * psi[i])

    return U


def bs(eta, i, ind, m):

    """
    Returns matrix of beam splitter for two random 
    channels of a circuit.
       

    Args:
        eta (list or np.ndarray): List or array of angles of rotation for beam splitters.
        i (int): Index of the current angle of rotation.
        ind (list or tuple): List or tuple of indices for channels' mixing.
        m (int): Size of the U-matrix.

    Returns:
        np.ndarray: A complex matrix representing the beam splitter.
        
    """

    U = np.eye(m, dtype=np.complex128)

    U[ind[0], ind[0]] = np.cos(eta[i])
    U[ind[0], ind[1]] = -np.sin(eta[i])
    U[ind[1], ind[0]] = np.sin(eta[i])
    U[ind[1], ind[1]] = np.cos(eta[i])

    return U

def interferometer(n_bs, m):

    """
    Returns a random matrix of an interferometer,
    an array of random indices for channels' mixing,
    and lists of random angles of phase shifters and beam splitters.

    Args:
        n_bs (int): Number of beam splitters.
        m (int): Number of channels.

    Returns:
        tuple: A tuple containing:
            - U (np.ndarray): A complex matrix representing the interferometer.
            - ind (list): A list of random indices for channels' mixing.
            - phi (list): A list of random angles for phase shifters.
            - psi (list): A list of random angles for phase shifters.
            - eta (list): A list of random angles for beam splitters.
    """

    phi = [np.round(random.uniform(0, 2*np.pi), 5) for i in range(n_bs)]
    psi = [np.round(random.uniform(0, 2*np.pi), 5) for i in range(n_bs)]
    eta = [np.round(0.5*np.sin(random.uniform(0, np.pi)), 5) for i in range(n_bs)]
    
    ind = []

    while len(ind) < (n_bs):
        k, j = random.randint(0, m - 1), random.randint(0, m - 1)
        if k != j:
            k_j = sorted([k, j])
            ind.append(k_j)

    U = np.eye(m, dtype=np.complex128)

    for i in range(n_bs):

        U = (
            U
            @ ps_1(phi, ind[i], i, m)
            @ bs(eta, i, ind[i], m)
            @ ps_2(psi, ind[i], i, m)
        )
        
    return U, ind, phi, psi, eta

def interferometer_approx(n_bs, ind, phi_0, psi_0, eta_0, error,  m):

    """
    Returns an approximate interferometer matrix U 
    with phases phi, psi, and angles eta, distributed 
    normally with variance equals `error` around 
    specified `phi_0`, `psi_0`, `eta_0`.  
    Parameters:
        n_bs (int): Number of random beamsplitters.
        ind (list): List of index pairs for beamsplitters.
        phi_0 (list): List of initial phases for the first phase shifters.
        psi_0 (list): List of initial phases for the second phase shifters.
        eta_0 (list): List of initial angles for the beamsplitters.
        error (float): Standard deviation of normal distribution
                       for random phases and angles.
        m (int): Number of modes.
    Returns:
        U (ndarray): Interferometer matrix.
        
    """
    phi = [np.random.normal(phi_0[i], error, 1)  for i in range(n_bs)]
    psi = [np.random.normal(psi_0[i], error, 1) for i in range(n_bs)]
    eta = [np.random.normal(eta_0[i], error, 1) for i in range(n_bs)]

    U = np.eye(m, dtype=np.complex128)

    for i in range(n_bs):

        U = (
            U
            @ ps_1(phi, ind[i], i, m)
            @ bs(eta, i, ind[i], m)
            @ ps_2(psi, ind[i], i, m)
        )
        

    return U

def get_random_interferometer(m, n_bs, path=None):
    
    """
    Generates a random interferometer matrix U with beam splitters and phase shifters.
    
    Args:
        m (int): The size of the square matrix U.
        n_bs (int): The number of beam splitters (BS) and phase shifters (PS) to use in the interferometer.
        path (str, optional): A path to the directory
        
    Returns:
        U (np.ndarray): A complex128 matrix of shape (m, m), representing the random interferometer with n_bs beam splitters
        and phase shifters.
        
    Note:
        This function generates random phases phi, psi, and eta for the beam splitters and phase shifters, as well as
        random indices ind for the locations of the beam splitters and phase shifters in the matrix U. It then applies
        the beam splitters and phase shifters sequentially to the matrix U to construct the interferometer. If export=True,
        the generated matrix U and parameters are saved to files in the specified path.
    """
    
    phi = [np.round(random.uniform(0, 2*np.pi), 5) for i in range(n_bs)]
    psi = [np.round(random.uniform(0, 2*np.pi), 5) for i in range(n_bs)]
    eta = [np.round(0.5*np.sin(random.uniform(0, np.pi)), 5) for i in range(n_bs)]
    
    n_ps = int(n_bs*2)

    ind = []

    while len(ind) < (n_bs):
        k, j = random.randint(0, m - 1), random.randint(0, m - 1)
        if k != j:
            k_j = sorted([k, j])
            ind.append(k_j)

    U = np.eye(m, dtype=np.complex128)

    for i in range(n_bs):

        U = (
            U
            @ ps_1(phi, ind[i], i, m)
            @ bs(eta, i, ind[i], m)
            @ ps_2(psi, ind[i], i, m)
        )
        

    if path is not None:

        with open(path + "/parameters_of_interferometer.dat", "w") as ouf:

            ouf.write(
                f"m\tn_BS\n{m}\t{n_bs}\n[n1, n2]\tphi\tpsi\teta\n"
            )

            for z in range(n_bs):
                ouf.write(
                    f"{ind[z][0]}\t{ind[z][1]}\t{phi[z]}\t{psi[z]}\t{eta[z]}\n"
                )

        export_complex_matrix(path + r"/matrix_U.dat", U)
        print("Data were exported to " + path + r"/matrix_U.dat")
        

    return U

def M_matrix(U, A):

    """
    Calculates and returns the Gaussian multi-mode matrix of a GBS scheme.

    Args:
        U (np.ndarray): Interferometer unitary matrix of shape (m, m), where m is the number of modes.
        A (np.ndarray):  Matrix of squeezing parameters of shape (m, m), where m is the number of modes.

    Returns:
        M (np.ndarray): Gaussian multi-mode matrix of shape (m, m), where m is the number of modes.
        
    Raises:
        ValueError: If the dimensions of U and A do not match.
        
    """

    m = len(U)

    M = np.zeros((m, m), dtype=np.complex128)

    for k in range(m):
        for i in range(m):
            for j in range(m):
                M[i, j] += U[k, i] * U[k, j] * A[k, k]

    return M


def average_photon_number(r):
    
    """  
    Calculates and returns the average photon number in a mode
    from a list of squeezing parameters for each mode.

    Args:
        r (list or array): List of squeezing parameters for each mode.

    Returns:
        n_av (float): Average photon number in the mode.
    """
        
    n_av  = 0
    n = len(r)
    
    for r_i in r:
        n_av += np.sinh(r_i)**2/n
        
    return n_av


# Import/Export

def export_input(path, r_s, phi_s, A, ind, phi, psi, eta, n_bs, U, M, n, n_mc=10**5, n_cutoff=0, batch_size=10**3):

    """
    Exports four files containing the following data:
    1) Gaussian multi-mode matrix of a GBS scheme (GBS_matrix.dat); the odd columns contain the real parts of matrix elements, 
    the even columns contain the imaginary parts of matrix elements);
    2) initial state data (initial_state.dat);
    3) parameters of interferometer, which can be used to reconstruct the interferometer matrix (parameters_of_interferometer.dat);
    4) the interferometer unitary matrix (matrix_U.dat).

    Args:
        r_s (ndarray): Array of size m containing the squeezing parameters of the Gaussian modes.
        phi_s (ndarray): Array of size m containing the phase parameters of the Gaussian modes.
        A (ndarray): Array of size m x m containing the matrix elements of the Gaussian multi-mode matrix.
        ind (ndarray): Array of size n_bs x 2 containing the indices of the interferometer modes.
        phi (ndarray): Array of size n_bs containing the angles of the interferometer modes.
        psi (ndarray): Array of size n_bs containing the angles of the interferometer modes..
        eta (ndarray): Array of size n_bs containing the angles of the interferometer modes.
        n_bs (int): Number of beamsplitters in the interferometer.
        U (ndarray): Array of size m x m containing the unitary matrix of the interferometer.
        M (ndarray): Array of size m x m containing the multi-mode matrix of the GBS scheme.
        n (int): Number of nonzero elements in r_s.
        path (str): Path where the output files will be saved.

    Returns:
        None. Prints a message indicating the data has been exported to the specified path.
    """
    
    m = len(M)
    n_ps = int(n_bs*2)
    n = np.count_nonzero(np.array(r_s))
    #n_cutoff=average_photon_number(r_s)

    with open(path + r"/initial_state.dat", "w") as ouf:
        
        ouf.write("N\tr\tphi\tA_real\tA_imag\n")

        for k in range(A.shape[0]):
            ouf.write(
                f"{k}\t{r_s[k]}\t{phi_s[k]}\t{A[k, k].real}\t{A[k, k].imag}\n"
            )
            
    with open(path + "/parameters_of_interferometer.dat", "w") as ouf:

        ouf.write(
            f"m\tn_BS\n{m}\t{n_bs}\n[n1, n2]\tphi\tpsi\teta\n"
        )
        
        for z in range(n_bs):
            ouf.write(
                f"{ind[z][0]}\t{ind[z][1]}\t{phi[z]}\t{psi[z]}\t{eta[z]}\n"
            )
            
                
    with open(path + "/GBS_matrix.dat", "w") as ouf:
        
        ouf.write(
            f"{m}\t{n}\t{r_s[0]}\n"
        )
        
        # f"{m}\t{n}\t{r_s[0]}\t{n_cutoff}\t{n_mc}\t{batch_size}\n"
        
        for k in range(m):
            for j in range(m):
                ouf.write(str(M[k, j].real) + "\t" + str(M[k, j].imag) + "\t")
            if k < (m + 1):
                ouf.write("\n")
   
    export_complex_matrix(path + r"/matrix_U.dat", U)


    return print("Data were exported to " + path)

def export_complex_matrix(file_path, M):
    
    """
    Exports a complex matrix M to a tab-separated text file at file_path.

    Args:
        file_path (str): The file path to write the exported data to.
        M (np.ndarray): A complex matrix to be exported.

    Returns:
        str: A message indicating the file path where data were exported.
    """
    
    with open(file_path, "w") as ouf:
        for k in range(M.shape[0]):
            for j in range(M.shape[1]):
                ouf.write(str(M[k, j].real) + "\t" + str(M[k, j].imag) + "\t")
            if k < (M.shape[0] + 1):
                ouf.write("\n")
    
    
def import_input(path, file_name):
    
    """
    Imports input data for a Gaussian multi-mode matrix of a GBS (Gaussian Boson Sampling) scheme.
    
    Args:
        path (str): Path to the directory containing the input file.
        file_name (str): Name of the input file.
        
    Returns:
        tuple: A tuple containing the following elements:
            - M (numpy.ndarray): Matrix of shape (m, m) representing the Gaussian multi-mode matrix.
            - m (int): Number of modes in the Gaussian multi-mode matrix.
            - n (int): Number of input photons for the GBS scheme.
            - r (float): Parameter r for the GBS scheme.
            
    Raises:
        FileNotFoundError: If the specified input file is not found at the given path.
        ValueError: If the input file is not in the expected format or contains invalid data.
    """
   
    data_M = np.genfromtxt(path + file_name, skip_header=1)

    m = len(data_M)
    
    data_ = np.genfromtxt(path + file_name, skip_footer = m )
    
    n, r = int(data_[1]), data_[2] # int(data_[3]), int(data_[4]), int(data_[5])
    #n, r, n_cutoff, n_mc, batch_size
    M = np.zeros((m, m), dtype=np.complex128)

    real_part = []
    imaginary_part = []

    for i in range(m):
        for k in range(0, 2 * m, 2):
            real_part.append(data_M[i, k])

    for i in range(m):
        for k in range(1, 2 * m + 1, 2):
            imaginary_part.append(data_M[i, k])

    for i in range(m**2):
        M[i // m, i % m] = real_part[i] + 1j * imaginary_part[i]

    #print("Data were imported from " + path + file_name)
    
    #M, m, n, r, n_cutoff, n_mc, batch_size 

    return M, m, n, r

def import_interferometer(path, file_name):

    """
    Imports the interferometer matrix of a GBS scheme.
    
    Args:
        path (str): The path to the directory where the file is located.
        file_name (str): The name of the file to be imported.
        
    Returns:
        numpy.ndarray: The imported matrix as a numpy array with complex128 dtype.
    """
   
    data_U = np.genfromtxt(path + file_name)

    m = len(data_U)
    
    U = np.zeros((m, m), dtype=np.complex128)

    real_part = []
    imaginary_part = []

    for i in range(m):
        for k in range(0, 2 * m, 2):
            real_part.append(data_U[i, k])

    for i in range(m):
        for k in range(1, 2 * m + 1, 2):
            imaginary_part.append(data_U[i, k])

    for i in range(m**2):
        U[i // m, i % m] = real_part[i] + 1j * imaginary_part[i]

    #print("Data were imported from " + path + file_name)

    return U

def import_complex_matrix(path, file_name):

    """
    Imports complex matrix.
    
    Args:
        path (str): The path to the directory where the file is located.
        file_name (str): The name of the file to be imported.
        
    Returns:
        numpy.ndarray: The imported matrix as a numpy array with complex128 dtype.
    """
   
    data_U = np.genfromtxt(path + file_name)

    m = len(data_U)
    
    U = np.zeros((m, m), dtype=np.complex128)

    real_part = []
    imaginary_part = []

    for i in range(m):
        for k in range(0, 2 * m, 2):
            real_part.append(data_U[i, k])

    for i in range(m):
        for k in range(1, 2 * m + 1, 2):
            imaginary_part.append(data_U[i, k])

    for i in range(m**2):
        U[i // m, i % m] = real_part[i] + 1j * imaginary_part[i]

    #print("Data were imported from " + path + file_name)

    return U


def import_parameters_interferometer(path, file_name):
    
    """
    Imports interferometer parameters from a file.

    Args:
        path (str): Path to the directory containing the file.
        file_name (str): Name of the file containing the parameters.

    Returns:
        Tuple: A tuple containing the following arrays:
        - ind (np.ndarray): An array of shape (n_bs, 2) containing pairs of indices
                            representing the beamsplitters in the interferometer.
        - phi (np.ndarray): An array of shape (n_bs,) containing the values of the first
                            phase shifters in the interferometer.
        - psi (np.ndarray): An array of shape (n_bs,) containing the values of the second
                            phase shifters in the interferometer.
        - eta (np.ndarray): An array of shape (n_bs,) containing the values of the
                            beamsplitters in the interferometer.
        - n_bs (int): Number of beamsplitters.
        - m (int): Number of modes. 
    """
   
    data = np.genfromtxt(path + file_name, skip_header=2)
    ind_1 = data[1:, 0].astype(int)
    ind_2 = data[1:, 1].astype(int)
    ind = np.stack((ind_1, ind_2), axis=-1)
    phi = data[1:, 2]
    psi = data[1:, 3]
    eta = data[1:, 4]
    
    data_header = np.genfromtxt(path + file_name, skip_header=1, skip_footer = len(data+1))
    
    m, n_bs = int(data_header[0]), int(data_header[1]) 
    
    return ind, phi, psi, eta, n_bs, m 
    
def import_initial_state(path, file_name):
    
    data = np.genfromtxt(path + file_name, skip_header=1)
    r_lst = list(data[:,1])
    phi_lst = list(data[:,2])
    
    return r_lst, phi_lst
    
def set_device_parameters(r, A, U, path=None):
    
    n_mc = 0
    n_cutoff = 0 
    m = len(U) 
    n = sum(np.diagonal(A)==0)
    batch_size = 0
    
    M = M_matrix(U, A) 
    
    if path is not None:
        
        with open(path + "/GBS_matrix.dat", "w") as ouf:

            ouf.write(str(m) + "\t" + str(n)+ "\t" + str(r) +
                      "\t" + str(n_cutoff)+ "\t" + str(n_mc) + "\t" + str(batch_size) + "\n") 

            for k in range(m):
                for j in range(m):
                    ouf.write(str(M[k, j].real) + "\t" + str(M[k, j].imag) + "\t")
                if k < (m + 1):
                    ouf.write("\n")

        print("Data were exported to " + path + "/GBS_matrix.dat")
        

    return M 

# Generate samples 
def uniform_sampling(batch_size, sample_length, n_photons):
    
    """
    Generates samples from the uniform distribution using the Mersenne Twister algorithm from the random module.
    It gives pseudo-random uniformly distributed bits  (probability 1/2 producing 0 and 1/2 producing 1) on 
    sample_length places and it adds iteratively integers to nonzero elements up to n_photons also randomly. 
    Stops when the number of samples is equal to batch_size. 
    
    Args:
        batch_size (int): Number of samples to generate.
        sample_length (int): Length of each sample, i.e., the total number of photon modes.
        n_photons (int): Number of photons to be placed in the initial seed sample.
        
    Returns:
        numpy.ndarray: Array of shape (batch_size, sample_length) containing the uniform samples.
        
    Raises:
        RuntimeError: If the maximum number of iterations is reached without generating enough samples.
    """
    
    samples = [] 
    max_iterations = 10000  # Maximum number of iterations to avoid infinite loop
    
   
    while len(samples) < batch_size and max_iterations > 0:
        
        seed_sample = [random.getrandbits(1) for i in range(n_photons)] + [0]*int(sample_length - n_photons) 
        
        seed_sample = random.sample(seed_sample, len(seed_sample))
        

        add_ph = n_photons - seed_sample.count(1)
        
        uniform_sample = np.copy(seed_sample)
        
        ind_list = [index for (index, item) in enumerate(uniform_sample) if item == 1]
        
        counter = 0 
        
        if ind_list != []:
        
            while counter < add_ph:

                place = random.choice(ind_list)

                uniform_sample[place] += 1

                counter += 1
                    
            samples.append(uniform_sample)
            
        max_iterations -= 1
        
    if max_iterations <= 0:
        raise RuntimeError(f"Unable to generate {batch_size} samples within the given parameters.")

    return np.array(samples)



def uniform_sampling_tr(batch_size, n, m):
    
    """Performs uniform sampling of batch_size number of sequences of length m with n clicked elements.

        Args:
            batch_size (int): The number of samples.
            n (int): The number of clicked detectors.
            m (int): The length of each sample (number of modes).

        Returns:
            A list of batch_size number of sequences of length m with n clicked detectors.

        Raises:
            ValueError: If n is greater than m or less than 0.

        Example:
            >>> uniform_sampling_tr(2, 5, 2)
            [[1, 1, 0, 0, 0], [1, 0, 0, 0, 1]]

        """ 
    
    samples = []
    
    if n<=m and n>0:
        for i in range(batch_size):
            
            n_clicked_det = random.sample(range(m), k=n)

            seed = convert_0123_01(n_clicked_det,m)
            
            samples.append(seed)
            
    elif n==0:
        for i in range(batch_size):
            samples.append([0]*(m+1))
        
    else:
        raise ValueError(f"Number of clicked detectors n can't be larger than the number of modes m. ")
    return samples
    

def choose_default_device(m, r, path=None):
    
    """
    Initializes the Gaussian Boson Sampling (GBS) device by setting up the input state, interferometer matrix,
    and other parameters for the simulation.
    
    Args:
        m (int): The number of modes in the GBS device.
        r (float): The squeezing parameter for the input state.
        path (str, optional): A path to the directory. 
        
    Returns:
        tuple: A tuple containing two elements:
            - numpy array: The Gaussian matrix M used in the GBS simulation.
            - numpy array: The interferometer matrix U used in the GBS simulation.
    """
    
    
    # Input initialization
    n=round(m/2)
    r_s, phi_s = input_state(r, m, n)
    A = input_matrix(r_s, phi_s, m, n)
    n_bs=m**2
    
    # Interferometer initialization
    U, ind, phi, psi, eta = interferometer(n_bs, m)
        
    # The GBS device initializtion
    M = M_matrix(U, A)
    
    # Export all files related to the simulation
    if path is not None:
        export_input(path, r_s, phi_s, A, ind, phi, psi, eta, n_bs, U, M, n)
        print("Data were exported " 
              + path 
              + r" in files initial_state.dat, parameters_of_interferometer.dat, matrix_U.dat, GBS_matrix.dat ")
        
    
 
    return M, U

def import_gbs_output(path):

    """
    Imports the output of the Gaussian Boson Sampling (GBS) emulator.
    
    Args:
        path (str): The file path to the directory containing the GBS output files.
        
    Returns:
        tuple: A tuple containing two elements:
            - numpy array: An array of samples obtained from the GBS emulator, where each sample is represented
                          as a list of integers.
            - list: A list of probabilities associated with each sample in the samples array.
    """

    samples_data = np.genfromtxt(path + r"/samples.dat", dtype=str)
    samples_  = samples_data[:,0]
    samples = []
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    dir_alphabet = { alphabet[i]: 10+i for i in range(len(alphabet))}
    
    for s in samples_:
        sample = []
        for i in s:
            if i not in alphabet:
                sample.append(int(i))
            else:
                sample.append(dir_alphabet[str(i)])
                
        samples.append(sample)
    
    probs = []
    
    for i in range(len(samples_data)):
        probs.append(float(samples_data[i,1]))

    #test = np.genfromtxt(path + r"/test.dat", dtype=str)
    
    return np.array(samples), probs   #, test


def convert_01_0123(list_det):
    
    """
    Converts a binary list (0 or 1) representing indices from a 0-1 representation to a 0-1-2-3 representation.
    
    Args:
        list_det (list): A binary list (0 or 1) representing the indices in the 0-1 representation.
        
    Returns:
        list: A list of indices in the 0-1-2-3 representation, obtained by finding the indices in list_det
              where the value is 1 and returning them in a new list.
              
    Example:
        >>> [0,1,1,0] 
        <<< [1,2]  
    """

    list_det_ = [i for i in range(len(list_det)) if list_det[i] == 1]

    return list_det_


def convert_0123_01(indices, m):
    
    """
    Converts a list of indices from a 0-1-2-3 representation to a 0-1 representation.
    
    Args:
        indices (list): A list of indices in the 0-1-2-3 representation.
        m (int): The total number of indices, i.e., the size of the output list.
        
    Returns:
        list: A list representing the indices in the 0-1 representation.
        The length of the output list is equal to m, and the values are to
        1 or 0 according to indices.
        
    Example:
        >>> [1,2]
        <<< [0,1,1,0] 
    """

    return  [1 if j in indices else 0 for j in range(m)]


def red_mat(M_big, list_det):  
    
    """
    Extracts a reduced matrix from a larger matrix based on a list of indices.
    
    Args:
        M_big (np.ndarray): The larger input matrix.
        list_det (list): A list of indices specifying the desired rows and columns of the reduced matrix.
        
    Returns:
        np.ndarray: A reduced matrix obtained from M_big by selecting rows and columns based on list_det.
                    The shape of the reduced matrix is (n, n), where n is the length of list_det.
    """


    n = len(list_det)

    small_mat = np.zeros((n, n), dtype=np.complex128)

    # [0,+,0,+] == [1,3]

    for i in range(n):
        for j in range(n):
            ind_i = list_det[i]
            ind_j = list_det[j]
            small_mat[i, j] = M_big[ind_i, ind_j]

    return small_mat

 

def number_of_perm(n,m):
    """
    Calculates the number of permutations of n items taken m at a time.
    
    Args:
        n (int): The total number of items.
        m (int): The number of items taken at a time.
        
    Returns:
        int: The number of permutations of n items taken m at a time, rounded to the nearest integer. 
             If n is less than or equal to m, returns 0.
    """
    if n > m: 
        return round(factorial(n)/factorial(n-m))  
    else:     
        return 0

def number_of_comb(n,m):
    
    """
    Calculates the number of combinations of n items taken m at a time.
    
    Args:
        n (int): The total number of items.
        m (int): The number of items taken at a time.
        
    Returns:
        int: The number of combinations of n items taken m at a time, rounded to the nearest integer.
             If n is less than or equal to m, returns 0.
    """
    
    if n > m:
        return round(factorial(n)/(factorial(n-m)*factorial(m)))
    else:
        return 0
    
    
    
def frobenius_distance(A, B):
    
    """
    Calculates the Frobenius distance between two matrices A and B.

    Args:
        A (numpy array): A matrix.
        B (numpy array): Another matrix with the same shape as A.

    Returns:
        Frobenius distance between A and B.

    Raises:
        ValueError: If matrices A and B do not have the same shape.
    """
    
    if len(A)==len(B):
        return ((((A-B).T.conj()@(A-B)).trace())**0.5).real
    else:
        raise ValueError(f"Input matrices must have the same shape.")
        
def convert_str_to_list(sample):
    """
    Maps sample from list to string in the next way: 
    
    Input: '101'
    Output: [1,0,1]
    """
    
    s = [int(element) for element in list(sample)]
    
    return s

def convert_list_to_str(sample):
    
    """
    Maps sample from list to string in the next way:
    
    Input: [1,0,1]
    Output: '101'
    """
    
    s = (','.join(map(str, sample)).replace(',',''))
    
    return s

    
def export_samples(samples, path, file_name):
    
    """Exports a list of lists to a text file.

        Args:
            samples (list): A list of lists to export.
            path (str): The file path where the text file will be created.
            file_name (str): The name of the text file to be created.

        Returns:
            A string indicating the file path and name of the exported text file.

        """
    with open(path+file_name, 'w') as ouf:
        ouf.writelines(','.join(map(str, s)).replace(',','') + '\n' for s in samples)
            
    return "Data were exported to " + path + file_name


def import_samples(path, file_name):
    
    """Imports a nested list of samples from a text file.

        Args:
            path (str): The file path where the text file is located.
            file_name (str): The name of the text file to import.

        Returns:
            A nested list of samples imported from the text file.
        """ 
    
    samples_data = np.loadtxt(path + "/samples.dat", dtype=str,ndmin=1)
    
    samples = []

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    dir_alphabet = {alphabet[i]: 10 + i for i in range(len(alphabet))}

    
    for s in samples_data:
        sample = []
        for i in s:
            if i.isdigit():  # Check if the character is a digit
                sample.append(int(i))
            elif i.isalpha():  # Check if the character is an alphabet letter
                sample.append(dir_alphabet[i])  # Changed str(i) to i
        samples.append(sample)
            
    return samples

def submatriсes_export(M, samples, path):
    
    """
    Exports complex submatrices and their corresponding sample ids to a specified path.

    Args:
        M (complex matrix): The GBS complex matrix.
        samples (list): A list of samples.
        path (str): The path where the submatrices and sample ids will be exported.

    Returns:
        str: A message indicating the successful export of submatrices and sample ids.
    """
    
    batch_of_samples = samples
    file_path = path + '/input'

    for i in range(len(batch_of_samples)):
        clicked_detectors = convert_01_0123(batch_of_samples[i])    
        M_sub = red_mat(M, clicked_detectors)
        export_complex_matrix(file_path + f'/Submatrix_{i}.dat', M_sub)

    # Export ids and samples 
    # Submatrices are saved according samples ids (id is a serial number in a batch of samples)
    with open(file_path+'/samples_ids.dat', 'w') as ouf:
        ouf.writelines(
            (
                (str(i) + "\t"
                 +','.join(map(str, batch_of_samples[i])).replace(',',''))
                + '\n' for i in range(len(batch_of_samples))
            )
        )
        
    return f'Submatrices and their ids were exported to {file_path}'




def fock_basis_size(n,m):
    
    """
    Calculates the size of the Fock basis for a given number of photons and modes.

    The Fock basis size represents the number of possible states in a quantum system
    with 'n' photons in 'm' modes.

    Args:
        n (int): Number of particles in the system.
        m (int): Number of modes in the system.

    Returns:
        int: The size of the Fock basis.

    Examples:
        >>> fock_basis_size(2, 3)
        10
        >>> fock_basis_size(3, 2)
        6
    """
    
    return number_of_comb(m+n-1,n)

def permut(lst):
    """
    Generates all permutations of a given list.

    This function takes a list 'lst' and generates all possible permutations
    of its elements. Each permutation is returned as a list.

    Args:
        lst (list): The input list for which permutations are to be generated.

    Yields:
        list: A permutation of the input list.

    Returns:
        list: A list of all permutations generated.

    Raises:
        None.

    Examples:
        >>> list(permut([1, 2, 3]))
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        >>> list(permut(['a', 'b']))
        [['a', 'b'], ['b', 'a']]
    """
        
    if len(lst) == 0:
        yield []
        #return []
        
    elif len(lst) == 1:
        yield lst
        
        #return [lst]
    else:
        l = []
        prev_x = []
        
        for i in range(len(lst)):
            x = lst[i]
            xs = lst[:i] + lst[i+1:]
            
            if not x in prev_x:
                prev_x.append(x)
                for p in permut(xs):
                    yield [x]+p
                    #l.append([x]+p)
        return l
    
def find_partition(k,N):
    """
    Finds all partitions of an integer 'k' with parts no larger than 'N'.

    This function takes an integer 'k' and an upper bound 'N', and returns
    a list of all possible partitions of 'k' into positive integers, where
    each part is no larger than 'N'. A partition is represented as a list of
    positive integers.

    Args:
        k (int): The integer to be partitioned.
        N (int): The upper bound for the parts in the partition.

    Returns:
        list: A list of all partitions of 'k' with parts no larger than 'N'.

    """

    res = []

    if k == 0:
        return [[]]

    elif k == 1:
        return [[1]]

    i_max = min(k, N)

    for i in range(1, i_max + 1):
        res_i = find_partition(k - i, i)

        for a in res_i:
            res.append([i] + a)

    return res
        

def gen_states(N, m):
    """
    Generates possible distrubution of N photons in m modes.

    This function takes the number of particles 'N' and the number of modes 'm'
    and generates all possible occupied states (without permutations). 
    A state is represented as a list of non-negative integers, where the sum 
    of the integers is equal to 'N' and the length of the state is equal to 'm'. 

    Args:
        N (int): The number of photons.
        m (int): The number of modes.

    Returns:
        list: A list of all valid states for the given 'N' and 'm'.

    Raises:
        ValueError: If 'N' or 'm' is not a positive integer.

    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer.")

    res = []

    states = find_partition(N, N)

    for s in states:
        if len(s) < m:
            zer = m - len(s)
            res.append(s + [0] * zer)
        else:
            res.append(s)

    for r in res:
        if len(r) > m:
            res.remove(r)

    return res            
        
def fock_basis_choice(N,m):
    
    """
    Generates the Fock basis choice for a given number of particles and modes.

    This function takes the number of particles 'N' and the number of modes 'm'
    and generates the Fock basis choice. The Fock basis choice represents all
    possible permutations of states where the sum of the integers is equal to 'N'
    and the length of the state is equal to 'm'. The function utilizes the 'gen_states'
    and 'permut' functions to generate the Fock basis choice.

    Args:
        N (int): The number of particles.
        m (int): The number of modes.

    Returns:
        list: The Fock basis choice, which is a list of all possible permutations of
              states that satisfy the given 'N' and 'm'.

    Raises:
        ValueError: If 'N' or 'm' is not a positive integer.

    Examples:
        >>> fock_basis_choice(3, 2)
        [[2, 1], [1, 2], [3, 0], [0, 3]]
        >>> fock_basis_choice(2, 3)
        [[1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer.")
        
    
    basis = []
    
    states = gen_states(N,m)
    
    for s in states:
        for p in permut(s):
            basis.append(p)
        
    return basis

def prn_to_tr_detectors(samples):
    
    """
    Convert a nested list of PRN-detected samples to a list of threshold-detected samples.

    Parameters:
    samples (list): A nested list of samples with PRN detection. 

    Returns:
    list: A nested list of samples with threshold detection. 
    """
    return [[1 if item > 1 else item for item in s] for s in samples]



def threshold_basis_set(m):
    
    """
    Generates the threshold basis set for a given number of modes.

    This function takes the number of modes 'm' and generates the threshold basis set.
    The threshold basis set represents all possible permutations of states with 'm'
    modes, where each state consists of 1's and 0's.

    Args:
        m (int): The number of modes.

    Returns:
        list: The threshold basis set, which is a list of all possible permutations of
              states consisting of 1's followed by 0's for the given 'm'.

    Raises:
        ValueError: If 'm' is not a positive integer.

    Examples:
        >>> threshold_basis_set(2)
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> threshold_basis_set(3)
        [[0, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 1],
         [0, 1, 1],
         [1, 1, 1]]
    """
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer.")
        
    all_permutations = []

    for i in range(m+1):

        all_permutations.extend(
            list(permut([1]*i + [0]*(m-i)))
        )
    return all_permutations


def prob_exact(sample, M):
    
    """
    Calculates the exact probability for a given sample and Gaussian matrix.

    This function takes a sample and a matrix, and calculates the exact probability
    based on the given sample. The sample is converted using the 'convert_01_0123' function,
    and a reduced matrix is obtained using the 'red_mat' function.

    Args:
        sample (list): The sample to calculate the probability for.
        M (numpy.ndarray): The matrix for computation.

    Returns:
        float: The exact probability for the given sample and matrix.
        
    """

    
    clicked_detectors = convert_01_0123(sample)    
    M_sub = red_mat(M, clicked_detectors)
    
    m = len(M_sub)
    stat =  np.zeros((m+1), dtype = np.float64) 
    
    for i in range(m+1):
        detect_event = [1 for j in range(i)] + [0]*(m-i) 
        permutations = list(permut(detect_event))

        if i == 0:
            stat[i] += 1

        else:
            for s in permutations:
                stat[i] += Z_i(s, M_sub,nu=0)


    for c in range(m):
        for h in range(c+1, m+1):
            stat[h] -= (
                stat[c]*number_of_comb(m - c, m - h)
            )
    
    if len(M_sub) != len(M): 
        norm = Z(M_sub)/Z(M)
        probability = norm*stat[m]/sum(stat)    
    else:
        if np.allclose(M_sub, M):
            probability = stat[m]/sum(stat) 
        else:
            raise 'Error'
    
    return probability


# @njit
def Z(M, nu=0):
    
    """
    Calculates the partition function value for a given matrix.

    This function takes a matrix 'M' and calculates the the partition function
    based on its eigenvalues. The matrix is first transformed into its Hermitian
    conjugate, and the eigenvalues are computed. 
    If `nu!=0` computes partition function for sectors. 

    Args:
        M (numpy.ndarray): The matrix for computation.
        nu (float, optional): Coordinate of a sector. Defaults to 0.

    Returns:
        if nu != 0:
            complex: The partition function value for a sector.
        else:
            float: The partition function value for the given matrix.

    """
    
    M_hermitian = M.conjugate().T@M
    
    eig = np.linalg.eigh(M_hermitian)[0]
    
    z = 1.0 
    for i in range(len(eig)):
        if nu!=0:
            p = 4 * eig[i] * np.exp(1j * nu)
        else:
            p = 4 * eig[i]
            
        z *= (1 - p)**(-0.5)
            
    return z


def Z_i(sample, M, nu=0):
    
    """
    Calculates the partition function value for a specific sample.

    This function takes a sample, a Gaussian matrix 'M', and an optional parameter 'nu',
    and calculates the the partition function. The sample is
    converted using the 'convert_01_0123' function, and a reduced matrix is
    obtained using the 'red_mat' function. 
    If `nu!=0` computes partition function for sectors. 

    Args:
        sample (list): The sample for computation.
        M (numpy.ndarray): The matrix for computation.
        nu (float, optional): The parameter for Z calculation. Defaults to 0.

    Returns:
        if nu != 0:
            complex: The partition function value of the sample and the sector.
        else:
            float: The partition function value of the sample.
    """
    
    clicked_detectors = convert_01_0123(sample)            
    M_sub = red_mat(M, clicked_detectors)
    z = Z(M_sub,nu)
    
    return z

def prob_sectors_exact(M, sample):
    
    """
    Calculates the exact probabilities over sectors for a given sample 
    and Gaussian matrix.

    Args:
        M (numpy.ndarray): The Gaussian matrix for computation.
        sample (list, optional): The sample to calculate the probabilities for. Defaults to [1]*len(M).
    
    Returns:
        numpy.ndarray: The exact probabilities over sectors for the given sample and Gaussian matrix.

        
    """
 
    
    clicked_detectors = convert_01_0123(sample)    
    M_sub = red_mat(M, clicked_detectors)
    
    nu_max=10*len(M_sub)
    dnu = 2 * np.pi / nu_max
    m = len(M_sub)

    stat = np.zeros((m + 1, nu_max), dtype=np.complex128)
    sectors = np.zeros((m + 1, nu_max), dtype=np.float64)
                    
    for i in range(m+1):
        detect_event = [1 for j in range(i)] + [0]*(m-i) 
        permutations = list(permut(detect_event))

        if i == 0:
            for nu in range(nu_max):
                stat[i,nu] += 1

        else:
            for nu in range(nu_max):
                for s in permutations:
                    stat[i,nu] += Z_i(s, M_sub, nu=nu*dnu) 
                    
    for k in range(m):
        for h in range(k + 1, m + 1):
            for nu in range(nu_max):
                stat[h, nu] -= stat[k, nu] * number_of_comb(m - k, m - h)

    for n in range(m + 1):
        for j in range(nu_max):
            for k in range(nu_max):
                sectors[n, j] += (stat[n, k]*np.exp(-1j*j*k*dnu)/nu_max).real
                
    return sectors/Z(M)

def compute_minors(path=return_path):
    """
    Computes minors for all samples and saves the results in the specified path.
    In the specified directory it needs to have directories with names `input` 
    and `output`. `input` contains files `samples_ids.dat` and `Submatrix_i.dat` 
    (`i` is an integer index specified in  `samples_ids.dat`). 

    Args:
        path (str): The path to `input` and `output` directories.

    Returns:
        str: A message confirming the completion of minor computation for all samples.
    """

    # It is a script which 0) import ids
    # 1) takes file input/Submatrix_1.dat and copies it as input/Submatrix.dat 
    # 2) runs Minors.cpp 
    # 3) saves the output files output/Minors_01.dat as output/Minors_01_1.dat
    
    data_ids = np.genfromtxt(path + '/input/samples_ids.dat', dtype=str)

    ids = [int(i) for i in data_ids[:,0]]
    files = (
            [
                '/output/Minors0-1.dat',
                '/output/Minors2.dat', 
                '/output/Minors3.dat', 
                '/output/Minors4.dat',
                '/input/Submatrix.dat'
            ])

    for i in ids:
        shutil.copy(path+f'/input/Submatrix_{i}.dat', path+'/input/Submatrix.dat')

        # cmd = "cpp/Minors.cpp"
        # subprocess.call(["g++", cmd])
        subprocess.call("./a.out") 
        #print("Finished:", cmd.split("/")[1], f"for sample #{i}")

        shutil.copy(path+f'/output/Minors0-1.dat', path+f'/output/Minors0-1_{i}.dat')
        shutil.copy(path+f'/output/Minors2.dat', path+f'/output/Minors2_{i}.dat')
        shutil.copy(path+f'/output/Minors3.dat', path+f'/output/Minors3_{i}.dat')
        shutil.copy(path+f'/output/Minors4.dat', path+f'/output/Minors4_{i}.dat')

        for f in files:
            if os.path.isfile(path + f):
                os.remove(path + f)
            else:
                print("Error: %s file not found" % path + f)
                
    return print(f"Minors for all {len(ids)} samples are computed.")


class MomentUtility():
    
    def __init__(
        self,
        id_,
        n_moments=4,
        path=return_path(filename='tutorial.ipynb'),
    ): 
        self.id_ = id_
        self.path = path
        self.n_moments = n_moments
        self.matrix = import_complex_matrix(path, f"/input/Submatrix_{id_}.dat") #attribute (a thing which is stored in our class)
        self.n_modes = len(self.matrix)
        self.n_sector_max = round(10*self.n_modes)
        self.n_sector_step = 2*np.pi/self.n_sector_max
        
        
    def export_minors(self):
        """
        Computes minors for a sample and saves the results in the specified path.
        In the specified directory it needs to have directories with names `input` 
        and `output`. `input` contains files `samples_ids.dat` and `Submatrix_i.dat` 
        (`i` is an integer index specified in  `samples_ids.dat`). 

        Returns:
            str: A message confirming the completion of minor computation for a sample. 
        """
        index = self.id_
        path = self.path

        files = (
                [
                    '/Minors0-1.dat',
                    '/Minors2.dat', 
                    '/Minors3.dat', 
                    '/Minors4.dat',
                    '/Submatrix.dat'
                ])

        path_level_down = os.path.split(path)[0]
        shutil.copy(path+f'/input/Submatrix_{index}.dat', path_level_down + files[4])
        
        cmd = "cpp/Minors.cpp"
        #subprocess.call(["g++", cmd])
        subprocess.call("./a.out") 
        #print("Finished:", cmd.split("/")[1], f"for sample #{index}")
        
        #print(path_level_down+files[0], path+f'/output/Minors0-1_{index}.dat' )

        shutil.copy(path_level_down+files[0], path+f'/output/Minors0-1_{index}.dat')
        shutil.copy(path_level_down+files[1], path+f'/output/Minors2_{index}.dat')
        shutil.copy(path_level_down+files[2], path+f'/output/Minors3_{index}.dat')
        shutil.copy(path_level_down+files[3], path+f'/output/Minors4_{index}.dat')

        for f in files:
            if os.path.isfile(path_level_down + f):
                os.remove(path_level_down + f)
            else:
                print("Error: %s file not found" % path_level_down + f)

        return f"Minors for the sample #{index} are computed."



    def moment_formula(self, n, *args):
    
        m = 0 

        for x in args:
                moments = x

        if n == 2:
            m = moments[0] + 2*moments[1] - moments[0]**2

        if n == 3:
            m = (moments[0] + 6*moments[1] + 6*moments[2] 
                 - 3*moments[0]*(moments[0] + 2*moments[1]) 
                 +  2*moments[0]**3)


        if n == 4:
            m_2 = moments[0] + 2*moments[1]

            m_3 = moments[0] + 6*moments[1] + 6*moments[2]

            m_4 = moments[0] + 14*moments[1] + 36*moments[2] + 24*moments[3]

            m =  m_4 - 4*m_3*moments[0]- 3*m_2**2 + 12*m_2*moments[0]**2 - 6*moments[0]**4

        if n == 5:

            m_2 = moments[0] + 2*moments[1]

            m_3 = moments[0] + 6*moments[1] + 6*moments[2]

            m_4 = moments[0] + 14*moments[1] + 36*moments[2] + 24*moments[3]

            m_5 = moments[0] + 30*moments[1] + 200*moments[2] + 40*moments[3] + 120*moments[4]

            m = (m_5 + 5*moments[0]*m_4 + 10*m_2*m_3 + 10*m_3*moments[0]**2 + 
                 15*moments[0]*m_2**2 + 10*m_2*moments[0]**3 + moments[0]**5)

        return m
    
    def import_minors(self):
        
        m = self.n_modes
        Nu = self.n_sector_max
        dnu = self.n_sector_step

        # Import Minors
        
        
        # For the fist moment
        data_minors = np.genfromtxt(self.path+f'/output/Minors0-1_{self.id_}.dat')
        Z_v_0 = np.zeros((Nu),dtype=np.complex128)
        Z_v_1 = np.zeros((m, Nu),dtype=np.complex128)
        
        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1:2] + 1j*data_minors[j,2:3]

        for j in range(Nu):
            for n in range(0,2*m,2):
                Z_v_1[n//2,j] =  data_minors[j,int(3+n)] + 1j*data_minors[j,int(4+n)]               
        # FFT 
        Z_v_0f = np.fft.fft(Z_v_0)/Nu
        Z_v_1f = np.fft.fft(Z_v_1)/Nu      
        
        # For the second moment
        data_minors2 = np.genfromtxt(self.path+f'/output/Minors2_{self.id_}.dat')
        p2 = round(factorial(m)/(factorial(m-2)*2)) 
        Z_v_2 = np.zeros((p2, Nu),dtype=np.complex128)
        
        for j in range(Nu):
            for n in range(0,2*p2,2):
                Z_v_2[n//2,j] =  data_minors2[j,int(1+n)] + 1j*data_minors2[j,int(2+n)]             
        # FFT 
        Z_v_2f = np.fft.fft(Z_v_2)/Nu
        
        # For the third moment
        data_minors3 = np.genfromtxt(self.path+f'/output/Minors3_{self.id_}.dat')
        p3 = round(factorial(m)/(factorial(m - 3)*factorial(3)))
        Z_v_3 = np.zeros((p3, Nu),dtype=np.complex128)
        
        for j in range(Nu):
            for n in range(0,2*p3,2):
                Z_v_3[n//2,j] =  data_minors3[j,int(1+(n))] + 1j*data_minors3[j,int(2+(n))] 
        # FFT        
        Z_v_3f = np.fft.fft(Z_v_3)/Nu
        
        # For the fourth moment
        data_minors4 = np.genfromtxt(self.path+f'/output/Minors4_{self.id_}.dat')
        p4 = round(factorial(m)/(factorial(m - 4)*factorial(4)))
        Z_v_4 = np.zeros((p4, Nu),dtype=np.complex128)
        
        for j in range(Nu):
            for n in range(0,2*p4,2):
                Z_v_4[n//2,j] =  data_minors4[j,int(1+(n))] + 1j*data_minors4[j,int(2+(n))] 
        # FFT             
        Z_v_4f = np.fft.fft(Z_v_4)/Nu 
        
        return Z_v_0f, Z_v_1f, Z_v_2f, Z_v_3f, Z_v_4f

 
    def compute_moments(self, Z_v_0f, Z_v_1f, Z_v_2f, Z_v_3f, Z_v_4f):
        
        m = self.n_modes
        Nu = self.n_sector_max
        dnu = self.n_sector_step
        
        mean_ = np.zeros(Nu)
        disp_ = np.zeros(Nu)
        m3_ = np.zeros(Nu)
        m4_ = np.zeros(Nu)


        n_ij_v =  np.zeros(Nu)
        n_ijk_v = np.zeros(Nu)
        n_ijkl_v = np.zeros(Nu)


        ind_2 = []
        ind_3 = []
        ind_4 = []


        for i in range(m):
                for j in range(i+1, m):
                    ind_2.append([i,j]) 

        for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        ind_3.append([i,j,k]) 

        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    for l in range(k+1, m):
                        ind_4.append([i,j,k,l]) 


        for z in range(Nu): 
            for j in range(m):
                mean_[z] += 1 - (Z_v_1f[j,z]/Z_v_0f[z]).real


        for nu in range(Nu):
            i_ = 0
            for i in range(m):
                for j in range(i+1, m):
                    n_ij_v[nu] += 1 - (( Z_v_1f[j,nu] + Z_v_1f[i,nu] - Z_v_2f[i_,nu])/Z_v_0f[nu]).real
                    i_ += 1
            disp_[nu] =  self.moment_formula(2, [mean_[nu], n_ij_v[nu]])


        for nu in range(Nu):
            i_= 0
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):

                        z1 = ind_2.index([i,j])
                        z2 = ind_2.index([i,k])
                        z3 = ind_2.index([j,k])

                        n_ijk_v[nu] += 1 - ((Z_v_1f[i,nu] + Z_v_1f[j,nu] + Z_v_1f[k,nu] - Z_v_2f[z1,nu] - Z_v_2f[z2,nu] - Z_v_2f[z3,nu] + Z_v_3f[i_,nu])/Z_v_0f[nu]).real
                        i_ += 1 

            m3_[nu] = self.moment_formula(3, [mean_[nu], n_ij_v[nu], n_ijk_v[nu]])


        for nu in range(Nu): 
            i_= 0
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        for l in range(k+1, m):

                            z1 = ind_2.index([i,j])
                            z2 = ind_2.index([i,k])
                            z3 = ind_2.index([i,l])

                            z4 = ind_2.index([j,k])
                            z5 = ind_2.index([k,l])
                            z6 = ind_2.index([j,l])

                            h1 = ind_3.index([i,j,k])
                            h2 = ind_3.index([j,k,l])
                            h3 = ind_3.index([i,k,l])
                            h4 = ind_3.index([i,j,l])

                            n_ijkl_v[nu] += 1 - ((Z_v_1f[i,nu] + Z_v_1f[j,nu] + Z_v_1f[k,nu] + Z_v_1f[l,nu] - Z_v_2f[z1,nu] - Z_v_2f[z2,nu] - Z_v_2f[z3,nu] - Z_v_2f[z4,nu] - Z_v_2f[z5,nu] - Z_v_2f[z6,nu] + Z_v_3f[h1,nu] + Z_v_3f[h2,nu] +  Z_v_3f[h3,nu] + Z_v_3f[h4,nu] -  Z_v_4f[i_,nu])/Z_v_0f[nu]).real 

                            i_ += 1 

            m4_[nu] = self.moment_formula(4, [mean_[nu], n_ij_v[nu], n_ijk_v[nu], n_ijkl_v[nu]])

        return mean_ , disp_ , m3_ , m4_ 

    def get_moments(self):
        
        Z_v_0f, Z_v_1f, Z_v_2f, Z_v_3f, Z_v_4f = self.import_minors()

        mean_, disp_, m3_, m4_ = self.compute_moments(Z_v_0f, Z_v_1f, Z_v_2f, Z_v_3f, Z_v_4f)
        
        return mean_, disp_, m3_, m4_


    # Export Moments

    def export_moments(self):
        
        m = self.n_modes
        Nu = self.n_sector_max
           
        mean_, disp_, m3_, m4_ = self.get_moments()
        
        with open(self.path+f"/output/Moments_{self.id_}.dat", 'w') as ouf:
            for nu in range(Nu):
                ouf.write(f"{mean_[nu].real}\t{disp_[nu].real}\t{m3_[nu].real}\t{m4_[nu].real}\t")
                if nu < (Nu+1):
                    ouf.write('\n')
                    
        return f"Moments were exported to {self.path}/output/Moments_{self.id_}.dat"
    
    
class CumulantUtility_old(MomentUtility):
    
    
    def import_moments(self):
        
        m = self.n_modes 
        Nu = self.n_sector_max
        dnu = self.n_sector_step
        


        data_minors = np.genfromtxt(self.path + f'/output/Minors0-1_{self.id_}.dat')

        Z_v_0 = np.zeros((Nu),dtype=np.complex128)

        Z_v_1 = np.zeros((m, Nu),dtype=np.complex128)

        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1] + 1j*data_minors[j,2]

        for j in range(Nu):
            for n in range(0,2*m,2):
                Z_v_1[n//2,j] =  data_minors[j,int(3+n)] + 1j*data_minors[j,int(4+n)] 


        Z_v_0f =  np.zeros((Nu), dtype = np.complex128)
        Z_v_1f = np.zeros((m, Nu), dtype = np.complex128)

        Z_v_0f = np.fft.fft(Z_v_0)/Nu
        Z_v_1f = np.fft.fft(Z_v_1)/Nu

        data_moments = np.genfromtxt(self.path+f'/output/Moments_{self.id_}.dat')

        m0 = (Z_v_0f[:]/Z_v_0[0]).real #normalization
        m1 = data_moments[:, 0]
        m2 = data_moments[:, 1]
        m3 = data_moments[:, 2]
        m4 = data_moments[:, 3]
        
        return m0, m1, m2, m3, m4
        
    def gauss_fun(self, x, *args):

        """
        Computes the Gaussian function with optional additional terms.

        Args:
            x: Numeric input value.
            *args: Variable number of arguments representing the coefficients.

        Returns:
            Numeric result of the Gaussian function computation.

        Examples:
            >>> gauss_fun(2, 1, 0, 1)
            0.1353352832366127
        """

        for c in args:
            c = args

        if len(c) == 3:    

            res = c[0]*np.exp(-(x - c[1])**2/(2*c[2])**2) 

        elif len(c) == 4:

            res = c[0]*np.exp(-(x - c[1])**2/(2*c[2])**2) * np.exp(+ c[3]*(x - c[1])**3/(6*c[2]**3)) 

        elif len(c) == 5:

            res = c[0]*np.exp(-(x - c[1])**2/(2*c[2])**2) * np.exp(+ c[3]*(x - c[1])**3/(6*c[2]**3)) * np.exp(+ c[4]*(x - c[1])**4/(8*c[2]**4))

        return res 
    
    def weighted_error(self, m, m_, coeff):
        
        err = coeff*(m - m_) #0.1*coeff * (m - m_)/m #
        
        return err

    def get_cumulants(self, m0, m1, m2, m3, m4):
    
        # Approximation 
        # The 2nd order approximation 
        m = self.n_modes
        Nu = self.n_sector_max
        n_cut = int(m+10)

        A_2 = np.zeros(Nu)
        Mu1_2 = np.zeros(Nu)
        Mu2_2 = np.zeros(Nu) # , dtype=np.longdouble

        for nu in range(Nu):

            A_2[nu] =  m0[nu] 
            Mu1_2[nu] = m1[nu] 
            Mu2_2[nu] = m2[nu] 


            for z in range(300):
                s0 = 0
                s1 = 0 
                s2 = 0 
                s3 = 0
                s4 = 0

                for j in np.arange(n_cut):

                    s0 +=  self.gauss_fun(j, A_2[nu], Mu1_2[nu], Mu2_2[nu])
                    s1 +=  self.gauss_fun(j, A_2[nu], Mu1_2[nu], Mu2_2[nu])* j
                    s2 +=  self.gauss_fun(j, A_2[nu], Mu1_2[nu], Mu2_2[nu])* j**2

                if s0 > 10**(-15) and s0==s0:

                    m0_ = s0 
                    m1_ = s1/s0 
                    m2_ = s2/s0 - m1_**2
                    
                   

                    A_2[nu] += self.weighted_error(m0[nu], m0_, 0.1) #0.1*(m0[nu]  - m0_ )
                    Mu1_2[nu] += self.weighted_error(m1[nu], m1_, 0.1) # 0.1*(m1[nu] - m1_)
                    Mu2_2[nu] += self.weighted_error(m2[nu], m2_, 0.1) #0.1*(m2[nu] - m2_)  

                else:

                    A_2[nu] += 0
                    Mu1_2[nu] += 0
                    Mu2_2[nu] += 0


        # The 3nd order approximation 

        A_3 = np.zeros(Nu)
        Mu1_3 = np.zeros(Nu)
        Mu2_3 = np.zeros(Nu)
        Mu3_3 = np.zeros(Nu)

        for nu in range(Nu):

            A_3[nu] = m0[nu]
            Mu1_3[nu] = m1[nu] 
            Mu2_3[nu] = m2[nu] 
            Mu3_3[nu] = 0



            for z in range(500):
                s0 = 0
                s1 = 0 
                s2 = 0 
                s3 = 0



                for j in range(n_cut):

                    s0 += self.gauss_fun(j, A_3[nu], Mu1_3[nu], Mu2_3[nu], Mu3_3[nu])
                    s1 += self.gauss_fun(j, A_3[nu], Mu1_3[nu], Mu2_3[nu], Mu3_3[nu])* j
                    s2 += self.gauss_fun(j, A_3[nu], Mu1_3[nu], Mu2_3[nu], Mu3_3[nu])* j**2
                    s3 += self.gauss_fun(j, A_3[nu], Mu1_3[nu], Mu2_3[nu], Mu3_3[nu])* j**3

                # attention! s0 may diverge 

                if s0==s0:


                    m0_ = s0 
                    m1_ = s1/s0 
                    m2_ = s2/s0 - m1_**2
                    m3_ = s3/s0 - 3*m2_*m1_ - m1_**3

                    A_3[nu] += self.weighted_error(m0[nu], m0_, 0.05) #0.05*(m0[nu] - m0_)
                    Mu1_3[nu] += self.weighted_error(m1[nu], m1_, 0.1) #0.1*(m1[nu] - m1_ )
                    Mu2_3[nu] += self.weighted_error(m2[nu], m2_, 0.1) #0.1*(m2[nu] - m2_)  
                    Mu3_3[nu] += self.weighted_error(m3[nu], m3_, 0.05) #0.05*(m3[nu] - m3_) 

                else:
                    A_3[nu] += 0
                    Mu1_3[nu] += 0
                    Mu2_3[nu] += 0
                    Mu3_3[nu] += 0 

        # The 4th order approximation 

        A_4 = np.zeros(Nu)
        Mu1_4 = np.zeros(Nu)
        Mu2_4 = np.zeros(Nu)
        Mu3_4 = np.zeros(Nu)
        Mu4_4 = np.zeros(Nu)


        for nu in range(Nu):

            A_4[nu] = m0[nu]
            Mu1_4[nu] = m1[nu] 
            Mu2_4[nu] = m2[nu] 
            Mu3_4[nu] = 0
            Mu4_4[nu] = 0 

            n_0 = 0

            for z in range(800):

                s0 = 0
                s1 = 0 
                s2 = 0 
                s3 = 0
                s4 = 0


                for j in range(n_0,n_cut):

                    s0 +=  self.gauss_fun(j, A_4[nu], Mu1_4[nu], Mu2_4[nu], Mu3_4[nu], Mu4_4[nu])
                    s1 +=  self.gauss_fun(j, A_4[nu], Mu1_4[nu], Mu2_4[nu], Mu3_4[nu], Mu4_4[nu])* j
                    s2 +=  self.gauss_fun(j, A_4[nu], Mu1_4[nu], Mu2_4[nu], Mu3_4[nu], Mu4_4[nu])* j**2
                    s3 +=  self.gauss_fun(j, A_4[nu], Mu1_4[nu], Mu2_4[nu], Mu3_4[nu], Mu4_4[nu])* j**3
                    s4 +=  self.gauss_fun(j, A_4[nu], Mu1_4[nu], Mu2_4[nu], Mu3_4[nu], Mu4_4[nu])* j**4

                # attention! s0 may diverge 

                if s0==s0:
                    m0_ = s0 
                    m1_ = s1/s0 
                    m2_ = s2/s0 - m1_**2
                    m3_ = s3/s0 - 3*m2_*m1_ - m1_**3
                    m4_ = s4/s0 - 4*m3_*m1_ - 3*m2_**2 - 6*m2_*m1_**2 - m1_**4

                    step_ini_0 =  0.1
                    step_ini_1 =  0.1
                    step_ini_2 =  0.1
                    step_ini_3 =  0.05
                    step_ini_4 =  0.008

                    A_4[nu] +=  self.weighted_error(m0[nu], m0_, step_ini_0) #step_ini_0*(m0[nu] - m0_) 
                    Mu1_4[nu] +=  self.weighted_error(m1[nu], m1_, step_ini_1) #step_ini_1*(m1[nu] - m1_)
                    Mu2_4[nu] +=  self.weighted_error(m2[nu], m2_, step_ini_2) #step_ini_2*(m2[nu] - m2_) 
                    Mu3_4[nu] +=  self.weighted_error(m3[nu], m3_, step_ini_3) #step_ini_3*(m3[nu] - m3_) 
                    Mu4_4[nu] +=  self.weighted_error(m4[nu], m4_, step_ini_4) #step_ini_4*(m4[nu] - m4_)

                else:
                    A_4[nu] += 0
                    Mu1_4[nu] += 0
                    Mu2_4[nu] += 0 
                    Mu3_4[nu] += 0 
                    Mu4_4[nu] += 0
         
                    
        return A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3,  A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4


                
    def prob_approx(self, export_probabilities=True, export_cumulants=True):
        
        m = self.n_modes
        Nu = self.n_sector_max
        m0, m1, m2, m3, m4 = self.import_moments()
        A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3,  A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4 = self.get_cumulants(m0, m1, m2, m3, m4)
        
        data_minors = np.genfromtxt(self.path + f'/output/Minors0-1_{self.id_}.dat')

        Z_v_0 = np.zeros((Nu),dtype=np.complex128)

        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1] + 1j*data_minors[j,2]

        normalization = Z_v_0[0].real


        probability_approx_2 = 0
        probability_approx_3 = 0
        probability_approx_4 = 0
        
        k_0=0
        k_cut=Nu


        for j in range(Nu):
            if self.gauss_fun(m, A_4[j],Mu1_4[j], Mu2_4[j],Mu3_4[j],Mu4_4[j]) > 10**(-15) and A_2[j]!= 0:
                k_cut = j
                


        for j in range(int(Nu/10)):
            if  self.gauss_fun(m, A_4[j],Mu1_4[j], Mu2_4[j],Mu3_4[j],Mu4_4[j] ) < 10**(-15):
                k_0 = j


        # choosing of k_0 and k_cut (cut off of the number of sectors) 
        # is heuristic, it may vary for more accurate computation 

        for k in range(k_0,k_cut):
            probability_approx_2 += self.gauss_fun(m, A_2[k], Mu1_2[k], Mu2_2[k])/normalization 
            probability_approx_3 += self.gauss_fun(m, A_3[k], Mu1_3[k], Mu2_3[k], Mu3_3[k])/normalization
            probability_approx_4 += self.gauss_fun(m, A_4[k], Mu1_4[k], Mu2_4[k], Mu3_4[k], Mu4_4[k])/normalization       
            
        if export_probabilities == True:
            
            
            with open(self.path + f"/output/Result_{self.id_}.dat", 'w') as ouf:
                header = '\t'.join(
                    ['m', 'k_0', 'k_cut', 'p2', 'p3', 'p4']
                )
                ouf.write(header + '\n')

                values = '\t'.join(
                    [str(m), str(k_0), str(k_cut), 
                    str(probability_approx_2), 
                     str(probability_approx_3),
                     str(probability_approx_4)]
                )
                ouf.write(values + '\n')
            
            
        if export_cumulants==True:
            
            with open(self.path+f"/output/Cumulants_{self.id_}.dat", 'w') as ouf:
                header = '\t'.join(
                    ['nu', 'A2', 'M12', 'M22', 'A3', 'M13', 'M23', 'M33', 'A4', 'M14', 'M24', 'M34', 'M44']
                )
                ouf.write(header + '\n')

                for k in range(Nu):
                    values = '\t'.join(
                        [str(k), 
                         str(A_2[k]), str(Mu1_2[k]), str(Mu2_2[k]), 
                         str(A_3[k]), str(Mu1_3[k]), str(Mu2_3[k]), str(Mu3_3[k]), 
                         str(A_4[k]), str(Mu1_4[k]), str(Mu2_4[k]), str(Mu3_4[k]), str(Mu4_4[k])]
                    )
                    ouf.write(values + '\n')
            

        return probability_approx_2, probability_approx_3, probability_approx_4
    
class CumulantUtility(MomentUtility):
    
    
    def import_moments(self):
        
        m = self.n_modes 
        Nu = self.n_sector_max
        dnu = self.n_sector_step
        

        data_minors = np.genfromtxt(self.path + f'/output/Minors0-1_{self.id_}.dat')

        Z_v_0 = np.zeros((Nu),dtype=np.complex128)

        Z_v_1 = np.zeros((m, Nu),dtype=np.complex128)

        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1] + 1j*data_minors[j,2]

        for j in range(Nu):
            for n in range(0,2*m,2):
                Z_v_1[n//2,j] =  data_minors[j,int(3+n)] + 1j*data_minors[j,int(4+n)] 


        Z_v_0f =  np.zeros((Nu), dtype = np.complex128)
        Z_v_1f = np.zeros((m, Nu), dtype = np.complex128)

        Z_v_0f = np.fft.fft(Z_v_0)/Nu
        Z_v_1f = np.fft.fft(Z_v_1)/Nu

        data_moments = np.genfromtxt(self.path+f'/output/Moments_{self.id_}.dat')

        m0 = (Z_v_0f[:]/Z_v_0[0]).real #normalization
        m1 = data_moments[:, 0]
        m2 = data_moments[:, 1]
        m3 = data_moments[:, 2]
        m4 = data_moments[:, 3]
        
        return m0, m1, m2, m3, m4
        
    def guess_fun(self, x, *args):

        # if cutoff == 'None':
        #     prob_notcut = 1.0
        # else:
        #     prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

        for m in args:
            m = args

        if len(m) == 3:
            
            # why not m[0] *(1/(m[2] * np.sqrt(2 * np.pi)))?
            # further I will include a correct normalization 

            pdf_vals = m[0] *(
                         np.exp( - (x - m[1])**2 / (2 * m[2]**2)) 
                      ) 

        if len(m) == 4: 

            pdf_vals = m[0] *(
                         np.exp( - (x - m[1])**2 / (2 * m[2]**2)) 
                       * np.exp(+ m[3]*(x - m[1])**3/(3*m[2]**3))
                      ) 

        if len(m) == 5: 

            pdf_vals = m[0] *(
                         np.exp( - (x - m[1])**2 / (2 * m[2]**2)) 
                       * np.exp(+ m[3]*(x - m[1])**3/(3*m[2]**3))
                       * np.exp(+ m[4]*(x - m[1])**4/(8*m[2]**4))

                    ) 

                       #/prob_notcut)

        return pdf_vals
 

    def get_cumulants(self):
    
        m = self.n_modes
        Nu = self.n_sector_max
        cutoff = int(m*2)
        moments_data = np.array(self.import_moments())

        # The 2nd order approximation 
        A_2 = np.zeros(Nu)
        Mu1_2 = np.zeros(Nu)
        Mu2_2 = np.zeros(Nu) 

        for nu in range(Nu):
            
            params_init = np.array( moments_data[:3, nu])
            #print(params_init)
            A_2[nu], Mu1_2[nu],  Mu2_2[nu] = self.GMM(params_init, cutoff, nu)

        # The 3rd order approximation 
        A_3 = np.zeros(Nu)
        Mu1_3 = np.zeros(Nu)
        Mu2_3 = np.zeros(Nu)
        Mu3_3 = np.zeros(Nu)

        for nu in range(Nu):
            
            params_init =  moments_data[:4, nu]
            A_3[nu], Mu1_3[nu],  Mu2_3[nu], Mu3_3[nu] = self.GMM(params_init, cutoff, nu)


        # The 4th order approximation 
        A_4 = np.zeros(Nu)
        Mu1_4 = np.zeros(Nu)
        Mu2_4 = np.zeros(Nu)
        Mu3_4 = np.zeros(Nu)
        Mu4_4 = np.zeros(Nu)

        for nu in range(Nu):
            
            params_init =  moments_data[:, nu]
            A_4[nu], Mu1_4[nu],  Mu2_4[nu], Mu3_4[nu], Mu4_4[nu] = self.GMM(params_init, cutoff, nu) 
                    
        return A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3, A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4

    def model_moments(self, cutoff, *args):
        
        for m in args:
            m = args
            
        
        if len(m) == 3:

            s0 = 0
            s1 = 0
            s2 = 0


            for x in np.arange(cutoff):

                s0 +=  self.guess_fun(x,  m[0], m[1], m[2])
                s1 +=  self.guess_fun(x,  m[0], m[1], m[2]) * x
                s2 +=  self.guess_fun(x,  m[0], m[1], m[2]) * x**2

            m0_model = s0
            m1_model = s1/s0
            m2_model = s2/s0 - m1_model**2

            model_moms = m0_model, m1_model, m2_model

        elif len(m) == 4:

            s0 = 0
            s1 = 0
            s2 = 0
            s3 = 0


            for x in np.arange(1,cutoff):

                s0 +=  self.guess_fun(x, m[0], m[1], m[2], m[3])
                s1 +=  self.guess_fun(x, m[0], m[1], m[2], m[3]) * x
                s2 +=  self.guess_fun(x, m[0], m[1], m[2], m[3]) * x**2
                s3 +=  self.guess_fun(x, m[0], m[1], m[2], m[3]) * x**3

            m0_model = s0
            m1_model = s1/s0
            m2_model = s2/s0 - m1_model**2
            m3_model = s3/s0 - 3*m2_model*m1_model - m1_model**3

            model_moms = m0_model, m1_model, m2_model, m3_model

        elif len(m) == 5:

            s0 = 0
            s1 = 0
            s2 = 0
            s3 = 0
            s4 = 0


            for x in np.arange(cutoff):

                s0 +=  self.guess_fun(x, m[0], m[1], m[2], m[3], m[4])
                s1 +=  self.guess_fun(x, m[0], m[1], m[2], m[3], m[4]) * x
                s2 +=  self.guess_fun(x, m[0], m[1], m[2], m[3], m[4]) * x**2
                s3 +=  self.guess_fun(x, m[0], m[1], m[2], m[3], m[4]) * x**3
                s4 +=  self.guess_fun(x, m[0], m[1], m[2], m[3], m[4]) * x**4

            m0_model = s0
            m1_model = s1/s0
            m2_model = s2/s0 - m1_model**2
            m3_model = s3/s0 - 3*m2_model*m1_model - m1_model**3
            m4_model = s4/s0 - 4*m3_model*m1_model - 3*m2_model**2 - 6*m2_model*m1_model**2 - m1_model**4

            model_moms = m0_model, m1_model, m2_model, m3_model, m4_model
            
        else:
            print('The number of moments is unexpected; it must be equal to 3, 4, or 5.')


        return model_moms


    def err_vec(self, cutoff, nu, *args, simple):

        for m in args:
            m = args

        m_data = np.array(self.import_moments())[:,nu]  

        if len(m) == 3: 
            m0_data, m1_data, m2_data = m_data[0], m_data[1], m_data[2]
            moms_data = np.array([[m0_data], [m1_data], [m2_data]])

            m0_model, m1_model, m2_model = self.model_moments(cutoff, m[0], m[1], m[2])
            moms_model = np.array([[m0_model],[m1_model], [m2_model]])

        elif len(m) == 4: 
            m0_data, m1_data, m2_data, m3_data = m_data[0], m_data[1], m_data[2], m_data[3]
            moms_data = np.array([[m0_data], [m1_data], [m2_data],[m3_data]])

            m0_model, m1_model, m2_model, m3_model =self.model_moments(cutoff, m[0], m[1], m[2], m[3])
            moms_model = np.array([[m0_model],[m1_model], [m2_model], [m3_model]])

        elif len(m) == 5: 
            m0_data, m1_data, m2_data, m3_data, m4_data = m_data[0], m_data[1], m_data[2], m_data[3], m_data[4]
            moms_data = np.array([[m0_data], [m1_data], [m2_data],[m3_data], [m4_data]])

            m0_model, m1_model, m2_model, m3_model, m4_model = self.model_moments(cutoff, m[0], m[1], m[2], m[3], m[4])
            moms_model = np.array([[m0_model],[m1_model], [m2_model], [m3_model], [m4_model]])
            
        else:
            print('The number of moments is unexpected; it must be equal to 3, 4, or 5.')
            
        if simple:
            err_vec = moms_model - moms_data
        else:
            err_vec = (moms_model - moms_data) / moms_data

        return err_vec


    def criterion(self, params, *args):

        cutoff, nu, W = args
        simple=False

        if len(params) == 3:
            m0, m1, m2 = params
            err = self.err_vec(cutoff, nu, m0, m1, m2, simple=simple)

        if len(params) == 4:
            m0, m1, m2, m3 = params
            err = self.err_vec(cutoff, nu, m0, m1, m2, m3, simple=simple)

        if len(params) == 5:
            m0, m1, m2, m3, m4 = params
            err = self.err_vec(cutoff, nu, m0, m1, m2, m3, m4, simple=simple)

        crit_val = np.dot(np.dot(err.T, W), err) 

        return crit_val

    def GMM(self, params_init, cutoff, nu):

        simple = False

        W_hat = np.eye(len(params_init)) # ? because we don't have data -  only moments - we don't need to optimize this ?  
        gmm_args = (cutoff, nu, W_hat) #(pts, cutoff, W_hat)
        
        norm_bond = (0,1),
        mu1_bond = (0,cutoff),
        mu2_bond = (0, cutoff),
        # mu3_bond = (-cutoff, cutoff),
        # mu4_bond = (-cutoff, cutoff),
        mu3_bond = (-1, 1),
        mu4_bond = (-1, 1),
        
        
        bs = norm_bond + mu1_bond + mu2_bond + mu3_bond +  mu4_bond

        results = opt.minimize(self.criterion, params_init, args=(gmm_args),
                               method='L-BFGS-B',bounds=bs[:len(params_init)] )
        
        
        # results = opt.minimize(self.criterion, params_init, args=(gmm_args),
        #                        method='L-BFGS-B', bounds=bs[:len(params_init)] )

        return results.x 

                
    def prob_approx(self, export_probabilities=True, export_cumulants=True):
        
        m = self.n_modes
        Nu = self.n_sector_max
        
        A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3,  A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4 = self.get_cumulants()

        data_minors = np.genfromtxt(self.path + f'/output/Minors0-1_{self.id_}.dat')

        Z_v_0 = np.zeros((Nu),dtype=np.complex128)

        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1] + 1j*data_minors[j,2]
        
     
        M, _m, _n, _r = import_input(self.path, f"/GBS_matrix.dat")

        normalization = Z_v_0[0].real/Z(M) 
       
        probability_approx_2 = 0
        probability_approx_3 = 0
        probability_approx_4 = 0
        
        k_min_2 = Nu
        k_max_2 = 0
        k_min_3 = Nu
        k_max_3 = 0
        k_min_4 = Nu
        k_max_4 = 0

        for k in range(Nu):
            p2 = self.guess_fun(m, A_2[k], Mu1_2[k], Mu2_2[k])/normalization 
            p3 = self.guess_fun(m, A_3[k], Mu1_3[k], Mu2_3[k], Mu3_3[k])/normalization
            p4 = self.guess_fun(m, A_4[k], Mu1_4[k], Mu2_4[k], Mu3_4[k], Mu4_4[k])/normalization 
            
            if p2==p2 and p2<1 and p2>0:
                probability_approx_2 += p2 
                k_min_2 = min(k, k_min_2)
                k_max_2 = max(k, k_max_2)
    
            if p3==p3 and p3<1 and p3>0: 
                probability_approx_3 += p3
                k_min_3 = min(k, k_min_3)
                k_max_3 = max(k, k_max_3)
                
            if p4 == p4 and p4<1 and p4>0:
                probability_approx_4 += p4
                k_min_4 = min(k, k_min_4)
                k_max_4 = max(k, k_max_4)
           
        if export_probabilities == True:
            
            names = (
                    [
                        'm', 
                        'p2', 'k_min_2', 'k_max_2', 
                        'p3', 'k_min_3', 'k_max_3',
                        'p4', 'k_min_4', 'k_max_4',
                    ]
                )

            values = (
                    [
                        m,
                        probability_approx_2, 
                        k_min_2, k_max_2, 
                        probability_approx_3,
                        k_min_3, k_max_3, 
                        probability_approx_4,
                        k_min_4, k_max_4, 
                    ]
                )
            
            with open(self.path + f"/output/Result_{self.id_}.dat", 'w') as ouf:
                
                for i in range(len(values)):
                    ouf.writelines(names[i]+ '\t' + str(values[i]) + '\n')
            
            
        if export_cumulants==True:
            
            with open(self.path+f"/output/Cumulants_{self.id_}.dat", 'w') as ouf:
                header = '\t'.join(
                    ['nu', 'A2', 'M12', 'M22', 'A3', 'M13', 'M23', 'M33', 'A4', 'M14', 'M24', 'M34', 'M44']
                )
                ouf.write(header + '\n')

                for k in range(Nu):
                    values = '\t'.join(
                        [str(k), 
                         str(A_2[k]), str(Mu1_2[k]), str(Mu2_2[k]), 
                         str(A_3[k]), str(Mu1_3[k]), str(Mu2_3[k]), str(Mu3_3[k]), 
                         str(A_4[k]), str(Mu1_4[k]), str(Mu2_4[k]), str(Mu3_4[k]), str(Mu4_4[k])]
                    )
                    ouf.write(values + '\n')
            

        return probability_approx_2, probability_approx_3, probability_approx_4
    
    def import_cumulants(self):

   
        Nu = self.n_sector_max
        
        A_2 = np.zeros(Nu)
        Mu1_2 = np.zeros(Nu)
        Mu2_2 = np.zeros(Nu)
        A_3 = np.zeros(Nu)
        Mu1_3 = np.zeros(Nu)
        Mu2_3 = np.zeros(Nu)
        Mu3_3 = np.zeros(Nu)
        A_4  = np.zeros(Nu)
        Mu1_4 = np.zeros(Nu)
        Mu2_4 = np.zeros(Nu)
        Mu3_4 = np.zeros(Nu)
        Mu4_4 = np.zeros(Nu)
        
        
        data_cumulants = np.genfromtxt(self.path + f'/output/Cumulants_{self.id_}.dat', skip_header=1)
        
        A_2 = data_cumulants[:,1]
        Mu1_2 = data_cumulants[:,2]
        Mu2_2 = data_cumulants[:,3]
        A_3 = data_cumulants[:,4]
        Mu1_3 = data_cumulants[:,5]
        Mu2_3 = data_cumulants[:,6]
        Mu3_3 = data_cumulants[:,7]
        A_4  = data_cumulants[:,8]
        Mu1_4 = data_cumulants[:,9]
        Mu2_4 = data_cumulants[:,10]
        Mu3_4 = data_cumulants[:,11]
        Mu4_4 = data_cumulants[:,12]

        return A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3,  A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4

    def prob_approx_sectors(self, export_probabilities=True, export_cumulants=True):

        m = self.n_modes
        Nu = self.n_sector_max

        A_2, Mu1_2, Mu2_2,  A_3, Mu1_3, Mu2_3, Mu3_3,  A_4, Mu1_4, Mu2_4, Mu3_4, Mu4_4 = self.import_cumulants()

        data_minors = np.genfromtxt(self.path + f'/output/Minors0-1_{self.id_}.dat')

        Z_v_0 = np.zeros((Nu),dtype=np.complex128)

        for j in range(Nu):
             Z_v_0[j] =  data_minors[j,1] + 1j*data_minors[j,2]


        M, _m, _n, _r = import_input(self.path, f"/GBS_matrix.dat")

        normalization = Z_v_0[0].real/Z(M) 

        probability_approx_2 = np.zeros(Nu)
        probability_approx_3 = np.zeros(Nu)
        probability_approx_4 = np.zeros(Nu)

        k_min_2 = Nu
        k_max_2 = 0
        k_min_3 = Nu
        k_max_3 = 0
        k_min_4 = Nu
        k_max_4 = 0

        for k in range(Nu):
            p2 = self.guess_fun(m, A_2[k], Mu1_2[k], Mu2_2[k])/normalization 
            p3 = self.guess_fun(m, A_3[k], Mu1_3[k], Mu2_3[k], Mu3_3[k])/normalization
            p4 = self.guess_fun(m, A_4[k], Mu1_4[k], Mu2_4[k], Mu3_4[k], Mu4_4[k])/normalization 

            if p2==p2 and p2<1 and p2>0:
                probability_approx_2[k]= p2 
                k_min_2 = min(k, k_min_2)
                k_max_2 = max(k, k_max_2)

            if p3==p3 and p3<1 and p3>0: 
                probability_approx_3[k] = p3
                k_min_3 = min(k, k_min_3)
                k_max_3 = max(k, k_max_3)

            if p4 == p4 and p4<1 and p4>0:
                probability_approx_4[k] = p4
                k_min_4 = min(k, k_min_4)
                k_max_4 = max(k, k_max_4)



        return probability_approx_2, probability_approx_3, probability_approx_4
    

## Get approximate probabilities

def get_approx_probabilities( path=return_path() ):
    
    data_ids = np.loadtxt(path + '/input/samples_ids.dat', dtype=str,ndmin=1)
    
    if data_ids.ndim == 1:
        ids = [int(data_ids[0])]
        samples =  [data_ids[1]]
    else:
        ids = [int(i) for i in data_ids[:,0]]
        samples = data_ids[:,1]

    dict_probabilities = {}
    
    # Compile cpp file - see on the top
    #cmd = "cpp/Minors.cpp"
    #subprocess.call(["g++", cmd])
    
    for i in ids:
        sample = samples[i]
        
        moments = MomentUtility(id_ = i, path=path)
        moments.export_minors()
        moments.export_moments()
        
        cumulants = CumulantUtility(id_ = i, path=path)
        probability_approx_2, probability_approx_3, probability_approx_4 = cumulants.prob_approx()

        dict_probabilities[sample] = (
            [
                probability_approx_2, 
                probability_approx_3, 
                probability_approx_4
            ]
        )
        
        print(f"Computation for sample #{i} of {len(ids)} is completed.")
        
    return  dict_probabilities

def import_approx_probabilities( path=return_path() ):

#     data_ids = np.genfromtxt(path + '/input/samples_ids.dat', dtype=str)

#     ids = [int(i) for i in data_ids[:,0]]
#     samples = data_ids[:,1]
    
    data_ids = np.loadtxt(path + '/input/samples_ids.dat', dtype=str,ndmin=1)
    
    if data_ids.ndim == 1:
        ids = [int(data_ids[0])]
        samples =  [data_ids[1]]
    else:
        ids = [int(i) for i in data_ids[:,0]]
        samples = data_ids[:,1]
        
    dict_probabilities = {}
    
    for i in ids:
        sample = samples[i]
        
        data_result = np.genfromtxt(path + f'/output/Result_{i}.dat')
        probability_approx_2 =  data_result[1,1]
        probability_approx_3 = data_result[4,1]
        probability_approx_4 = data_result[7,1]
                
        dict_probabilities[sample] = (
            [
                probability_approx_2, 
                probability_approx_3, 
                probability_approx_4
            ]
        )
        
    return  dict_probabilities

def compute_probabilities(samples, path=return_path() ):
    
    M, m, n, r = import_input(path, "/GBS_matrix.dat")
    #M, m, n, r, n_cutoff, n_mc, batch_size
    submatriсes_export(M, samples, path)
    
    dict_probabilities = get_approx_probabilities(path = path)
        
    return dict_probabilities

## Generate DataFrame with results

def get_basis_df(M):
    
    m = len(M)
    
    # Obtain all possible samples for theshold detection
    all_permutations = threshold_basis_set(m)
    
    # Calculate probabilities for all possible samples 

    probabilities_exact = []

    for s in all_permutations:
        probabilities_exact.append(prob_exact(s, M))
        
    basis_dictionary = {
         convert_list_to_str(all_permutations[i]): [sum(all_permutations[i]), probabilities_exact[i]] 
        for i in range(len(all_permutations))
    }
    
    df_basis = (
    pd.DataFrame
    .from_dict(
        basis_dictionary, 
        orient='index',
        columns=["n_clicks","probability_exact"])
    )

    df_basis.index.name = "sample"
    
    # Sum all probabilities to obtain 1 
    print("Sum of Probabilities:", "{:.3e}".format(sum(probabilities_exact)))
    
    return df_basis


def count_samples(samples, samples_dictionary):
    # we can't use np.unique() because it returns SORTED list
    batch_size = len(samples)
    n_unique = len(samples_dictionary.keys())
    n_counts = [0]*n_unique

    unique_samples = list(samples_dictionary.keys())

    for i in range(batch_size):
        s = convert_list_to_str(samples[i])
        #(','.join(map(str, samples[i])).replace(',',''))
        if s in unique_samples:
            index =  unique_samples.index(s)
            n_counts[index]+=1
        else:
            raise 'Incomplete list of unique samples in the dictionary'
    return n_counts

def get_result_df(samples, M, dict_prob, exact_prob = True):
    
    if exact_prob == True:
        samples_dictionary = {
            convert_list_to_str(s): [sum(s),  prob_exact(s, M) ] for s in samples
        }

        df_1 = (
            pd.DataFrame
            .from_dict(
                samples_dictionary, 
                orient='index', 
                columns=["n_clicks","probability_exact"]
            )
        )

        #df_1.index.name = "sample"

        df_1["n_counts"] = count_samples(samples, samples_dictionary)

        df_2 = pd.DataFrame.from_dict(
            dict_prob, 
            orient='index', 
            columns=["probability_approx_2", "probability_approx_3", "probability_approx_4"]
        )

        #df_2.index.name = "sample"

        df = pd.merge(df_1, df_2, on=df_1.index).set_index("key_0")
        df.index.name = "sample"
        
        return df
    
    else:
        
        samples_dictionary = {
            convert_list_to_str(s): sum(s) for s in samples
        }
        
        df_1 = (
            pd.DataFrame
            .from_dict(
                samples_dictionary, 
                orient='index', 
                columns=["n_clicks"]
            )
        )
        
        df_1["n_counts"] = count_samples(samples, samples_dictionary)

        df_2 = pd.DataFrame.from_dict(
            dict_prob, 
            orient='index', 
            columns=["probability_approx_2", "probability_approx_3", "probability_approx_4"]
        )

        #df_2.index.name = "sample"

        df = pd.merge(df_1, df_2, on=df_1.index).set_index("key_0")
        df.index.name = "sample"
        
        return df
        
def get_dict_format(df):
    dict_format = {}
    for key in df.keys():
        if "probability" in key: 
            dict_format[key] = "{:.3e}"
            
    return dict_format

### Metrics
def relative_weighted_error(p, q):
    
    rwe = np.mean([abs(1 - p[i]/q[i]) for i in range(len(p))])
    
    return rwe

def cosine_similarity(p, q):
    
    cs = sum([p[i]*q[i] for i in range(len(p))])/(sum([p**2 for p in p])*sum([q**2 for q in q]))**0.5
    
    return cs

def mean_absolute_percentage_error(p, q):
    
    mape = np.mean([abs(1 - q[i]/p[i]) for i in range(len(p))])
    
    return mape

def fidelity(p, q):

    f = sum([(p[i]*q[i])**0.5 for i in range(len(p))])**2

    return f

def total_variation_distance(p, q):

    tvd = max([abs(p[i]-q[i]) for i in range(len(p))])
        
    return tvd

def cross_entropy(p, q):

    xe =  -sum([p[i]*np.log2(q[i]) for i in range(len(p))])

    return xe


def get_tests_df(df): 
    
    m = len(df.index[0]) 
    
    p_ex = df["probability_exact"].to_list()
    p_app_2 =  df["probability_approx_2"].to_list()
    p_app_3 =  df["probability_approx_3"].to_list()
    p_app_4 =  df["probability_approx_4"].to_list()
    
    p_unif = [1/2**m]*len(p_ex)
    
    tests_dictionary = (
        {

            "relative weighted error": 
            [relative_weighted_error(p_ex,p_ex), 
             relative_weighted_error(p_ex,p_app_2),
             relative_weighted_error(p_ex,p_app_3),
             relative_weighted_error(p_ex,p_app_4),
             relative_weighted_error(p_ex,p_unif)
             
            ],
            
            "mape": 
            [mean_absolute_percentage_error(p_ex,p_ex), 
             mean_absolute_percentage_error(p_ex,p_app_2),
             mean_absolute_percentage_error(p_ex,p_app_3),
             mean_absolute_percentage_error(p_ex,p_app_4),
             mean_absolute_percentage_error(p_ex,p_unif)
             
            ],
            
            "cosine similarity": 
            [cosine_similarity(p_ex,p_ex), 
             cosine_similarity(p_ex,p_app_2),
             cosine_similarity(p_ex,p_app_3),
             cosine_similarity(p_ex,p_app_4),
             cosine_similarity(p_ex,p_unif)
             
            ],
            
            "total variation distance" :  
            [total_variation_distance(p_ex,p_ex), 
             total_variation_distance(p_ex,p_app_2),
             total_variation_distance(p_ex,p_app_3),
             total_variation_distance(p_ex,p_app_4),
             total_variation_distance(p_ex,p_unif) 
            ],
            "fidelity": 
            [fidelity(p_ex,p_ex), 
             fidelity(p_ex,p_app_2),
             fidelity(p_ex,p_app_3), 
             fidelity(p_ex,p_app_4), 
             fidelity(p_ex,p_unif) 
            ] ,
            "cross entropy" :
            [cross_entropy(p_ex,p_ex),
             cross_entropy(p_ex,p_app_2),
             cross_entropy(p_ex,p_app_3),
             cross_entropy(p_ex,p_app_4),
             cross_entropy(p_ex,p_unif)
            ]

        }
    )


    df_tests = (pd.DataFrame
                .from_dict(
                    tests_dictionary, 
                    orient='index',
                    columns=
                    ["(p_exact, p_exact)", 
                     "(p_exact, p_appr_2)",
                     "(p_exact, p_appr_3)",
                     "(p_exact, p_appr_4)",
                     "(p_exact, p_uniform)"
                    ])
                )

    df_tests.index.name = "metric"

    return df_tests

## Samples tests 

def HOG_rate(experimental_samples, mockup_samples, M):
    
    """Returns a value of Heavy Output Generation rate 
    for target samples (experimental_samples) and from
    adversary samples (mockup_samples)."""
    
    r = 0 
    batch_size = len(experimental_samples)
    
    for i in range(batch_size):
        
        r += (
            prob_exact(experimental_samples[i], M)/
            (
                 prob_exact(experimental_samples[i], M) +
                 prob_exact(mockup_samples[i], M)
            ) / batch_size
        )
    
    return r

# exp_samples = np.copy(samples) 
# adv_samples = np.copy(uniform_samples)  

# res_forw = []
# res_back = []

# for i in range(len(exp_samples)):
#     res_forw.append(HOG_rate(exp_samples, adv_samples,i, M))
#     res_back.append(HOG_rate(adv_samples, exp_samples,i, M))
    