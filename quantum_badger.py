import numpy as np
import random
import os 
from math import * 

# Methods

def input_state(r, m, n):

    """
    Returns a list of squeezing parameters and a list of input random phases.
    Fills n modes out of m with equal squeezing parameter r.
    
    Args:
        r (float): The squeezing parameter.
        m (int): The total number of modes.
        
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

def set_input(r, phi, export = True):
    
    """
    Returns a diagonal input matrix A.
    
    Args:
        r (list): A list of squeezing parameters with length m.
        phi (list): A list of input random phases with length m.
        export (bool, optional): Whether to export a file with information about the input state. Defaults to True.
        
    Returns:
        np.ndarray: A diagonal input matrix A of shape (m, m) with complex128 dtype, where the diagonal elements are 
                    calculated as -exp(1j * phi[i]) * tanh(r[i]) / 2 for i in the range [0, m).
    """
    
    m = len(r)
    A = np.zeros((m, m), dtype=np.complex128)

    for i in range(m):
        A[i, i] = -np.exp(1j * phi[i]) * np.tanh(r[i]) / 2
        
    if export == True:
        with open(path + r"/initial_state.dat", "w") as ouf:
        
            ouf.write("N\tr\tphi\tA_real\tA_imag\n")

            for k in range(A.shape[0]):
                ouf.write(
                    f"{k}\t{r[k]}\t{phi[k]}\t{A[k, k].real}\t{A[k, k].imag}\n"
                )
            

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

def get_random_interferometer(m, n_bs, export = True):
    
    """
    Generates a random interferometer matrix U with beam splitters and phase shifters.
    
    Args:
        m (int): The size of the square matrix U.
        n_bs (int): The number of beam splitters (BS) and phase shifters (PS) to use in the interferometer.
        export (bool, optional): If True, the generated matrix U and parameters are exported to files. Default is True.
        
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
        
    if export == True:
        
        with open(path + "/parameters_of_interferometer.dat", "w") as ouf:

            ouf.write(
                f"# N_modes = {m}\tN_bs = {n_bs}\tN_ps = {n_ps}\n[n1, n2]\tphi\tpsi\teta\n"
            )

            for z in range(n_bs):
                ouf.write(
                    f"{ind[z][0]}\t{ind[z][1]}\t{phi[z]}\t{psi[z]}\t{eta[z]}\n"
                )

        export_complex_matrix(path + r"/matrix_U.dat", U)

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
        n_mc (int, optional): Number of Monte Carlo samples for the emulation. Default is 10^5.
        n_cutoff (float, optional): Photon number cutoff. Default is the result of the function average_photon_number(r_s).
        batch_size (int, optional): Batch size for writing data to file. Default is 10^3.
        path (str): Path where the output files will be saved.

    Returns:
        None. Prints a message indicating the data has been exported to the specified path.
    """
    
    m = len(M)
    n_ps = int(n_bs*2)
    n = np.count_nonzero(np.array(r_s))
    n_cutoff=average_photon_number(r_s)

    with open(path + r"/initial_state.dat", "w") as ouf:
        
        ouf.write("N\tr\tphi\tA_real\tA_imag\n")

        for k in range(A.shape[0]):
            ouf.write(
                f"{k}\t{r_s[k]}\t{phi_s[k]}\t{A[k, k].real}\t{A[k, k].imag}\n"
            )
            
    with open(path + "/parameters_of_interferometer.dat", "w") as ouf:

        ouf.write(
            f"# N_modes = {m}\tN_bs = {n_bs}\tN_ps = {n_ps}\n[n1, n2]\tphi\tpsi\teta\n"
        )
        
        for z in range(n_bs):
            ouf.write(
                f"{ind[z][0]}\t{ind[z][1]}\t{phi[z]}\t{psi[z]}\t{eta[z]}\n"
            )
                
    with open(path + "/GBS_matrix.dat", "w") as ouf:
        
        ouf.write(
            f"{m}\t{n}\t{r_s[0]}\t{n_cutoff}\t{n_mc}\t{batch_size}\n"
        )
        
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
            - n_cutoff (int): Cutoff parameter for the GBS scheme.
            - n_mc (int): Number of Monte Carlo samples for the GBS scheme.
            - batch_size (int): Batch size for the GBS scheme.
            
    Raises:
        FileNotFoundError: If the specified input file is not found at the given path.
        ValueError: If the input file is not in the expected format or contains invalid data.
    """
   
    data_M = np.genfromtxt(path + file_name, skip_header=1)

    m = len(data_M)
    
    data_ = np.genfromtxt(path + file_name, skip_footer = m )
    
    n, r, n_cutoff, n_mc, batch_size = int(data_[1]), data_[2], int(data_[3]), int(data_[4]), int(data_[5])

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

    print("Data were imported from " + path + file_name)

    return M, m, n, r, n_cutoff, n_mc, batch_size 

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
    """

    data = np.genfromtxt(path + '/' + file_name, skip_header=2)
    ind_1 = data[:, 0].astype(int)
    ind_2 = data[:, 1].astype(int)
    ind = np.stack((ind_1, ind_2), axis=-1)
    phi = data[:, 2]
    psi = data[:, 3]
    eta = data[:, 4]

    return ind, phi, psi, eta
    
    
def set_device_parameters(r, A, U, export = False):
    
    n_mc = 0
    n_cutoff = 0 
    m = len(U) 
    n = sum(np.diagonal(A)==0)
    batch_size = 0
    
    M = M_matrix(U, A) 
    
    if export == True:
        
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
    

def choose_default_device(m, r, export = True):
    
    """
    Initializes the Gaussian Boson Sampling (GBS) device by setting up the input state, interferometer matrix,
    and other parameters for the simulation.
    
    Args:
        m (int): The number of modes in the GBS device.
        r (float): The squeezing parameter for the input state.
        export (bool, optional): Whether to export all files related to the simulation. Defaults to False.
        
    Returns:
        tuple: A tuple containing two elements:
            - numpy array: The Gaussian matrix M used in the GBS simulation.
            - numpy array: The interferometer matrix U used in the GBS simulation.
    """
    
    
    # Input initialization
    r_s, phi_s = input_state(r, m, n)
    A = input_matrix(r_s, phi_s, m, n)
    n_bs=m**2
    
    # Interferometer initialization
    U, ind, phi, psi, eta = interferometer(n_bs, m)
        
    # The GBS device initializtion
    M = M_matrix(U, A)
    
    # Export all files related to the simulation
    if export == True:
        export_input(path, r_s, phi_s, A, ind, phi, psi, eta, n_bs, U, M, n)
    
 
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


def C_nm(x, y):
    
    """
    Calculates the binomial coefficient C(x, y) using the factorial formula.
    
    Args:
        x (int): The total number of items.
        y (int): The number of items to choose.
        
    Returns:
        int: The binomial coefficient C(x, y) as an integer, obtained by dividing the factorial
             of x by the product of the factorials of (x - y) and y.
    """

    res = np.math.factorial(int(x)) / (
        np.math.factorial(int(x - y)) * np.math.factorial(int(y))
    )

    return int(res)


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
    
    samples_data = np.genfromtxt(path + file_name, dtype=str)
    samples = []

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    dir_alphabet = { alphabet[i]: 10+i for i in range(len(alphabet))}

    for s in samples_data:
        sample = []
        for i in s:
            if i not in alphabet:
                sample.append(int(i))
            else:
                sample.append(dir_alphabet[str(i)])

        samples.append(sample)
            
    return samples
