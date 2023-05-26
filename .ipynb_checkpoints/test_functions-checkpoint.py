# Tests 
from quantum_badger import *

def check_matrix(U):
    
    """
    Сhecks the matrix for zero elements.
    """
    
    condition = np.all(U)
    
    if condition == True:
        
        return 'None of the elements of the given matrix is zero'
    else:
        return 'There are zero elements of the given matrix'
    
def check_unitarity(U): 
    
    """
    Сhecks the matrix for unitarity.
    """
    
    condition = np.allclose(np.eye(len(U), dtype=np.complex128), U.T.conj() @ U)
    
    if condition == True:
        
        return 'The matrix is a unitary one'
    else:
        return 'The matrix is non-unitary one'
    
def check_hermitianity(M):
    
    """
    Сhecks the matrix for hermitianity.
    """
    
    condition = np.allclose(M, M.T.conj())
    
    if condition == True:
        
        return 'The matrix is a hermitian one'
    else:
        return 'The matrix is non-hermitian one'
    
    
def check_set_parameters(U,M):
    
    assert len(M)==len(U)
    
    return print(f"Interferometer matrix U: {check_matrix(U)}; {check_unitarity(U)}", 
                 f"\n Gaussian matrix M: {check_matrix(M)} ; M.H * M {check_hermitianity(M.T.conj()@ M)}") 
    
    
def check_uniform_sampler_tr(m):
    
    for s in uniform_sampling_tr(int(1e6),2,m):
        if sum(s) != 2:
            return print("Error: number is clicked detectors wrong.")
        else:
            return print("Test passed.")