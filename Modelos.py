import numpy as np
import sympy as sp

sx= np.array([[0,1],[1,0]])
sy= np.array([[0,-1j],[1j,0]])
sz= np.array([[1,0],[0,-1]])
s0= np.array([[1,0],[0,1]])

def TP(A, B):
    return np.kron(A, B)

def QWZ(kx, ky, v, a, u, alpha, eps):
    """QWZ model Hamiltonian.

    Parameters
    ----------
    kx, ky : float
        Momentum components.
    a, u, v : float
        Model parameters.

    Returns
    -------
    ndarray
        2x2 Hamiltonian matrix.
    """
    d_x = (v / a) * np.sin(kx * a * alpha)
    d_y = (v / a) * np.sin(ky * a * alpha)
    d_z = u + np.cos(kx * a * alpha) + np.cos(ky * a * alpha)
    
    sigma_0 = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    H = d_x * sigma_x + d_y * sigma_y + d_z * sigma_z + eps*sigma_0
    return H

def ChernInsulator(kx, ky, v, a, m0, m1):
    """Chern insulator model.

    Parameters
    ----------
    kx, ky : float
        Momentum components.
    v, a, m0, m1 : float
        Model parameters.

    Returns
    -------
    ndarray
        2x2 Hamiltonian matrix.
    """
    d_x = (v / a) * np.sin(kx * a)
    d_y = (v / a) * np.sin(ky * a)
    d_z = m0 * v**2 - (4 * m1 / a**2) * (np.sin(kx * a / 2)**2 + np.sin(ky * a / 2)**2)
    
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    H = d_x * sigma_x + d_y * sigma_y + d_z * sigma_z
    return H


def ModifiedMultiLayerQWZ(kx, ky, v, a, u, C, alpha, eps):
    """Multilayer QWZ model.

    Parameters
    ----------
    kx, ky : float
        Momentum components.
    a, u, v, C : float
        Model parameters.

    Returns
    -------
    ndarray
        4x4 Hamiltonian matrix.
    """
    sigma_0 = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    H_Multilayer = np.block([
        [QWZ(kx, ky, v, a, u, alpha,eps), C * sigma_0],
        [np.conj(C) * sigma_0, QWZ(kx, ky, v, a, u, alpha,-eps)]])
    return H_Multilayer

def Graphene(kx,ky, Delta, t):
    
    # Define the vector d(k)
    d = (
        t*np.cos(kx)*(1+2*np.cos(np.sqrt(3)/2*ky)*np.cos(3*kx/2))+np.sin(kx)*(2*np.cos(np.sqrt(3)/2*ky)*np.sin(3*kx/2)), 
        Delta*np.sin(kx)*(-1-2*np.cos(np.sqrt(3)/2*ky)*np.cos(3*kx/2))+np.cos(kx)*(2*np.cos(np.sqrt(3)/2*ky)*np.sin(3*kx/2)), 
        Delta
    )
    
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    H = d[0] * sigma_x + d[1] * sigma_y + d[2] * sigma_z
    
    return H


def HgTeQW(kx, ky, GE, GH, C0=0.0, C2=0.0, M0=1.0, M2=1.0, A=1.0):
    """Hamiltonian for a HgTe quantum well.

    Parameters
    ----------
    kx, ky : float
        Momentum components.
    GE, GH : float
        Splitting parameters.
    C0, C2, M0, M2, A : float, optional
        Model constants with default values.

    Returns
    -------
    ndarray
        Full 4×4 Hamiltonian matrix.
    """
    # Matrices de Pauli
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Function to compute ε(k) y M(k)
    def epsilon_k(kx, ky):
        k2 = kx**2 + ky**2
        return C0 + C2 * k2

    def M_k(kx, ky):
        k2 = kx**2 + ky**2
        return M0 + M2 * k2

    # Hamiltoniano h_+(k)
    def h_plus(kx, ky):
        A_term = A * (kx * sigma_x - ky * sigma_y)
        H = epsilon_k(kx, ky) * np.eye(2) + M_k(kx, ky) * sigma_z + A_term
        return H

    # Hamiltonian h_-(k) as the Hermitian conjugate of h_+(-k)
    def h_minus(kx, ky):
        return h_plus(-kx, -ky).conj().T

    # Splitting term H_s
    def splitting_term(GE, GH):
        return np.array([[GE,  0,  0,  0],
                         [0,  GH, 0,  0],
                         [0,  0, -GE, 0],
                         [0,  0,  0, -GH]], dtype=complex)

    # Calcular h_+ y h_-
    H_plus = h_plus(kx, ky)
    H_minus = h_minus(kx, ky)

    # Construir el Hamiltoniano completo de 4x4
    H0 = np.block([[H_plus, np.zeros((2, 2), dtype=complex)],
                   [np.zeros((2, 2), dtype=complex), H_minus]])

    # Add the splitting term
    H_total = H0 + splitting_term(GE, GH)

    return H_total




def LatticeHgTeQW(kx, ky, GE, GH, a=1.0, C0=0.0, C2=1.0, M0=1.0, M2=1.0, A=1.0):
    """
    Calcula el Hamiltoniano para un Quantum Well de HgTe.

    Parameters:
    ----------
    kx, ky : float
        Componentes del momento.
    GE, GH : float
        Splitting parameters.
    C0, C2, M0, M2, A : float, opcional
        Constantes del modelo con valores por defecto.

    Retorna:
    -------
    H_total : ndarray de 4x4
        Matriz Hamiltoniana total.
    """
    # Matrices de Pauli
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Function to compute ε(k) y M(k)
    def epsilon_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return C0 + 2*C2/(a**2) * ck2

    def M_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return M0 + 2*M2/(a**2) * ck2

    # Hamiltoniano h_+(k)
    def h_plus(kx, ky, a):
        A_term = A/a * (np.sin(kx*a) * sigma_x - np.sin(ky*a) * sigma_y)
        H = epsilon_k(kx, ky,a) * np.eye(2) + M_k(kx, ky,a) * sigma_z + A_term
        return H

    # Hamiltonian h_-(k) as the Hermitian conjugate of h_+(-k)
    def h_minus(kx, ky, a):
        return h_plus(-kx, -ky, a).conj().T

    # Splitting term H_s
    def splitting_term(GE, GH):
        return np.array([[GE,  0,  0,  0],
                         [0,  GH, 0,  0],
                         [0,  0, -GE, 0],
                         [0,  0,  0, -GH]], dtype=complex)

    # Calcular h_+ y h_-
    H_plus = h_plus(kx, ky, a)
    H_minus = h_minus(kx, ky, a)

    # Construir el Hamiltoniano completo de 4x4
    H0 = np.block([[H_plus, np.zeros((2, 2), dtype=complex)],
                   [np.zeros((2, 2), dtype=complex), H_minus]])

    # Add the splitting term
    H_total = H0 + splitting_term(GE, GH)

    return H_total



def LatticeHgTeQWMixed(kx, ky, GE, GH, T=0., a=1.0, C0=0.0, C2=1.0, M0=1.0, M2=1.0, A=1.0):
    """
    Calcula el Hamiltoniano para un Quantum Well de HgTe.

    Parameters:
    ----------
    kx, ky : float
        Componentes del momento.
    GE, GH : float
        Splitting parameters.
    C0, C2, M0, M2, A : float, opcional
        Constantes del modelo con valores por defecto.

    Retorna:
    -------
    H_total : ndarray de 4x4
        Matriz Hamiltoniana total.
    """
    # Matrices de Pauli
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Function to compute ε(k) y M(k)
    def epsilon_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return C0 + 2*C2/(a**2) * ck2

    def M_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return M0 + 2*M2/(a**2) * ck2

    # Hamiltoniano h_+(k)
    def h_plus(kx, ky, a):
        A_term = A/a * (np.sin(kx*a) * sigma_x - np.sin(ky*a) * sigma_y)
        H = epsilon_k(kx, ky,a) * np.eye(2) + M_k(kx, ky,a) * sigma_z + A_term
        return H

    # Hamiltonian h_-(k) as the Hermitian conjugate of h_+(-k)
    def h_minus(kx, ky, a):
        return h_plus(-kx, -ky, a).conj().T

    # Splitting term H_s
    def splitting_term(GE, GH):
        return np.array([[GE,  0,  0,  0],
                         [0,  GH, 0,  0],
                         [0,  0, -GE, 0],
                         [0,  0,  0, -GH]], dtype=complex)
    # Calcular h_+ y h_-
    H_plus = h_plus(kx, ky, a)
    H_minus = h_minus(kx, ky, a)

    # Construir el Hamiltoniano completo de 4x4
    H0 = np.block([[H_plus,T*sigma_y],
                   [-T*sigma_y, H_minus]])

    # Add the splitting term
    H_total = H0 + splitting_term(GE, GH)

    return H_total


def SlabHam(kx,ky,kz, GE, GH, T=0.0, a=1.0, C0=0.0, C1=0.0, C2=0.0, M0=1.0, M1=1.0 ,M2=1.0, A=1.0):

    # Matrices de Pauli
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Function to compute ε(k) y M(k)
    def epsilon_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return C0 + 2*C2/(a**2) * ck2 + 2*C1/(a**2) * (1-np.cos(kz*a))

    def M_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return M0 + 2*M2/(a**2) * ck2 + 2*M1/(a**2) * (1-np.cos(kz*a))

    # Hamiltoniano h_+(k)
    def h_plus(kx, ky, a):
        A_term = A/a * (np.sin(kx*a) * sigma_x - np.sin(ky*a) * sigma_y)
        H = epsilon_k(kx, ky,a) * np.eye(2) + M_k(kx, ky,a) * sigma_z + A_term
        return H

    # Hamiltonian h_-(k) as the Hermitian conjugate of h_+(-k)
    def h_minus(kx, ky, a):
        return h_plus(-kx, -ky, a).conj().T

    # Splitting term H_s
    def splitting_term(GE, GH):
        return np.array([[GE,  0,  0,  0],
                         [0,  GH, 0,  0],
                         [0,  0, -GE, 0],
                         [0,  0,  0, -GH]], dtype=complex)
    # Calcular h_+ y h_-
    H_plus = h_plus(kx, ky, a)
    H_minus = h_minus(kx, ky, a)

    # Construir el Hamiltoniano completo de 4x4
    H0 = np.block([[H_plus,1j*T*np.sin(kz*a)*sigma_y/a],
                   [-1j*T*sigma_y*np.sin(kz*a)/a, H_minus]])

    # Add the splitting term
    H_total = H0 + splitting_term(GE, GH)

    return H_total


def BHZ(kx, ky, G=0 , a=1.0, C0=0.0, C2=1.0, M0=1.0, M2=1.0, A=1.0,T=0):
    """
    Calcula el Hamiltoniano para el modelo BHZ con un G zeeman

    Parameters:
    ----------
    kx, ky : float
        Componentes del momento.
    G : float
        Splitting parameters.
    C0, C2, M0, M2, A : float, opcional
        Constantes del modelo con valores por defecto.

    Retorna:
    -------
    H_total : ndarray de 4x4
        Matriz Hamiltoniana total.
    """
    # Matrices de Pauli
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Function to compute ε(k) y M(k)
    def epsilon_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return C0 + 2*C2/(a**2) * ck2

    def M_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return M0 + 2*M2/(a**2) * ck2

    # Hamiltoniano h_+(k)
    def h_plus(kx, ky, a):
        A_term = A/a * (np.sin(kx*a) * sigma_x - np.sin(ky*a) * sigma_y)
        H = epsilon_k(kx, ky,a) * np.eye(2) + M_k(kx, ky,a) * sigma_z + A_term
        return H

    # Hamiltonian h_-(k) as the Hermitian conjugate of h_+(-k)
    def h_minus(kx, ky, a):
        return h_plus(kx, -ky, a)

    # Splitting term H_s
    def splitting_term(G):
        return np.array([[-G,  0,  0,  0],
                         [0,  G, 0,  0],
                         [0,  0, G, 0],
                         [0,  0,  0, -G]], dtype=complex)

    # Calcular h_+ y h_-
    H_plus = h_plus(kx, ky, a)
    H_minus = h_minus(kx, ky, a)

    # Construir el Hamiltoniano completo de 4x4
    H0 = np.block([[H_plus, 1j*T*sigma_y],
                   [-1j*T*sigma_y, H_minus]])

    # Add the splitting term
    H_total = H0 + splitting_term(G)

    return H_total

def HamAcoploConSuperonductor(kx, ky, G=0 , a=1.0, C0=0.0, C2=1.0, M0=1.0, M2=1.0, A=1.0,T=0, Delta=1.0):
    Acop = 1j*Delta * TP(sy,sz) 
    Acop2 = -1j*Delta * TP(sy,sz) 
    BH1 = BHZ(kx, ky, G , a, C0, C2, M0, M2, A, T) 
    BH2=  -BHZ(-kx, -ky, G , a, C0, C2, M0 ,M2, A, T).T

    #H = TP(sz,BH1) +  TP(sy,Acop)
    H =np.block([[BH1,Acop],[Acop2,BH2]])

    return H
    
def BHZ_h(kx, ky, G=0 , a=1.0, C0=0.0, C2=1.0, M0=1.0, M2=1.0, A=1.0,T=0):
 
    return -BHZ(-kx, -ky, G , a, C0, C2, M0 ,M2, A, T).T





import numpy as np

def BHZ_SC_Slab(kx, ky, kz,
           G=0.5, T=0.0, Delta=0.0, a=1.0,
           C0=0.0, C1=0.0, C2=1.0, M0=-1.0, M1=0.0, M2=1.0, A=1.0):
    """BHZ model with superconductivity in an 8×8 basis.

    The model is based on BHZ including superconducting terms and external
    fields.

    Parameters
    ----------
    kx, ky : float
        Momentum components in the Brillouin zone.
    G : float
        Zeeman-like coupling appearing in ``-G(tau_z sigma_z s_z)``.
    T : float
        Coupling parameter ``-T(tau_z sigma_y s_y)``.
    Delta : float
        Superconducting parameter ``-Delta(tau_y sigma_y s_z)``.
    a : float, optional
        Lattice constant.
    C0, C2, M0, M2, A : float, optional
        Standard BHZ model parameters.

    Returns
    -------
    ndarray (8×8)
        BHZ+SC Hamiltonian matrix.
    """

    # Matrices de Pauli 2x2
    pauli_x = np.array([[0, 1],[1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0],[0, -1]], dtype=complex)
    pauli_0 = np.eye(2, dtype=complex)

    # Funciones epsilon_k y M_k
    def epsilon_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)
        return C0 + 2*C2/(a**2)*ck2 + 2*C1/a**2*(1-np.cos(kz*a))

    def M_k(kx, ky, a):
        ck2 = 2 - np.cos(kx*a) - np.cos(ky*a)-np.cos(kz*a)
        return M0 + 2*M2/(a**2)*ck2 + 2*M1/a**2*(1-np.cos(kz*a))

    # Atajos para seno
    sin_kx = np.sin(kx*a)
    sin_ky = np.sin(ky*a)
    sin_kz = np.sin(kz*a)

    # Helper to compute triple Kronecker products more easily
    def kron3(A, B, C):
        return np.kron(np.kron(A, B), C)


    # tau_z (epsilon_k sigma_0 + M_k sigma_z) s_0
    term1 = np.kron(pauli_z, (epsilon_k(kx,ky,a)*pauli_0 + M_k(kx,ky,a)*pauli_z))
    term1 = np.kron(term1, pauli_0)

    # (A/a)[ sin(k_x a)(tau_0 sigma_0 s_x) + sin(k_y a)(tau_0 sigma_z s_y) ]
    term2 = (A/a)*(
          sin_kx * kron3(pauli_0, pauli_0, pauli_x)
        + sin_ky * kron3(pauli_0, pauli_z, pauli_y)
    )

    # - G (tau_z sigma_z s_z)
    term3 = -G * kron3(pauli_z, pauli_z, pauli_z)

    # - T (tau_z sigma_y s_y)
    term4 = -T * kron3(pauli_z, pauli_y, pauli_y) * sin_kz

    # - Delta (tau_y sigma_y s_z)
    term5 = -Delta * kron3(pauli_y, pauli_y, pauli_z)

    # Hamiltoniano total
    H = term1 + term2 + term3 + term4 + term5

    return H




