import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy.linalg as LA
import sympy as sp
from IPython.display import display, Math
from sympy.utilities.lambdify import lambdify
from sympy import latex, Matrix
from sympy import sin, cos, simplify 
from sympy import I as sympy_I
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.optimize import linear_sum_assignment


def NumeroChern(Ham, params, N=51):
    """
    Calcula el número de Chern para un Hamiltoniano general usando el método FHS.

    Parámetros:
    - Ham: función que representa el Hamiltoniano del sistema. Debe tomar kx, ky y los parámetros necesarios.
    - params: diccionario con los parámetros del Hamiltoniano.
    - N: Número de puntos de discretización en cada dirección k.

    Retorna:
    - Número de Chern (entero).
    """
    # Establecer distancia de Red
    a = params.get('a', 1.)
    
    # Discretizar el espacio de momentos
    kx_vals = np.linspace(-np.pi / a, np.pi / a, N, endpoint=False)
    ky_vals = np.linspace(-np.pi / a, np.pi/ a, N, endpoint=False)
    dkx = 2 * np.pi / (N * a)
    dky = 2 * np.pi / (N * a)

    # Evaluar el Hamiltoniano en un punto para obtener su dimensión
    H_sample = Ham(kx_vals[0], ky_vals[0], **params)
    dim = H_sample.shape[0]

    # Inicializar una cuadrícula para almacenar los vectores propios de las bandas ocupadas
    num_bandas_ocupadas = dim // 2
    eigenvectors = np.zeros((N, N, num_bandas_ocupadas, dim), dtype=complex)

    # Calcular los vectores propios en cada punto k
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H_k = Ham(kx, ky, **params)
            eigvals, eigvecs = LA.eigh(H_k)
            # Asumir que las bandas inferiores están ocupadas
            eigenvectors[i, j, :, :] = eigvecs[:, :num_bandas_ocupadas].T 

    # Inicializar el número de Chern
    Chern_total = 0.0

    # Calcular la curvatura de Berry usando el método de Link Variables
    for b in range(num_bandas_ocupadas):
        Chern_banda = 0.0
        for i in range(N):
            for j in range(N):
                # Condiciones de frontera periódicas
                ip = (i + 1) % N
                jp = (j + 1) % N

                # Obtener los vectores propios de la banda b en los puntos k actuales y vecinos
                v_ij = eigenvectors[i, j, b, :]
                v_ipj = eigenvectors[ip, j, b, :]
                v_ipjp = eigenvectors[ip, jp, b, :]
                v_ijp = eigenvectors[i, jp, b, :]

                # Superposiciones (variables de enlace)
                U1 = np.vdot(v_ij, v_ipj)
                U2 = np.vdot(v_ipj, v_ipjp)
                U3 = np.vdot(v_ipjp, v_ijp)
                U4 = np.vdot(v_ijp, v_ij)

                # Calcular la curvatura de Berry para la plaqueta
                F = np.angle(U1 * U2 * U3 * U4)
                Chern_banda += F

        # Normalizar el número de Chern para la banda b
        Chern_banda = Chern_banda / (2 * np.pi)
        Chern_total += Chern_banda

    # Redondear al entero más cercano
    numero_chern = ((Chern_total))

    return numero_chern

def calcular_gap(Ham, params, N_k=21):
    """
    Calcula el mínimo gap de energía para un par (m0, m1) sobre la malla de momentos.

    Parámetros:
    - m0, m1: Parámetros del modelo.
    - v, a: Parámetros del sistema (por defecto 1.0).
    - N_k: Número de puntos en cada dirección de k (kx, ky).

    Retorna:
    - El mínimo gap de energía encontrado en la malla de momentos.
    """
    # Establecer distancia de Red
    a = params.get('a', 1.)
    
    # Discretización del espacio de momentos en el primer BZ
    kx_vals = np.linspace(-np.pi / a, np.pi / a, N_k, endpoint=False)
    ky_vals = np.linspace(-np.pi / a, np.pi / a, N_k, endpoint=False)
    
    min_gap = np.inf  # Inicializamos el mínimo gap con infinito
            
    for kx in kx_vals:
        for ky in ky_vals:
            H = Ham(kx, ky, **params)
            eigvals = LA.eigvalsh(H)  # Calculamos los autovalores ordenados
            gap = np.abs(eigvals[1] - eigvals[0])  # Gap de energía entre las dos bandas
            if gap < min_gap:
                min_gap = gap
    # Barra de progreso
    return min_gap

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy.linalg as LA




# Solo sirve para sistemas H(k)=d(k)* σ
def curvatura_de_berry_analitica(d):
    kx, ky, kz = sp.symbols('kx ky kz')
    
    # Definir el vector d(k) simbólicamente
    dx = d[0]
    dy = d[1]
    dz = d[2]

    # Vector d(k)
    d = sp.Matrix([dx, dy, dz])

    # Calcular las derivadas parciales respecto a kx y ky
    d_kx = sp.Matrix([sp.diff(dx, kx), sp.diff(dy, kx), sp.diff(dz, kx)])
    d_ky = sp.Matrix([sp.diff(dx, ky), sp.diff(dy, ky), sp.diff(dz, ky)])

    # Calcular el producto cruz entre las derivadas parciales
    cross_product = d_kx.cross(d_ky)

    # Calcular el producto punto entre d y el producto cruzado
    dot_product = d.dot(cross_product)

    # Calcular la norma de d al cubo
    d_norm = d.norm()**3

    # Definir el integrando
    berryC = dot_product / d_norm

    # Simplificar el integrando
    berryC_s = sp.simplify(2*berryC)

    # Generar el código LaTeX
    latex_output = sp.latex(berryC_s)
    
    display(Math(r"\text{Curvatura de Berry} \;  \cdot \Omega(k): \quad " + latex_output))
    
    # Devolver el resultado simplificado y el código en LaTeX
    return berryC_s, latex_output




def plot_curvatura_de_berry(d, k_vars, params_symbols, params_values, 
                            k_range=(-np.pi, np.pi), num_points=100):
    """
    Calcula y visualiza la curvatura y conexión de Berry para un sistema dado.

    Parámetros:
    ----------
    d : tuple o list
        Vector d(k) en términos de las variables k y parámetros simbólicos.
        Por ejemplo: (d_x, d_y, d_z)

    k_vars : list
        Lista de símbolos de variables de momento, típicamente [kx, ky].

    params_symbols : list
        Lista de símbolos de parámetros del sistema, por ejemplo [v, a, alpha, u].

    params_values : dict
        Diccionario que mapea los nombres de los parámetros a sus valores numéricos.
        Por ejemplo: {'v':1.0, 'a':1.0, 'alpha':1.0, 'u':1.0}

    k_range : tuple, opcional
        Rango para cada variable de momento, por defecto (-π, π).

    num_points : int, opcional
        Número de puntos en cada dirección de la malla de k-space, por defecto 100.

    Retorna:
    -------
    None
        Muestra los gráficos de la curvatura y conexión de Berry.
    """
    
    # Asegurarse de que se trabaja en 2D
    if len(k_vars) != 2:
        raise ValueError("Esta función solo soporta sistemas 2D con dos variables de momento (kx, ky).")
    
    # Descomponer las variables de momento y los parámetros
    kx, ky = k_vars
    param_symbols = params_symbols
    param_names = [str(symbol) for symbol in param_symbols]
    
    # Verificar que todos los parámetros tengan valores proporcionados
    for name in param_names:
        if name not in params_values:
            raise ValueError(f"Falta el valor para el parámetro '{name}' en params_values.")
    
    # Paso 1: Calcular la Curvatura de Berry Simbólicamente
    def calcular_curvatura_de_berry(d, kx, ky):
        """
        Calcula la curvatura de Berry simbólicamente.

        Parámetros:
        ----------
        d : tuple o list
            Vector d(k) en términos de kx y ky.

        kx, ky : SymPy Symbols
            Símbolos de las variables de momento.

        Retorna:
        -------
        berryC_s : SymPy Expr
            Expresión simbólica simplificada de la curvatura de Berry.
        """
        # Vector d(k)
        d_vec = Matrix(d)
        
        # Derivadas parciales
        d_kx = d_vec.diff(kx)
        d_ky = d_vec.diff(ky)
        
        # Producto cruz entre las derivadas parciales
        cross_product = d_kx.cross(d_ky)
        
        # Producto punto entre d y el producto cruzado
        dot_product = d_vec.dot(cross_product)
        
        # Norma de d al cubo
        d_norm = d_vec.norm()**3
        
        # Curvatura de Berry
        berryC = dot_product / d_norm
        
        # Simplificar la expresión
        berryC_s = simplify(2*berryC)
        
        return berryC_s
    
    # Paso 2: Calcular la Conexión de Berry Simbólicamente
    def calcular_conexion_de_berry(d, kx, ky):
        """
        Calcula la conexión de Berry simbólicamente para un sistema de dos bandas.

        Parámetros:
        ----------
        d : tuple o list
            Vector d(k) en términos de kx y ky.
        
        kx, ky : SymPy Symbols
            Símbolos de las variables de momento.
        
        Retorna:
        -------
        A_x_s, A_y_s : SymPy Expr
            Expresiones simbólicas simplificadas de las componentes de la conexión de Berry.
        """
        # Definir el Hamiltoniano H(k) = d(k) · sigma
        sigma_x, sigma_y, sigma_z = sp.symbols('sigma_x sigma_y sigma_z')
        H = d[0]*sigma_x + d[1]*sigma_y + d[2]*sigma_z

        # Calcular los estados propios de H
        # Para un Hamiltoniano de dos bandas, los estados propios se pueden escribir en términos de d(k)
        # Estado positivo
        norm = sp.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        u_plus = Matrix([
            [d[0] + sympy_I*d[1]],
            [d[2] + norm]
        ]) / sp.sqrt(2*norm*(norm + d[2]))
        
        # Estado negativo (opcional, aquí nos enfocamos en u_plus)
        # u_minus = Matrix([
        #     [ -d[0] + sympy_I*d[1]],
        #     [d[2] + norm]
        # ]) / sp.sqrt(2*norm*(norm + d[2]))

        # Supongamos que estamos trabajando con el estado |u_plus>
        u_k = u_plus

        # Calcular las derivadas parciales
        du_k_dkx = u_k.diff(kx)
        du_k_dky = u_k.diff(ky)

        # Conexión de Berry
        A_x = sp.I * (u_k.H * du_k_dkx)[0]
        A_y = sp.I * (u_k.H * du_k_dky)[0]

        # Simplificar las expresiones
        #A_x_s = simplify(A_x)
        #A_y_s = simplify(A_y)

        return A_x, A_y
    
    # Paso 3: Calcular Simbólicamente la Curvatura de Berry
    berryC_s = calcular_curvatura_de_berry(d, kx, ky)
    
    # Generar y mostrar el código LaTeX para la curvatura de Berry
    display(Math(r"\text{Curvatura de Berry} \; \Omega(k): \quad " + latex(berryC_s)))
    
    # Paso 4: Calcular Simbólicamente la Conexión de Berry
    A_x_s, A_y_s = calcular_conexion_de_berry(d, kx, ky)
    
    # Generar y mostrar el código LaTeX para la conexión de Berry
    display(Math(r"\text{Conexión de Berry} \; \mathbf{A}(k): \quad " + latex(Matrix([A_x_s, A_y_s]))))
    
    # Paso 5: Convertir las Expresiones Simbólicas en Funciones Numéricas
    # Preparar las variables para lambdify: [kx, ky, param1, param2, ...]
    vars_lambdify = [kx, ky] + param_symbols
    berryC_func = lambdify(vars_lambdify, berryC_s, modules=['numpy'])
    A_x_func = lambdify(vars_lambdify, A_x_s, modules=['numpy'])
    A_y_func = lambdify(vars_lambdify, A_y_s, modules=['numpy'])
    
    # Paso 6: Crear una Malla de Puntos en el Espacio de Momentos (k-space)
    k_min, k_max = k_range
    kx_vals = np.linspace(k_min, k_max, num_points)
    ky_vals = np.linspace(k_min, k_max, num_points)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    # Paso 7: Evaluar la Curvatura y Conexión de Berry en Cada Punto de la Malla
    # Preparar los parámetros en el orden de param_symbols
    param_order = [params_values[name] for name in param_names]
    
    # Paso 7.1: Ignorar las advertencias de división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        BerryC = berryC_func(KX, KY, *param_order)
        A_x = A_x_func(KX, KY, *param_order)
        A_y = A_y_func(KX, KY, *param_order)
    
    # Paso 8: Manejar posibles valores NaN o infinitos
    BerryC = np.nan_to_num(BerryC, nan=0.0, posinf=0.0, neginf=0.0)
    A_x = np.nan_to_num(A_x, nan=0.0, posinf=0.0, neginf=0.0)
    A_y = np.nan_to_num(A_y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Paso 8: Asegurar que A_x y A_y sean Reales
    A_x = np.real(A_x)
    A_y = np.real(A_y)
    
    # Paso 9: Normalizar las Componentes de la Conexión de Berry para el Quiver
    # Calcular la magnitud de la conexión
    magnitude_conn = np.sqrt(A_x**2 + A_y**2)
    # Evitar división por cero
    magnitude_conn[magnitude_conn == 0] = 1
    U = A_x / magnitude_conn
    V = A_y / magnitude_conn
    
    # Paso 10: Reducir la Densidad de los Vectores para Mejorar la Visualización
    stride = max(num_points // 40, 1)  # Ajusta según el número de puntos
    KX_quiver = KX[::stride, ::stride]
    KY_quiver = KY[::stride, ::stride]
    U_quiver = U[::stride, ::stride]
    V_quiver = V[::stride, ::stride]
    magnitude_quiver = magnitude_conn[::stride, ::stride]
    
    # Paso 11: Visualizar los Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Gráfico 1: Mapa de Calor de la Curvatura de Berry
    ax1 = axes[0]
    contour = ax1.contourf(KX, KY, BerryC, levels=100, cmap='PiYG')
    fig.colorbar(contour, ax=ax1, label='Curvatura de Berry')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_title('Curvatura de Berry en $k$-space')
    
    # Gráfico 2: Campo Vectorial de la Conexión de Berry
    ax2 = axes[1]
    quiver = ax2.quiver(KX_quiver, KY_quiver, U_quiver, V_quiver, 
                        magnitude_quiver, cmap='inferno_r', pivot='middle', scale=50)
    fig.colorbar(quiver, ax=ax2, label='Magnitud de la Conexión de Berry')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_title('Conexión de Berry en $k$-space')
    
    plt.tight_layout()
    plt.savefig('ConexionYCurvatura.png')
    plt.show()
    




def compute_eigensystem(H_matrix, previous_evecs=None, threshold=None):
    
    evals, evecs = np.linalg.eigh(H_matrix)
    order = np.argsort(evals)
    evals_sorted = evals[order]
    evecs_sorted = evecs[:, order]
    
    if previous_evecs is not None:
        if threshold is None:
            N = H_matrix.shape[0]
            threshold = (2 * N) ** -0.25
        Q = np.abs(previous_evecs.T.conj() @ evecs_sorted)
        orig, perm = linear_sum_assignment(-Q)
        ######
        orden = np.argsort(orig)
        perm = perm[orden]
        ######
        disconnects = Q[orig, perm] < threshold
        evals_sorted = evals_sorted[perm]
        evecs_sorted = evecs_sorted[:, perm]
        return evals_sorted, evecs_sorted, perm, disconnects
    return evals_sorted, evecs_sorted


def calcular_mapa_gap_chern_Yuriko_act(Hamiltoniano, params, param_var1, rango_var1, param_var2, rango_var2, N=100, Nc=11):
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import linalg as LA
    from tqdm.notebook import tqdm

    # Desempaquetar los rangos de los parámetros
    inicio1, fin1, num1 = rango_var1
    inicio2, fin2, num2 = rango_var2

    # Crear los arrays de valores para cada parámetro
    valores1 = np.linspace(inicio1, fin1, num1)
    valores2 = np.linspace(inicio2, fin2, num2)

    # Crear una cuadrícula de valores
    grid1, grid2 = np.meshgrid(valores1, valores2)

    # Inicializar las matrices de resultados
    Gap_map = np.zeros_like(grid1, dtype=float)
    Chern_map = np.zeros_like(grid1, dtype=float)

    # Parámetros del espacio de momentos
    a = params.get('a', 1.0)
    kx_vals = np.linspace(-np.pi / a, np.pi / a, N, endpoint=False)
    ky_vals = np.linspace(-np.pi / a, np.pi / a, N, endpoint=False)
    dkx = 2 * np.pi / (N * a)
    dky = 2 * np.pi / (N * a)

    epsilon = 1e-10
    delta_k = 1e-6

    # Precalcular índices para la submalla si N != Nc
    if N != Nc:
        indices_kx = np.linspace(0, N - 1, Nc, dtype=int)
        indices_ky = np.linspace(0, N - 1, Nc, dtype=int)
    else:
        indices_kx = np.arange(N)
        indices_ky = np.arange(N)

    # Función auxiliar para calcular U_j con desplazamiento si es necesario
    def compute_U(v1, kx1, ky1, v2, kx2, ky2, b, num_bandas_ocupadas, params, shift_kx=False, shift_ky=False):
        U = np.vdot(v1, v2)
        if np.abs(U) < epsilon:
            # Desplazar kx o ky ligeramente y recalcular v2
            if shift_kx:
                kx2_shifted = kx2 + delta_k
                ky2_shifted = ky2
            elif shift_ky:
                kx2_shifted = kx2
                ky2_shifted = ky2 + delta_k
            else:
                kx2_shifted = kx2 + delta_k
                ky2_shifted = ky2

            H_k_shifted = Hamiltoniano(kx2_shifted, ky2_shifted, **params)
            eigvals_shifted, eigvecs_shifted = compute_eigensystem(H_k_shifted)
            v2_shifted = eigvecs_shifted[:, b]
            U = np.vdot(v1, v2_shifted)

        if np.abs(U) < epsilon:
            U_normalized = 1.0
        else:
            U_normalized = U / np.abs(U)

        return U_normalized

    # Definimos un valor mínimo para E_gap para evitar -log(0)
    min_gap_threshold = 1e-6

    total_iterations = grid1.size
    progress_bar = tqdm(total=total_iterations, desc="Procesando", unit="iteración")

    for idx in np.ndindex(grid1.shape):
        i, j = idx
        # Actualizar los parámetros variables
        params[param_var1] = grid1[idx]
        params[param_var2] = grid2[idx]

        # Inicializar variables para el cálculo del gap
        # Ahora el gap se definirá como la distancia mínima al cero energético.
        gap_local = np.inf

        # Primero calculamos todos los autovectores en la malla completa de k
        # También determinamos el número de bandas ocupadas una sola vez
        previous_evecs = None
        first_eigvals = None
        eigenvectors = None

        for iy, ky in enumerate(ky_vals):
            for ix, kx in enumerate(kx_vals):
                H_k = Hamiltoniano(kx, ky, **params)
                if previous_evecs is not None:
                    eigvals, eigvecs, _, _ = compute_eigensystem(H_k, previous_evecs)
                else:
                    eigvals, eigvecs = compute_eigensystem(H_k)
                
                if first_eigvals is None:
                    # Determinar número de bandas ocupadas (mitad inferiores)
                    dim = H_k.shape[0]
                    num_bandas_ocupadas = dim // 2
                    eigenvectors = np.zeros((N, N, dim, num_bandas_ocupadas), dtype=complex)
                    first_eigvals = eigvals

                # Guardamos los autovectores de las bandas ocupadas
                eigenvectors[ix, iy, :, :] = eigvecs[:, :num_bandas_ocupadas]

                # Calcular el gap local como la mínima distancia a cero
                min_abs_eig = np.min(np.abs(eigvals))
                if min_abs_eig < gap_local:
                    gap_local = min_abs_eig

                previous_evecs = eigvecs

        # Cálculo del número de Chern
        Chern_total = 0.0
        for b in range(num_bandas_ocupadas):
            Chern_banda = 0.0
            for iy_rel in range(len(indices_ky)):
                for ix_rel in range(len(indices_kx)):
                    idx_x = indices_kx[ix_rel]
                    idx_y = indices_ky[iy_rel]
                    idx_xp = indices_kx[(ix_rel + 1) % len(indices_kx)]
                    idx_yp = indices_ky[(iy_rel + 1) % len(indices_ky)]

                    v_ij = eigenvectors[idx_x, idx_y, :, b]
                    v_ipj = eigenvectors[idx_xp, idx_y, :, b]
                    v_ipjp = eigenvectors[idx_xp, idx_yp, :, b]
                    v_ijp = eigenvectors[idx_x, idx_yp, :, b]

                    kx_ij = kx_vals[idx_x]
                    ky_ij = ky_vals[idx_y]
                    kx_ipj = kx_vals[idx_xp]
                    ky_ipj = ky_vals[idx_y]
                    kx_ipjp = kx_vals[idx_xp]
                    ky_ipjp = ky_vals[idx_yp]
                    kx_ijp = kx_vals[idx_x]
                    ky_ijp = ky_vals[idx_yp]

                    U1 = compute_U(v_ij, kx_ij, ky_ij, v_ipj, kx_ipj, ky_ipj, b, num_bandas_ocupadas, params, shift_kx=True)
                    U2 = compute_U(v_ipj, kx_ipj, ky_ipj, v_ipjp, kx_ipjp, ky_ipjp, b, num_bandas_ocupadas, params, shift_ky=True)
                    U3 = compute_U(v_ipjp, kx_ipjp, ky_ipjp, v_ijp, kx_ijp, ky_ijp, b, num_bandas_ocupadas, params, shift_kx=True)
                    U4 = compute_U(v_ijp, kx_ijp, ky_ijp, v_ij, kx_ij, ky_ij, b, num_bandas_ocupadas, params, shift_ky=True)

                    F = np.angle(U1 * U2 * U3 * U4)
                    Chern_banda += F

            Chern_banda = Chern_banda / (2 * np.pi)
            Chern_total += Chern_banda

        ChernNumber = np.round(Chern_total)

        # Ajustar gap para plot: Evitar -log(0)
        gap = max(gap_local, min_gap_threshold)
        negative_log_gap = -np.log10(gap)

        Gap_map[idx] = negative_log_gap
        Chern_map[idx] = ChernNumber

        progress_bar.update(1)

    progress_bar.close()
    print("Cálculo completado.")

    # Generar los plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    extent = [inicio1, fin1, inicio2, fin2]

    im1 = axs[0].imshow(Gap_map, origin='lower', extent=extent, aspect='auto', cmap='plasma_r')
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label(r'$-\log({\Delta})$')
    axs[0].set_xlabel(param_var1)
    axs[0].set_ylabel(param_var2)
    axs[0].set_title('Mapa de Gap (basado en proximidad a cero)')

    num_colors = int(np.round(np.max(Chern_map) - np.min(Chern_map) + 1))
    cmap = plt.get_cmap('inferno_r', num_colors)
    im2 = axs[1].imshow(Chern_map, origin='lower', extent=extent, aspect='auto', cmap=cmap)
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label('Número de Chern')
    axs[1].set_xlabel(param_var1)
    axs[1].set_ylabel(param_var2)
    axs[1].set_title('Mapa del Número de Chern')

    plt.tight_layout()
    nombre_figura = f"{Hamiltoniano.__name__}_GapAndChernMap_Optimizado.png"
    plt.savefig(nombre_figura, dpi=300)
    plt.show()

    return grid1, grid2, Gap_map, Chern_map


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def calcular_chern_banda(Hamiltoniano, params ,eigenvectors, b, kx_vals, ky_vals):
    """
    Calcula el número de Chern para una banda específica a partir de los autovectores en el espacio k,
    utilizando la función compute_U.
    """
    Nkx, Nky, dim = eigenvectors.shape
    epsilon = 1e-10  # Umbral para evitar divisiones por cero
    delta_k = 1e-6   # Desplazamiento pequeño en k

    # Definir la función compute_U
    def compute_U(v1, kx1, ky1, v2, kx2, ky2, b, params, shift_kx=False, shift_ky=False):
        U = np.vdot(v1, v2)
        if np.abs(U) < epsilon:
            # Desplazar kx o ky ligeramente y recalcular v2
            if shift_kx:
                kx2_shifted = kx2 + delta_k
                ky2_shifted = ky2
            elif shift_ky:
                kx2_shifted = kx2
                ky2_shifted = ky2 + delta_k
            else:
                # Si no se especifica, desplazamos kx por defecto
                kx2_shifted = kx2 + delta_k
                ky2_shifted = ky2

            # Recalcular autovectores en el punto desplazado
            H_k_shifted = Hamiltoniano(kx2_shifted, ky2_shifted, **params)
            eigvals_shifted, eigvecs_shifted = LA.eigh(H_k_shifted)
            v2_shifted = eigvecs_shifted[:, b]

            U = np.vdot(v1, v2_shifted)

        # Normalizar U
        if np.abs(U) < epsilon:
            U_normalized = 1.0  # Asignamos 1 para evitar división por cero
        else:
            U_normalized = U / np.abs(U)

        return U_normalized

    # Inicializar el número de Chern
    ChernNumber = 0.0

    # Iterar sobre la malla de puntos k
    for iy in range(Nky):
        for ix in range(Nkx):
            # Índices de los vecinos con condiciones de contorno periódicas
            ix_plus = (ix + 1) % Nkx
            iy_plus = (iy + 1) % Nky

            # Autovectores en los cuatro vértices de la plaqueta
            v_ij = eigenvectors[ix, iy, :]
            v_ipj = eigenvectors[ix_plus, iy, :]
            v_ipjp = eigenvectors[ix_plus, iy_plus, :]
            v_ijp = eigenvectors[ix, iy_plus, :]

            kx_ij = kx_vals[ix]
            ky_ij = ky_vals[iy]
            kx_ipj = kx_vals[ix_plus]
            ky_ipj = ky_vals[iy]
            kx_ipjp = kx_vals[ix_plus]
            ky_ipjp = ky_vals[iy_plus]
            kx_ijp = kx_vals[ix]
            ky_ijp = ky_vals[iy_plus]

            # Cálculo de las superposiciones (variables de enlace)
            U1 = compute_U(v_ij, kx_ij, ky_ij, v_ipj, kx_ipj, ky_ipj,b,params, shift_kx=True)
            U2 = compute_U(v_ipj, kx_ipj, ky_ipj, v_ipjp, kx_ipjp, ky_ipjp, b,params,shift_ky=True)
            U3 = compute_U(v_ipjp, kx_ipjp, ky_ipjp, v_ijp, kx_ijp, ky_ijp, b,params,shift_kx=True)
            U4 = compute_U(v_ijp, kx_ijp, ky_ijp, v_ij, kx_ij, ky_ij, b, params,shift_ky=True)

            # Curvatura de Berry para la plaqueta
            F_ij = np.angle(U1 * U2 * U3 * U4)
            ChernNumber += F_ij

    # Normalizar por 2π
    ChernNumber = np.round(ChernNumber / (2 * np.pi))

    return ChernNumber

def Calculate_Everything(Ham, k_values, params, all_bands=True, bands=0):
# Rango de valores de k
    Nk = len(k_values)
    dim=Ham(0,0,**params).shape[0]

    if not all_bands:
        b = bands
    else:
        b = dim


    # Inicializar arrays para energías y autovectores
    eigenenergies = np.zeros((Nk, Nk, b))
    eigenvectors = np.zeros((Nk, Nk, dim, b), dtype=complex)

    previous_evecs_k = None

    

    # Cálculo de los autovectores en la malla kx-ky
    for iy, ky in enumerate(k_values):
        for ix, kx in enumerate(k_values):
            H_k = Ham(kx, ky, **params)
            if previous_evecs_k is not None:
                eigvals, eigvecs,_,_ = compute_eigensystem(H_k, previous_evecs_k)
            else:
                eigvals, eigvecs = compute_eigensystem(H_k)
            eigenvectors[ix, iy, :, :] = eigvecs[:,:b]
            eigenenergies[ix, iy, :] = eigvals[:b]
            previous_evecs_k = eigvecs


    return eigenenergies, eigenvectors

