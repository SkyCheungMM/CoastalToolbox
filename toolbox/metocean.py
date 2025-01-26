"""
MetOcean module
"""

import cmocean
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from ocean_wave_tracing import Wave_tracing
import rasterio
from rasterio.plot import show

import sys
sys.path.append("src")
from tb.toolbox.utils import *

def crest_distribution(X:np.ndarray, Hs:float, model="rayleigh", Tm=None, 
                       km=None, d=None, dir_spread=True):
    """
    Computes the probability of occurrence for a crest height in a given sea 
    state based on either the Rayleigh (linear) or Forristall (2nd order) model.
    
    Parameters
    ----------
    X : float or numpy array
        Crest heights to evaluate probability in meters
    Hs : float
        Significant wave height in meters
    model : str, default "rayleigh"
        Model used to compute probabilities. Available options are "rayleigh"
        and "forristall"
    Tm : float, default None
        Mean wave period in seconds
    km : float, default None
        Wave number corresponding to the mean wave period in rad/m
    d : float, default None
        Local water depth in meters
    dir_spread : bool, default True
        Whether the sea is directionally spreaded or not
        
    Returns
    -------
    eta : float or numpy array
        Values of which the probability has been evaluated at
    PDF : float or numpy array
        Probability density, evaulated at X
    CDF : float or numpy array
        Cumulative probability density, evaluated at X
    
    References
    ----------
    Forristall, G.Z. (2000), Wave crest distributions: Observations and second 
    order theory, J. Phys. Oceanogr., 30, 1931-1943.
    """

    # Convert input data type
    eta = np.array(X)

    # Check inputs
    assert len(eta.shape) <= 1, "Incorrect crest height input. Input must " + \
        "be a single number or a one-dimensional array.\n"
    if model == "forristall":
        assert (Tm is not None) & (km is not None) & (d is not None),\
            "Input missing for Forristall crest height model.\n"

    # Convert data type
    if len(eta.shape) == 0: eta = np.array([eta]) 

    # Constants
    g = 9.80665
    sigma = Hs/4

    # Rayleigh (linear) crest statistics
    if model == "rayleigh":
        PDF = eta/sigma**2*np.exp(-eta**2/(2*sigma**2))
        CDF = 1 - np.exp(-eta**2/(2*sigma**2))

    # Forristall (2nd order) crest statistics
    if model == "forristall":

        # Compute wave parameters
        S1 = 2*np.pi/g*Hs/Tm**2
        Ur = Hs/(km**2*d**3)

        if dir_spread: # Directionally Spread seas
            alpha = 0.3536 + 0.2568*S1 + 0.08*Ur
            beta = 2 - 1.7912*S1 - 0.5302*Ur + 0.2824*Ur**2

        else: # Uni-directional seas
            alpha = 0.3536 + 0.2892*S1 + 0.106*Ur
            beta = 2 - 2.1597*S1 + 0.0968*Ur**2

        PDF = beta*eta**(beta-1)/(alpha*Hs)**beta*np.exp(-(eta/(alpha*Hs))**beta)
        CDF = 1 - np.exp(-(eta/(alpha*Hs))**beta)

    return eta, PDF, CDF

def kfinder(om:float, d:float, tol=1e-6, max_iter=1000):
    """
    Computes the wave number using the linear dispersion equation.
    
    Parameters
    ----------
    om : float
        Angular frequency of the wave in rad/s
    d : float
        Local water depth in metres
    tol : float, default 1e-6
        Tolerance between iterations. The algorithm will terminate if successive
        values of k differ by less than this amount
    max_iter : int, default 1000
        Number of maximum iterations before the algorithm termiates. Very 
        unlikely to hit this limit and waves will be too nonlinear to apply this
        equation in shallow water anyways
        
    Returns
    -------
    k : float
        Wave number in rad/m
    """
    
    warnings.formatwarning = warning_output
    
    # Initialisation
    g = 9.80665
    k0 = -np.inf
    k1 = om**2/g

    # Main Loop
    count = 0
    while np.abs(k0 - k1) > tol:
        if count >= max_iter:
            warning_str = "\033[93m %s\033[00m"%("Maximum number of iterations"+
                                    " reached. Result may be inaccurate.", )
            warnings.warn(warning_str, RuntimeWarning)
            break

        k0 = k1
        k1 = om**2/(g*np.tanh(k0*d))
        count+=1
    
    return k1

def kfinder5(H:float, om:float, d:float, tol=1e-6, max_iter=1000):
    """
    Computes the wave number using the nonlinear dispersion equation.
    
    Parameters
    ----------
    H : float
        Wave height in metres
    om : float
        Angular frequency of the wave in rad/s
    d : float
        Local water depth in metres
    tol : float, default 1e-6
        Tolerance between iterations. The algorithm will terminate if successive
        values of k differ by less than this amount
    max_iter : int, default 1000
        Number of maximum iterations before the algorithm termiates. Very 
        unlikely to hit this limit and waves will be too nonlinear to apply this
        equation in shallow water anyways
        
    Returns
    -------
    k1 : float
        Wave number in rad/m

    References
    ----------
    Fenton, J. D. (1985) A fifth-order Stokes theory for steady waves, 
    J. Waterway Port Coastal and Ocean Engng 111, 216-234.
    """

    warnings.formatwarning = warning_output

    # Initialisation
    g  = 9.80665
    k0 = -np.inf
    k1 = kfinder(om, d, tol, max_iter)
    S  = 1/np.cosh(2*k1*d)
    C0 = np.sqrt(np.tanh(k1*d))
    C2 = np.sqrt(np.tanh(k1*d))*(2 + 7*S**2)/(4*(1 - S)**2)
    C4 = np.sqrt(np.tanh(k1*d))*(4 + 32*S - 116*S**2 - 400*S**3 - 71*S**4 + 
                        146*S**5)/(32*(1-S)**5)
    eps = H*k1/2

    # Iterate until value stabilises
    count = 0
    while np.abs(k1 - k0) > tol:
        if count >= max_iter:
            warning_str = "\033[93m %s\033[00m"%("Maximum number of iterations"+
                                    " reached. Result may be inaccurate.", )
            warnings.warn(warning_str, RuntimeWarning)
            break

        k0 = k1
        k1 = om**2/(g*(C0 + C2*eps**2 + C4*eps**4)**2)

        S  = 1/np.cosh(2*k1*d)
        C0 = np.sqrt(np.tanh(k1*d))
        C2 = np.sqrt(np.tanh(k1*d))*(2 + 7*S**2)/(4*(1 - S)**2)
        C4 = np.sqrt(np.tanh(k1*d))*(4 + 32*S - 116*S**2 - 400*S**3 - 71*S**4 + 
                            146*S**5)/(32*(1-S)**5)
        eps = H*k1/2
        
        count += 1
    
    return k1

def miche(T:float, d:float):
    """
    Applies the Miche Criteria to obtain the steepness limited wave height.
    
    Parameters
    ----------
    T : float
        Period of the wave in seconds
    d : float
        Local water depth in metres
        
    Returns
    -------
    Hmax : float
        Steepness limited wave height in metres
    """
    
    k = kfinder(2*np.pi/T, d)
    Hmax = 2*np.pi/k/7

    return Hmax

def wind_speed_adjustment(U:np.ndarray, z=10.0, ti=3600.0, tf=900.0):
    """
    Adjust wind speeds based on elevation and averaging period provided

    Parameters
    ----------
    U : numpy.ndarray
        Wind speed. Input should be 1D, i.e. of size (N,)
    z : float, default 10
        Elevation of the anemometer in metres
    ti : float, default 3600
        Input wind speed averaging period in seconds
    tf : float, default 900
        Output wind speed averaging period in seconds

    Returns
    -------
    Uadj : numpy.ndarray
        Height and averaging period adjusted wind speed, in same units as 
        the input values
    """

    U = np.array(U).ravel()

    # Power law
    kz = (10/z)**(1/7)

    # Averaging period 
    kt = lambda x: 1.277 + 0.296*np.tanh(0.9*np.log10(45/x)) if x <= 3600 else \
        lambda x: 1.5334 - 0.15*np.log10(x)

    Uadj = U*kz*kt(tf)/kt(ti)

    return Uadj

class Regular(object):
    """Compute wave kinematics using regular wave theories."""

    def __init__(self, H:float, T:float, d:float):
        """
        Class initiation.
        
        Parameters
        ----------
        H : float
            Wave height in metres
        T : float
            Wave period in seconds
        d : float
            Water depth in metres
        """

        # Wave characteristics
        self.H = H  # Wave height
        self.T = T  # Wave period
        self.d = d  # Water depth

        return None
    
    def fit(self, theory:str, N=15, **kwargs):
        """
        Fit wave theory to wave characteristics. Stokes 5th order and stream
        function wave theories are based on Fenton's work in 1985 and 1988 
        respectively. Additional keyword-value pairs get passed to the kfinder() 
        function
        
        Parameters
        ----------
        theory : str
            Wave theory to use. Available options include "Linear", "Stokes_5th" 
            and "Stream"
        N : int, default 15
            Number of fourier components used in stream function computation.
            Steep and near breaking waves may require more components
        
        References
        ----------
        Fenton, J. D. (1985) A fifth-order Stokes theory for steady waves, 
        J. Waterway Port Coastal and Ocean Engng 111, 216-234. <br>
        Fenton, J. D. (1988) The numerical solution of steady water wave problems, 
        Computers and Geosciences 14, 357-368.
        """

        # Check theory input
        available_theories = ["Linear", "Stokes_5th", "Stream"]
        assert theory in available_theories, "Incorrect method input. " + \
            "Available options are '%s', '%s' & '%s'.\n\n"%(*available_theories, )
        
        # Wave parameters
        self.theory = theory
        self.N = N
        self.Ce = 0
        self.Cs = 0
        
        # Pick the correct theory and fit to model
        if theory == "Linear":
            self._fit_linear(**kwargs)
        elif theory == "Stokes_5th":
            self._fit_stokes5()
        elif theory == "Stream":
            self._fit_stream()
        
        return None

    def _fit_linear(self, **kwargs):
        """
        Fits the linear wave theory to observations.
        """

        print("Fitting linear wave theory to input conditions...")

        # Wave parameters
        H = self.H
        T = self.T
        d = self.d
        g = 9.80665

        # Obtain linear k
        k = kfinder(2*np.pi/T, d, **kwargs)

        # Compute other properties
        c = np.sqrt(g/k*np.tanh(k*d))
        cg = 0.5*c*(1 + 2*k*d/np.sinh(2*k*d))

        # Store properties
        self.k = k
        self.c = c
        self.cg = cg

        return None

    def _fit_stokes5(self, **kwargs):
        """
        Fits the 5th order Stokes wave theory to observations.
        """

        print('Fitting Stokes 5th order wave theory to input conditions...')

        # Wave parameters
        H = self.H
        T = self.T
        d = self.d
        g = 9.80665

        # Obtain Stokes k
        k = kfinder5(H, 2*np.pi/T, d, **kwargs)

        # Compute coefficients (Table 1)
        S = 1/np.cosh(2*k*d)
        coef = {
            "A11": 1/np.sinh(k*d),
            "A22": 3*S**2/(2*(1-S)**2),
            "A31": (-4 - 20*S + 10*S**2 - 13*S**3)/(8*np.sinh(k*d)*(1-S)**3),
            "A33": (-2*S**2 + 11*S**3)/(8*np.sinh(k*d)*(1-S)**3),
            "A42": (12*S - 14*S**2 - 264*S**3 - 45*S**4 - 13*S**5)/(24*(1-S)**5),
            "A44": (10*S**3 - 174*S**4 + 291*S**5 + 278*S**6)/(48*(3+2*S)*(1-S)**5),
            "A51": (-1184 + 32*S + 13232*S**2 + 21712*S**3 + 20940*S**4 + 
                        12554*S**5 - 500*S**6 - 3341*S**7 - 670*S**8)/
                        (64*np.sinh(k*d)*(3+2*S)*(4+S)*(1-S)**6),
            "A53": (4*S + 105*S**2 + 198*S**3 - 1376*S**4 - 1302*S**5 - 
                        117*S**6 + 58*S**7)/(32*np.sinh(k*d)*(3+2*S)*(1-S)**6),
            "A55": (-6*S**3 + 272*S**4 - 1552*S**5 + 852*S**6 + 2029*S**7 +
                        430*S**8)/(64*np.sinh(k*d)*(3+2*S)*(4+S)*(1-S)**6),
            "B22": 1/np.tanh(k*d)*(1 + 2*S)/(2*(1-S)),
            "B31": -3*(1 + 3*S + 3*S**2 + 2*S**3)/(8*(1-S)**3),
            "B42": 1/np.tanh(k*d)*(6 - 26*S - 182*S**2 - 204*S**3 - 25*S**4 +
                        26*S**5)/(6*(3+2*S)*(1-S)**4),
            "B44": 1/np.tanh(k*d)*(24 + 92*S + 122*S**2 + 66*S**3 + 67*S**4 +
                        34*S**5)/(24*(3+2*S)*(1-S)**4),
            "B53": 9*(132 + 17*S - 2216*S**2 - 5897*S**3 - 6292*S**4 - 
                        2687*S**5 + 194*S**6 + 467*S**7 + 82*S**8)/
                        (128*(3+2*S)*(4+S)*(1-S)**6),
            "B55": 5*(300 + 1579*S + 3176*S**2 + 2949*S**3 + 1188*S**4 + 
                        675*S**5 + 1326*S**6 + 827*S**7 + 130*S**8)/
                        (384*(3+2*S)*(4+S)*(1-S)**6),
            "C0" : np.sqrt(np.tanh(k*d)),
            "C2" : np.sqrt(np.tanh(k*d))*(2 + 7*S**2)/(4*(1-S)**2),
            "C4" : np.sqrt(np.tanh(k*d))*(4 + 32*S - 116*S**2 - 400*S**3 -
                        71*S**4 + 146*S**5)/(32*(1-S)**5),
            "D2" : -np.sqrt(1/np.tanh(k*d))/2,
            "D4" : np.sqrt(1/np.tanh(k*d))*(2 + 4*S + S**2 + 2*S**3)/(8*(1-S)**3),
            "E2" : np.tanh(k*d)*(2 + 2*S + 5*S**2)/(4*(1-S)**2),
            "E4" : np.tanh(k*d)*(8 + 12*S - 152*S**2 - 308*S**3 - 42*S**4 + 
                        77*S**5)/(32*(1-S)**5)
        }
        
        # Store properties
        self.k = k
        self._coef = coef

        return None

    def _fit_stream(self, **kwargs):
        """
        Fits the stream function wave theory to observations.
        """

        tol = kwargs["tol"] if "tol" in kwargs.keys() else 1e-6
        max_iter = kwargs["max_iter"] if "max_iter" in kwargs.keys() else 1000

        print("Fitting stream wave theory to input conditions...\n")

        # Wave parameters
        H = self.H
        d = self.d
        N = self.N
        M = 10

        # Assemble linear solution based on Table 1
        z0 = self._stream_linear(H/M)
        
        # Increase wave height at each step, using solution from the last step 
        # as initial guess, until the desired wave height is reached
        for m in range(1,M+1):
            h = H*m/M

            # Iterate until solution has converged or reached the maximum 
            # number of iterations
            err = np.inf; count = 0
            while (err > tol) & (count < max_iter):

                # Evaluate the function value and Jacobian
                f = self._stream_eval(h, z0)
                J = self._stream_jacobian(h, z0)
                
                # Solve the system of linear equations
                f_calc, z_calc = f[1:].reshape(-1,1), z0[1:].reshape(-1,1)
                z1 = -np.linalg.solve(J, f_calc) + z_calc
                err = np.max(np.abs(z1 - z_calc))
                count += 1
                z0 = np.append(0, z1.ravel())
            
            # Calculate KFSBC and DFSBC and output to console
            f = self._stream_eval(h, np.append(0, z1.ravel()))
            KFSBC = np.sqrt(np.sum(f[9:10+N]**2))
            DFSBC = np.sqrt(np.sum(f[10+N:]**2))
            print("H=%.3fm: KFSBC=%.2e, DFSBC=%.2e"%(h, KFSBC, DFSBC))

        # Post-processing
        # Compute Fourier components using the equation under "Output of results"
        k = z0[1]/d
        Y = np.zeros((N, ))
        mul = np.ones((N+1, ))
        mul[0] = 0.5; mul[-1] = 0.5
        for j in range(N):
            Y[j] = np.sum(z0[10:10+N+1]*mul*np.cos(np.linspace(0,1,N+1)*(j+1)*np.pi))
        Y *= 2/N

        # Store properties
        self.k = k
        self.Y = Y
        self.z = z0
        
        return None

    def predict(self, x:np.ndarray, z=0, t=0):
        """
        Predicts wave kinematics based on fitted theory. Outputs are arranged
        in the following order: 
        - 1st dimension (ii): x
        - 2nd dimension (jj): z
        - 3rd dimension (kk): t
        
        Parameters
        ----------
        x : float or numpy.ndarray
            Spatial coordinates in the direction of wave travel in metres
        z : float or numpy.ndarray, default 0
            Elevation in metres. 0 and -d represent SWL and seabed respectively
        t : float or numpy.ndarray, default 0
            Time vector in seconds
        
        Returns
        -------
        eta : numpy.ndarray
            Surface elevation in metres. 0 corresponds to SWL
        u : numpy.ndarray
            Wave-induced horizontal particle velocity in metres/second
        v : numpy.ndarray
            Wave-induced vertical particle velocity in metres/second
        """

        # Create mesh grid for unique inputs
        x, z, t = [np.unique(np.sort(np.array(vec).ravel())) for vec in [x, z, t]]
        X, Z, T = np.meshgrid(x, z, t, indexing="ij")

        # Use the suitable theory to predict kinematics
        if self.theory == "Linear":
            return self._predict_linear(X, Z, T)
        elif self.theory == "Stokes_5th":
            return self._predict_stokes5(X, Z, T)
        elif self.theory == "Stream":
            return self._predict_stream(X, Z, T)
    
    def _predict_linear(self, x:np.ndarray, z:np.ndarray, t:np.ndarray):
        """
        Predicts wave kinematics using linear wave theory.
        """

        print("Predicting wave kinematics using linear wave theory...\n")

        # Wave parameters
        H = self.H
        T = self.T
        k = self.k
        d = self.d
        om = 2*np.pi/T
        g = 9.80665

        theta = k*x - om*t

        # Compute kinematics
        eta = H/2*np.cos(theta)
        u = H/2*om*np.cosh(k*(z+d))/np.sinh(k*d)*np.cos(theta)
        w = H/2*om*np.sinh(k*(z+d))/np.sinh(k*d)*np.sin(theta)

        return eta[:,[0],:], u, w

    def _predict_stokes5(self, x:np.ndarray, z:np.ndarray, t:np.ndarray):
        """
        Predicts wave kinematics using 5th order Stokes wave theory.
        """

        print("Predicting wave kinematics using Stokes 5th order wave theory...\n")

        # Wave parameters
        H = self.H
        T = self.T
        k = self.k
        d = self.d
        coef = self._coef.copy()
        om = 2*np.pi/T
        g = 9.80665

        theta = k*x - om*t
        eps = H*k/2
        scale = coef['C0']*np.sqrt(g/k**3)

        # Compute surface elevation using equation 14
        eta = eps*np.cos(theta) +\
                eps**2*coef['B22']*np.cos(2*theta) +\
                eps**3*coef['B31']*(np.cos(theta) - np.cos(3*theta)) +\
                eps**4*(coef['B42']*np.cos(2*theta) + coef['B44']*np.cos(4*theta)) +\
                eps**5*(-(coef['B53'] + coef['B55'])*np.cos(theta) +\
                    coef['B53']*np.cos(3*theta) + coef['B55']*np.cos(5*theta))
        eta /= k

        # Compute particle velocities by taking derivatives of equation 12
        u = 0; w = 0
        for i in range(1,6):
            for j in range(1,6):
                A_ij = 'A%d%d'%(i,j)
                if A_ij in coef.keys():
                    u += scale*eps**i*j*k*coef[A_ij]*\
                            np.cosh(j*k*(z+d))*np.cos(j*theta)
                    w += scale*eps**i*j*k*coef[A_ij]*\
                            np.sinh(j*k*(z+d))*np.sin(j*theta)
        
        return eta[:,[0],:], u, w

    def _predict_stream(self, x:np.ndarray, z:np.ndarray, t:np.ndarray):
        """
        Predict wave kinematics using stream function wave theory.
        """

        print("Predicting wave kinematics using Stream function wave theory...\n")

        # Wave parameters
        T = self.T
        d = self.d
        k = self.k
        N = self.N
        z0 = self.z
        Y = self.Y
        om = 2*np.pi/T
        g = 9.80665

        theta = k*x - om*t

        # Compute kinematics using equations under "Calculations of fluid
        # velocity and pressure"
        eta = np.zeros_like(x).astype(np.float64)
        u = np.zeros_like(x).astype(np.float64)
        w = np.zeros_like(x).astype(np.float64)
        mul = np.ones_like(Y)
        mul[-1] = 0.5
        for j in range(1,N+1):
            eta += Y[j-1]*mul[j-1]*np.cos(j*theta)
            u += j*z0[10+N+j]*np.cosh(j*k*(z+d))/np.cosh(j*z0[1])*np.cos(j*theta)
            w += j*z0[10+N+j]*np.sinh(j*k*(z+d))/np.cosh(j*z0[1])*np.sin(j*theta)

        eta /= k
        u *= np.sqrt(g/k)
        w *= np.sqrt(g/k)

        return eta[:,[0],:], u, w

    def _stream_eval(self, h:float, z:np.ndarray):
        """
        Assemble stream function's function vector. The first entry is a 
        dummy variable with value 0 such that the indexing is consistent with
        Fenton's approach.
        
        Parameters
        ----------
        h : float
            Wave height in metres
        z : numpy.ndarray
            Stream function's variable vector
            
        Returns
        -------
        f : numpy.ndarray
            Stream function's function vector, evaluated at `z`
        """

        # Wave parameters
        T = self.T
        d = self.d
        N = self.N
        Ce = self.Ce
        Cs = self.Cs
        g = 9.80665

        # Initiate function vector
        f = np.zeros((2*N+10+1, ))

        # Equations 1 through 8
        f[1] = z[2] - (h/d)*z[1]
        f[2] = z[2] - (h/(g*T**2))*z[3]**2
        f[3] = z[3]*z[4] - 2*np.pi
        f[4] = z[5] + z[7] - z[4]
        f[5] = z[6] + z[7] - z[4] - z[8]/z[1]
        f[6] = z[5] - Ce*np.sqrt(z[2])/np.sqrt(g*h) if Cs == 0 else \
                    z[6] - Cs*np.sqrt(z[2]/(g*h))
        f[7] = z[10] + 2*np.sum(z[10+1:N+10]) + z[N+10]
        f[8] = z[10] - z[N+10] - z[2]
        
        # KFSBC (9 through N+9) & DFSBC (N+10 through 2N+10)
        for m in range(N+1):
            f[9+m] = -z[8] - z[10+m]*z[7]
            u = -z[7]; w = 0

            for j in range(1, N+1):
                f[9+m] += z[10+N+j]*np.sinh(j*(z[1] + z[10+m]))/\
                            np.cosh(j*z[1])*np.cos(j*m*np.pi/N)
                u += j*z[10+N+j]*np.cosh(j*(z[1] + z[10+m]))/\
                            np.cosh(j*z[1])*np.cos(j*m*np.pi/N)
                w += j*z[10+N+j]*np.sinh(j*(z[1] + z[10+m]))/\
                            np.cosh(j*z[1])*np.sin(j*m*np.pi/N)
            
            f[10+N+m] = (0.5*u**2)+(0.5*w**2) + z[10+m] - z[9]
            
        return f

    def _stream_linear(self, h:float):
        """
        Assemble linear estimates for stream function's variable vector. 
        The first entry is a dummy variable with value 0 such that the indexing
        is consistent with Fenton's approach.
        
        Parameters
        ----------
        h : float
            Wave height in metres
            
        Returns
        -------
        z : numpy.ndarray
            Stream function's variable vector (linear estimate)
        """

        # Wave parameters
        T = self.T
        d = self.d
        N = self.N
        Ce = self.Ce
        Cs = self.Cs
        g = 9.80665
        
        # Linear k
        k = kfinder(2*np.pi/T, d)

        # Initiate variable vector
        z = np.zeros((2*N+10+1, ))
        k_eta = 0.5*k*h*np.cos(np.linspace(0,1,N+1)*np.pi)

        # Input estimates for variables from Table 1
        z[1] = k*d
        z[2] = k*h
        z[3] = 2*np.pi/np.sqrt(np.tanh(k*d))
        z[4] = np.sqrt(np.tanh(k*d))
        z[5] = Ce*np.sqrt(k/g)
        z[6] = Cs*np.sqrt(k/g)
        z[7] = np.sqrt(np.tanh(k*d))
        z[8] = 0
        z[9] = 0.5*np.tanh(k*d)
        z[10:10+N+1] = k_eta
        z[10+N+1] = 0.5*k*h/np.sqrt(np.tanh(k*d))

        return z

    def _stream_jacobian(self, h:float, z:np.ndarray):
        """
        Assemble the Jacobian for the system of linear equations. The matrix
        is of the right size, i.e. both dimensions are equal to 2N+10.
        
        Parameters
        ----------
        h : float
            Wave height in metres
        z : numpy.ndarray
            Stream function's variable vector
            
        Returns
        -------
        J : numpy.ndarray
            Jacobian for the system of linear equations
        """

        # Wave parameters
        N = self.N

        # Initialise the Jacobian
        J = np.zeros((2*N+10, 2*N+10))

        # Obtain the original function vector
        f = self._stream_eval(h, z)

        # Evaluate the Jacobian
        for i in range(1,2*N+10+1):
            z1 = z.copy()

            # Determine step size and perturb input vector
            dz = z1[i]/100 if z1[i] > 1e-4 else 1e-5
            z1[i] += dz

            # Recompute the perturbed function values
            f1 = self._stream_eval(h, z1)

            # Compute the differential and hence the Jacobian
            J_i = (f1 - f)/dz
            J[:,[i-1]] = J_i[1:].reshape(-1,1)

        return J

class Ray_Tracing(object):
    """Estimation of wave travel direction based on bathymetric inputs."""

    def __init__(self, X:np.ndarray, Y:np.ndarray, Z:np.ndarray, U:np.ndarray, 
                    V:np.ndarray, T:float, dt:float):
        """
        Class initialisation. All inputs must follow the SI system. 
        
        Parameters
        ----------
        X : numpy.ndarray
            X-coordinates of grid
        Y : numpy.ndarray
            Y-coordinates of grid
        Z : numpy.ndarray
            Grid elevation
        U : numpy.ndarray
            Eastward velocity 2D field
        V : numpy.ndarray
            Northward velocity 2D field
        T : float
            Duration of wave tracing in seconds
        dt : float
            Temporal resolution. This value should be adjusted such that the 
            CFL number does not exceed 1
        """

        self.X = X
        self.Y = Y
        self.Z = Z
        self.U = U
        self.V = V
        self.simT = T
        self.dt = dt

        # Compute additional parameters based on inputs
        self.nx, self.ny = Z.shape
        self.xlim = (X.min(), X.max())
        self.ylim = (Y.min(), Y.max())
        self.dx = (X.max() - X.min())/self.nx
        self.dy = (Y.max() - Y.min())/self.ny

        self.nT = int(np.round(T/dt + 1))

        return None
    
    def solve(self, T:float, theta:np.ndarray, x_track:np.ndarray, y_track:np.ndarray):
        """
        Solve the refraction problem using numerical integration. Details of 
        the implementation can be found in the references section below
        
        Parameters
        ----------
        T : float
            Wave period in seconds
        theta : float or numpy.ndarray
            Initial wave angle in degrees (Cartesian convention)
        x_track : numpy.ndarray
            X-coordinates for the starting positions
        y_track : numpy.ndarray
            Y-coordinates for the starting positions
            
        References
        ----------
        Halsne, T., Christensen, K. H., Hope, G., and Breivik, Ø.: Ocean wave 
        tracing v.1: a numerical solver of the wave ray equations for ocean 
        waves on variable currents at arbitrary depths, Geosci. Model Dev., 16, 
        6515–6530, https://doi.org/10.5194/gmd-16-6515-2023, 2023.
        """
        
        # Fetch data
        Z = self.Z
        U = self.U
        V = self.V
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        simT = self.simT 
        nT = self.nT
        X0, XN = self.xlim
        Y0, YN = self.ylim

        # Compute additional inputs
        self.waveT = T
        self.theta = theta
        self.x = x_track
        self.y = y_track
        nrays = len(x_track)

        # Instance initialisation
        wt = Wave_tracing(U=U, V=V, nx=nx, ny=ny, nt=nT, T=simT, dx=dx, dy=dy,
                nb_wave_rays=nrays, domain_X0=X0, domain_XN=XN, domain_Y0=Y0,
                domain_YN=YN, d=-Z.copy())
        
        # Set initial condition and solve the refraction problem
        wt.set_initial_condition(wave_period=T, theta0=theta/180*np.pi, 
                                 ipx=x_track, ipy=y_track)
        wt.solve()

        self.wt = wt

        return None
    
    def plot_rays(self, fig:Figure, ax:Axes, scale=1, bm=None, ticks=None, 
                  contour_kw=None, **kwargs):
        """
        Plot wave rays over input bathymetry. Additional keyword arguments
        will be passed to wave ray plotting.
        
        Parameters
        ----------
        fig, ax : matplotlib.figure.Figure & matplotlib.axes.Axes
            Figure and axes of which data will be casted to
        scale : float, default 1
            Scaling factor applied to the elevation dataset for plotting purposes.
            E.g. scale=3.28 to plot contours in feet instead of metres
        bm : None or str, default None
            File path to basemap raster
        ticks : None or array-like, default None
            Colourbar tick labels
        contour_kw : None or dict, default None
            Keyword arguments to be passed to contour plotting
        """
        
        # Pull previously solved results
        wt = self.wt

        # Initialisation of plotting keywords
        contour_kw = contour_kw if isinstance(contour_kw, dict) else {}
        ct_kw = {"cmap": cmocean.cm.haline}
        ct_kw.update(contour_kw)
        cbar_kw = {} if ticks is None else {"ticks": ticks}
        kw = {"color": "red"}; kw.update(kwargs)

        # Plot basemap or coloured contour
        if bm is not None:
            src = rasterio.open(bm)
            raster = src.read()
            show(raster, transform=src.transform, ax=ax)
            ct = ax.contour(self.X, self.Y, self.Z*scale, **ct_kw)
            cbar = fig.colorbar(ct, ax=ax, **cbar_kw)
        
        else:
            cf = ax.contourf(self.X, self.Y, self.Z*scale, **ct_kw)
            ct_kw.update({"cmap": None, "colors": "black", "linestyles": "solid",
                          "negative_linestyles": "solid"})
            ct = ax.contour(self.X, self.Y, self.Z*scale, **ct_kw)
            cbar = fig.colorbar(cf, ax=ax, **cbar_kw)
        
        cbar.outline.set_edgecolor(None)
        ax.clabel(ct, inline=True)

        # Plot wave rays
        for ii in range(wt.ray_x.shape[0]):
            ax.plot(wt.ray_x[ii,:], wt.ray_y[ii,:], **kw)

        ax.set(xlim=self.xlim, ylim=self.ylim)
        ax.set_aspect("equal", adjustable="box")

        return None


