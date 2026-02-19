import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import erf
import os

class ReadFiles:

  def __init__(self):
    pass

  def ReadStructure(self, path):

    """
    Input:   1. path - Location of POSCAR or CONTCAR file as a string.

    Outputs: 1. Position vectors of all the atoms as numpy array of shape (total number of atoms, 3), where 3 is
                the x,y and z space coordinates.
             2. Dictionary of atomic species and there corresponding number of atoms.
    """
    with open(path,'r') as file:

      lines = file.readlines()

      scaling_factor = float(lines[1].strip())
      lattice_vectors = [lines[i].strip().split() for i in range(2,5)]
      lattice_vectors = scaling_factor*(np.array(lattice_vectors).astype(float))

      atomic_species = lines[5].strip().split()
      number_of_atoms = np.array(lines[6].strip().split()).astype(int)
      tot = sum(number_of_atoms)

      lattice_type = lines[7].strip()

      atomic_positions = [lines[i].strip().split() for i in range(8,8+tot)]
      atomic_positions = np.array(atomic_positions).astype(float)
      atoms = dict(zip(atomic_species, number_of_atoms))
      
      if lattice_type != "Direct":
        latticeInv = np.linalg.inv(lattice_vectors.T)
        Rd = np.array([np.dot(latticeInv,vec) for vec in atomic_positions])
        atomic_positions = Rd

      atomic_positions[atomic_positions > 0.99] -= 1
      atomic_positions = np.dot(atomic_positions, lattice_vectors)
      return (atomic_positions, atoms)
      
      

  def ReadPhononsPhonopy(self, path):

    """
    Input:   1. path: Location of band.yaml file as a string.

    Outputs: 1. Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
             2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
             3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """
    with open(path,'r') as file:
      lines = [ts.strip() for ts in file]

    atomic_masses = []
    freqs = []
    normal_modes = []
    with open(path,'r') as file:
      for line in file:
        if "mass:" in line:
          atomic_masses.append(line.split()[1])
    atomic_masses = np.array(atomic_masses).astype(float)
    total_atoms = len(atomic_masses)
    with open(path,'r') as file:
      line_number = -1
      for line in file:
        line_number += 1
        if "frequency:" in line:
          freqs.append(float(line.split()[1]))
          ev_internal = []
          for i in range(line_number+3,line_number + 4*total_atoms + 2,4):
            xyz = [lines[i+j].split()[2] for j in range(3)]
            ev_internal.append(xyz)
          normal_modes.append(ev_internal)
    freqs = np.array(freqs).astype(float)
    freqs[freqs<0] = 0
    normal_modes = np.array([[[float(x.strip(',')) for x in sublist] for sublist in outer] for outer in normal_modes])
    return atomic_masses, freqs, normal_modes

  def ReadPhononsVasp(self, path, atoms):

    """
    From VASP OUTCAR.
    Input:   1. path: Location of band.yaml file as a string.
             2. atoms: Dictionary of atomic species and there corresponding number of atoms.

    Outputs:  1.Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
              2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
              3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """

    freqs = []
    normal_modes = []
    number_of_atoms = np.array([i[1] for i in atoms.items()])
    total_atoms = np.sum(number_of_atoms)


    with open(path, 'r') as file:
          lines = [line.strip() for line in file]

          index = lines.index("Mass of Ions in am")
          atomic_masses = lines[index + 1].split()[2:]
          atomic_masses = np.array(atomic_masses).astype(float)

          index_init = lines.index("Eigenvectors and eigenvalues of the dynamical matrix")
          index_final = lines.index("ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)")


          for i in range(index_init, index_final + 1):
            internal_modes = []
            if "THz" in lines[i]:
              freqs.append(lines[i].split()[lines[i].split().index("THz") - 1])
              internal_modes = [lines[j].split() for j in range(i + 2, i + 2 + total_atoms)]
              normal_modes.append(internal_modes)

    atomic_masses = np.repeat(atomic_masses, number_of_atoms)
    freqs = np.array(freqs).astype(float)
    sort = np.argsort(freqs)
    freqs = freqs[sort]
    normal_modes = np.array(normal_modes).astype(float)[...,3:]
    normal_modes = normal_modes[sort]
    return atomic_masses, freqs, normal_modes

  def ReadForces(self, path):

    """
    Reads and stores the Forces (eV/Angstrom) on each atom from the OUTCAR file and returns a 2D array.
    """
    forces = []
    start_collecting = False
    lines_buffer = []

    with open(path, 'r') as file:
        for line in file:
          if "TOTAL-FORCE" in line:
                start_collecting = True
                continue
          if start_collecting:
              if "total drift:" in line:
                  break
              lines_buffer.append(line.strip())
    for line in lines_buffer[1:-1]:
      numbers = [float(num) for num in line.split()]
      forces.append(numbers)
    forces = np.array(forces)
    forces = forces[:,3:]
    return forces



class Photoluminescence(ReadFiles):

  def __init__(self):

    """
    Define all the variables by reading the input files like POSCAR_GS/CONTCAR_GS, POSCAR_ES/CONTCAR_ES, and band.yaml.
    """
    self.hbar = 0.6582*np.sqrt(9.646) #sqrt(meV*AMU)*Angstrom
    super().__init__()

  def IV(self, iv_low, iv_high, rv_high):

    """
    This function can be used to obtain a 1D time array with equal intervals.

    iv: Independent Variable;
    rv: Reciprocal Variable.

    Inputs: Min max values of independent variable, and
    max value of rv required by the user.

    div: Minimum resolution of iv.

    Output: 1D array of independent variable (usually time in this case).
    """

    div = (2*np.pi)/(2*rv_high)
    return np.arange(iv_low, iv_high, div)

  def Fourier(self, independent_variable, function):
      iv = independent_variable
      div = iv[1] - iv[0]
      rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
      sort = np.argsort(rv)
      reciprocal_variable = rv[sort]
      dft = np.fft.fft(function)[sort]
      fourier_transform = div*dft*np.exp(-1j*reciprocal_variable*iv[0])
      return reciprocal_variable, fourier_transform

  def InverseFourier(self, independent_variable, function):
      iv = independent_variable
      div = iv[1] - iv[0]
      rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
      sort = np.argsort(rv)
      reciprocal_variable = rv[sort]
      idft = np.fft.ifft(function)[sort]
      inverse_fourier_transform = div*idft*np.exp(1j*reciprocal_variable*iv[0])*len(rv)/(2*np.pi)
      return reciprocal_variable, inverse_fourier_transform

  def Trapezoidal(self, integrand, iv, equally_spaced = True):

    """
    Calculates the integral using Trapezoidal Rule.

    Inputs: integrand and iv are arrays of same dimension. equally_spaced: determines whether the method should integrate using
    equally spaced or unequally spaced intervals.

    Output: Integration result.
    """
    div = iv[1] - iv[0]
    return (div/2)*(np.sum(integrand[1:-1]) + integrand[0] + integrand[-1]) if equally_spaced \
    else np.sum(np.array([((iv[i+1] - iv[i])/2)*(integrand[i+1] + integrand[i]) for i in range(len(iv)-1)]))

  def FreqToEnergy(self, freqs):

    """Coversion of frequencies (THz) to Energy (meV)."""

    return 4.13566*freqs


  def TimeScaling(self, t, reverse = False):

    """
    Changes time array t from femtoseconds to meV^-1. This is a necesaary step after initializing time through IV
    function in order to maintain consistency in units while performing Fourier Transform.
    """
    return t/658.2119 if reverse == False else t*658.2119

  def Lorentzian(self, x, x0, sigma):

    """
    Used to fit Dirac-Delta as Lorentzian function, where sigma = 6 has units of meV.
    The factor of 0.8 multiplying sigma is to make this function have similarities to
    Gaussian for same standard deviation, sigma.
    """
    return ((1/np.pi)*(sigma*0.8))/(((sigma*0.8)**2) + ((x - x0)**2))

  def Gaussian(self, x, x0, sigma):

    """
    Gaussian fit for Dirac-Delta with sigma = 6 (meV) as standard deviation.
    """
    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-((x-x0)**2)/(2*(sigma**2)))

  def ConfigCoordinates(self, masses, R_es, R_gs, modes):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes.
    """
    masses = np.sqrt(masses)
    R_diff = R_es - R_gs
    mR_diff = np.array([masses[i]*R_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mR_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    return qk

  def ConfigCoordinatesF(self, masses, F_es, F_gs, modes, Ek):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes. This function uses forces on atoms rather than their position vectors
    as used in previous function.
    """
    masses = np.sqrt(masses)
    F_diff = (F_es - F_gs)
    mF_diff = np.array([(1/masses[i])*F_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mF_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    qk = (1/Ek**2)*qk*4180.069
    return qk

  def PartialHR(self, freqs, qk):

    """
    Calculates the Sk (unit less) as a 1D array of length equal to total number of normal modes.
    """
    return (2*np.pi*freqs*(qk**2))/(2*0.6582*9.646)

  def SpectralFunction(self, Sk, Ek, E_meV_positive, sigma = 6, Lorentz = False):

    """
    Calculates S(hbar_omega) or S(E) (unit less) by using Gaussian or Lorentzian fit
    for Direc-Delta with sigma = 6 meV by default.

    Ek: Normal mode phonon energies.
    """
    self.sigma = sigma
    if Lorentz == False:
      S_E = np.array([np.dot(Sk,self.Gaussian(i,Ek,sigma)) for i in E_meV_positive])
    else:
      S_E = np.array([np.dot(Sk,self.Lorentzian(i,Ek,sigma)) for i in E_meV_positive])
    return S_E

  def FourierSpectralFunction(self, Sk, Ek, S_E, E_meV_positive):

    """
    Calculates the Fourier transform of S(E) which is equal to S(t).
    """
    t_meV, S_t = self.Fourier(E_meV_positive, S_E)
    S_t_exact = np.array([np.dot(Sk,np.exp(-1j*Ek*i)) for i in t_meV])
    return t_meV, S_t, S_t_exact

  def GeneratingFunction(self, Sk, S_t, t_meV, Ek, E_meV_positive, T):

    """
    Calculates the generating function G(t).
    """
    if T == 0.0:
      G_t = np.exp((S_t) - (np.sum(Sk)))
    else:
      Kb = 8.61733326e-2 # Boltzmann constant in meV/k
      nk = 1/((np.exp(Ek/(Kb*T))) - 1)
      C_E = np.array([np.dot(nk*Sk,self.Gaussian(i,Ek,self.sigma)) for i in E_meV_positive])
      C_t = self.Fourier(E_meV_positive, C_E)[1]
      C_t_inv = self.InverseFourier(E_meV_positive, C_E)[1]
      G_t = np.exp((S_t) - (np.sum(Sk)) + C_t + C_t_inv - 2*np.sum(nk*Sk))
    return G_t

  def OpticalSpectralFunction(self, G_t, t_meV, zpl, gamma):

    """
    Calculates the optical spectra A(E).
    """
    E_meV, A_E =  self.Fourier(t_meV, (G_t*np.exp(1j*zpl*t_meV))*np.exp(-(gamma*np.abs(t_meV))))
    A_E = (1/len(t_meV))*A_E
    return E_meV, A_E

  def LuminescenceIntensity(self, E_meV, A_E, zpl):

    """
    Calculates the normalized photoluminescence (PL), L(E)
    """
    A_E = A_E[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
    E_meV = E_meV[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
    L_E = ((E_meV**3)*A_E)/(self.Trapezoidal(((E_meV**3)*A_E), E_meV))
    return E_meV, L_E

  def InverseParticipationRatio(self, modes):

    """
    Calculates the IPR (1D array) for each mode.
    """
    p = np.einsum("ijk -> ij", modes**2)
    IPR = 1/np.einsum("ij -> i", p**2)
    return IPR
  
  def anharmonic_coefficients(self, masses, F_es, F_gs, modes, Ek, qk):
     """
     Calculates the lamba_k from U = 1/2(wk**2)Q**2 + lambda_k(q**3) 
     """
     qfk = self.ConfigCoordinatesF(masses, F_es, F_gs, modes, Ek)
     qk[np.abs(qk) <= 0] = 1e-6
     lam_k = (qfk - ((Ek**2)/(self.hbar**2))*qk)/(3*(qk**2))
     return lam_k

     

def calculate_spectrum_analytical(
  path_structure_gs,  # Path to ground state structure
  path_structure_es,  # Path to excited state structure
  phonons_source,  # Options: "VASP" or "Phonopy"
  path_phonon_band,  # Path to phonon band data
  zpl,  # Zero Phonon Line (meV)           3405, algo-3395
  gamma,  # Gamma value (meV) - ZPL broadening
  sigma, # Sigma value (meV) - Phonon sideband broadening
  temperature = 0, # Temperature
  tmax = 2000,  # Upper time limit (fs)
  forces = None, #(os.path.expanduser("./OUTCAR_T"), os.path.expanduser("./OUTCAR_GS")),  # Options: None or tuple (ES file path, GS file path)
  spinpurification = False # External method dfor spin singlet excited state
):

    """
    Calculates all factors step by step.
    """
    results = {}
    pl = Photoluminescence()

    R_gs, atoms_gs = pl.ReadStructure(path_structure_gs)
    R_es, atoms_es = pl.ReadStructure(path_structure_es)

    if phonons_source == "Phonopy":
      masses, freqs, modes = pl.ReadPhononsPhonopy(path_phonon_band)
      freqs = freqs[:int(freqs.shape[0]/2)]
      modes = modes[:int(modes.shape[0]/2),...]
    else:
      masses, freqs, modes = pl.ReadPhononsVasp(path_phonon_band, atoms_es)
    

    freqs[freqs < 0.1] = 0.0
    Ek = pl.FreqToEnergy(freqs)
    Ek[Ek == 0] = 0.00001

    if forces != None:
      if spinpurification == True:
        F_es = np.loadtxt(forces[0])
      else:
        F_es = pl.ReadForces(forces[0])
      results["F_es"] = F_es
      F_gs = pl.ReadForces(forces[1])
      results["F_gs"] = F_gs
      qk = pl.ConfigCoordinatesF(masses, F_es, F_gs, modes, Ek)
    else:
      qk = pl.ConfigCoordinates(masses, R_es, R_gs, modes)

    Sk = pl.PartialHR(freqs, qk)

    if zpl != 0:
      Emax = 2.5*zpl
    else:
      Emax = 5000
    tmax_meV = pl.TimeScaling(tmax)
    E_meV_positive = pl.IV(0, Emax, tmax_meV)
    S_E = pl.SpectralFunction(Sk, Ek, E_meV_positive, sigma)

    t_meV, S_t, S_t_exact = pl.FourierSpectralFunction(Sk, Ek, S_E, E_meV_positive)

    G_t = pl.GeneratingFunction(Sk, S_t, t_meV, Ek, E_meV_positive, temperature)


    E_meV, A_E = pl.OpticalSpectralFunction(G_t, t_meV, zpl, gamma)



    E_meV, L_E = pl.LuminescenceIntensity(E_meV, A_E, zpl)



    t_fs = pl.TimeScaling(t_meV, reverse = True)

    IPR = pl.InverseParticipationRatio(modes)

    S_E = S_E[E_meV_positive <= (max(Ek) + 36)]
    E_meV_positive = E_meV_positive[E_meV_positive <= (max(Ek) + 36)]
    S_t = S_t[(t_fs >= 0) & (t_fs <= 550)]
    S_t_exact = S_t_exact[(t_fs >= 0) & (t_fs <= 550)]
    G_t = G_t[(t_fs >= 0) & (t_fs <= 550)]
    t_fs = t_fs[(t_fs >= 0) & (t_fs <= 550)]

    results["R_gs"] = R_gs
    results["R_es"] = R_es
    results["qk"] = qk
    results["modes"] = modes
    results["masses"] = masses
    results["Ek"] = Ek
    results["Sk"] = Sk
    results["E_meV_positive"] = E_meV_positive
    results["S_E"] = S_E
    results["t_meV"] = t_meV
    results["t_fs"] = t_fs
    results["S_t"] = S_t
    results["S_t_exact"] = S_t_exact
    results["G_t"] = G_t
    results["E_meV"] = E_meV
    results["A_E"] = A_E
    results["L_E"] = L_E
    results["IPR"] = IPR  
    
    return results




class NumericalPhotoluminescence(Photoluminescence):

    def __init__(self, num_states_max, Ek_gs, Ek_es, qk, anharmonic_coeffs_mode_gs, anharmonic_coeffs_mode_es):
        self.num_states_max = num_states_max  # Maximum number of basis states.
        self.Ek_gs = Ek_gs   # Phonon energies for 3N modes - 1d array
        self.Ek_es = Ek_es   # Phonon energies for 3N modes - 1d array
        self.qk = qk   # Displacement along mode k of the excited state oscillator - 1d array
        self.anharmonic_coeffs_mode_gs = anharmonic_coeffs_mode_gs  # Anharmonic coefficients for each mode - 1d array
        self.anharmonic_coeffs_mode_es = anharmonic_coeffs_mode_es  # Anharmonic coefficients for each mode - 1d array
        self.hbar = 0.6582*np.sqrt(9.646)
        self.wk_gs = 2*np.pi*(self.Ek_gs/4.13566)/(np.sqrt(9.646))
        self.wk_es = 2*np.pi*(self.Ek_es/4.13566)/(np.sqrt(9.646))
        self.x_coeffs_mode_gs = self.hbar/(2*self.wk_gs)
        self.x_coeffs_mode_es = self.hbar/(2*self.wk_es)

    def ground_state_hamiltonian(self):
        H = np.zeros((len(self.Ek_gs), self.num_states_max, self.num_states_max))
        for i in range(self.num_states_max):
            H[:, i, i] = self.hbar*self.wk_gs*(i + 0.5)
            if i < self.num_states_max - 1:
                H[:, i, i+1] = H[:, i+1, i] = self.anharmonic_coeffs_mode_gs*(self.x_coeffs_mode_gs**1.5)*(3*(i+1)*np.sqrt(i+1))
            if i < self.num_states_max - 3:
                H[:, i, i+3] = H[:, i+3, i] = self.anharmonic_coeffs_mode_gs*(self.x_coeffs_mode_gs**1.5)*(np.sqrt((i+1)*(i+2)*(i+3)))
        eigenenergies_ground, eigenstates_ground,  = np.linalg.eigh(H)
        return eigenenergies_ground, eigenstates_ground
    
    def excited_state_hamiltonian(self):
        coeff_linear_mode = 3*self.anharmonic_coeffs_mode_es*(self.qk**2) - (self.wk_es**2)*self.qk
        coeff_quadtratic_mode = -3*self.anharmonic_coeffs_mode_es*self.qk
        const_mode = 0.5*(self.wk_es**2)*(self.qk**2) - self.anharmonic_coeffs_mode_es*(self.qk**3)

        H = np.zeros((len(self.Ek_es), self.num_states_max, self.num_states_max))
        for i in range(self.num_states_max):
            H[:, i, i] = self.hbar*self.wk_es*(i + 0.5) + self.x_coeffs_mode_es*coeff_quadtratic_mode*(2*i+1) + const_mode
            if i < self.num_states_max - 1:
                H[:, i, i+1] = H[:, i+1, i] = self.anharmonic_coeffs_mode_es*(self.x_coeffs_mode_es**1.5)*(3*(i+1)*np.sqrt(i+1)) \
                    + (self.x_coeffs_mode_es**0.5)*(coeff_linear_mode)*np.sqrt(i+1)
            if i < self.num_states_max - 2:
                H[:, i, i+2] = H[:, i+2, i] = self.x_coeffs_mode_es*coeff_quadtratic_mode*np.sqrt((i+1)*(i+2))
            if i < self.num_states_max - 3:
                H[:, i, i+3] = H[:, i+3, i] = self.anharmonic_coeffs_mode_es*(self.x_coeffs_mode_es**1.5)*(np.sqrt((i+1)*(i+2)*(i+3)))
        vals, vecs = np.linalg.eigh(H)
        eigenenergies_excited = vals[:, 0]
        eigenstates_excited = vecs[:, :, 0]
        return eigenenergies_excited, eigenstates_excited

    def calculate_franck_condon_factors(self, eigenenergies_ground, eigenstates_ground, eigenstates_excited):
        franck_condon_factors = np.array([[(np.abs(np.dot(eigenstates_ground[k,:,i], eigenstates_excited[k,:]))**2) for i in range(self.num_states_max)] \
            for k in range(len(self.Ek_gs))])
        franck_condon_factors /= np.sum(franck_condon_factors, axis=1, keepdims=True)
        energy_phonon_mode = (eigenenergies_ground - eigenenergies_ground[:, 0][:, None])
        energy_phonon_mode[:, 0] = 0.0
        return franck_condon_factors, energy_phonon_mode
    
    def numerical_luminescence(self, franck_condon_factors, energy_phonon_mode, E_meV_positive, zpl, gamma, sigma):
        F = franck_condon_factors
        E = energy_phonon_mode
        G_k_E = np.array([np.dot(F[0,:], self.Gaussian(i, E[0,:], sigma)) for i in E_meV_positive])
        t_meV, _ = self.Fourier(E_meV_positive, G_k_E)
        G_k_t = np.array([[np.dot(F[k,:], np.exp(-1j*E[k,:]*i)) for i in t_meV] for k in range(F.shape[0])])
        G_t = np.prod(G_k_t, axis=0)
        G_t *= np.exp(-(sigma**2)*(t_meV**2)/2)
        E_meV, A_E = self.OpticalSpectralFunction(G_t, t_meV, zpl, gamma)
        E_meV, L_E = self.LuminescenceIntensity(E_meV, A_E, zpl)
        return (t_meV, G_t, E_meV, L_E)

    def test(self):
        self.Sk = self.wk_gs*(self.qk**2)/(2*self.hbar)
        poisson = np.array([[(np.exp(-self.Sk[k])*(self.Sk[k]**i)/(factorial(i))) for i in range(self.num_states_max)] for k in range(len(self.Ek_gs))])
        poisson /= np.sum(poisson, axis=1, keepdims=True)
        return poisson
    
    def monte_carlo_sampling(self, franck_condon_factors, energy_phonon_mode, num_samples, zpl, gamma, sigma):
       
       draws = np.array([np.random.choice(self.num_states_max, size=num_samples, p=franck_condon_factors[k,:]) for k in range(franck_condon_factors.shape[0])])
       energy_phonons = np.sum(energy_phonon_mode[np.arange(franck_condon_factors.shape[0]), draws.T], axis=1)
       energy_photon =  zpl - energy_phonons

       bins = 2*int(np.sqrt(num_samples))
       min_hist_energy_photon = zpl - 2*self.Ek_gs.max()
       max_hist_energy_photon = zpl + 20
       y_photon, x_photon = np.histogram(energy_photon, bins = bins, range = (min_hist_energy_photon, max_hist_energy_photon), density = True) 
       x_photon = x_photon[:-1] + (np.diff(x_photon) / 2)

       def func_photon(x):
          return np.sum(y_photon*np.exp(-((x - x_photon)**2)/(2*(gamma**2))))
       E_meV = np.linspace(x_photon.min(), 1.02*x_photon.max(), 600)
       L_E = np.array([func_photon(i) for i in E_meV])
       L_E = (E_meV**3)*L_E
       L_E = L_E/np.trapezoid(L_E, E_meV)


       min_hist_energy_phonons = self.Ek_gs.min()
       max_hist_energy_phonons = self.Ek_gs.max()
       y_phonons, x_phonons = np.histogram(energy_phonons, bins = bins, range = (min_hist_energy_phonons, max_hist_energy_phonons), density = True) 
       x_phonons = x_phonons[:-1] + (np.diff(x_phonons) / 2)

       def func_phonons(x):
          return np.sum(y_phonons*np.exp(-((x - x_phonons)**2)/(2*(sigma**2))))
       E_meV_phonons = np.linspace(x_phonons.min(), 1.08*x_phonons.max(), 600)
       S_E = np.array([func_phonons(i) for i in E_meV_phonons])

       return (E_meV, L_E), (E_meV_phonons, S_E)



def calculate_spectrum_numerical(num_states_max, 
                                 Ek_gs, 
                                 Ek_es, 
                                 qk, 
                                 anharmonic_coeffs_mode_gs, 
                                 anharmonic_coeffs_mode_es,  
                                 zpl, 
                                 gamma, 
                                 sigma, 
                                 generating_function_simulation=True, 
                                 monte_carlo_simulation=False, 
                                 num_samples=None,
                                 tmax = 2000):
    
    npl = NumericalPhotoluminescence(num_states_max = num_states_max, 
                                     Ek_gs = Ek_gs, 
                                     Ek_es = Ek_es, 
                                     qk = qk, 
                                     anharmonic_coeffs_mode_gs = anharmonic_coeffs_mode_gs, 
                                     anharmonic_coeffs_mode_es = anharmonic_coeffs_mode_es)
    results = {}
    eigenenergies_ground, eigenstates_ground = npl.ground_state_hamiltonian()
    eigenenergies_excited, eigenstates_excited = npl.excited_state_hamiltonian()
    franck_condon_factors, energy_phonon_mode = npl.calculate_franck_condon_factors(eigenenergies_ground, eigenstates_ground, eigenstates_excited)
    poisson_factors_harmonic = npl.test()
    
    results["franck_condon_factors"] = franck_condon_factors
    results["energy_phonon_mode"] = energy_phonon_mode
    results["poisson_factors_harmonic"] = poisson_factors_harmonic

    abs_diff = np.sum(np.abs(franck_condon_factors - poisson_factors_harmonic))
    print(f"Sum of absolute difference between harmonic-poissonian-factors and franck-condon-factor = {abs_diff: .5f}.") 

    if zpl != 0:
      Emax = 2.5*zpl
    else:
      Emax = 5000
    tmax_meV = npl.TimeScaling(tmax)
    E_meV_positive = npl.IV(0, Emax, tmax_meV)
    if generating_function_simulation:
      t_meV, G_t, E_meV, L_E = npl.numerical_luminescence(franck_condon_factors, energy_phonon_mode, E_meV_positive, zpl, gamma, sigma)
      t_fs = npl.TimeScaling(t_meV, reverse = True)

      G_t = G_t[(t_fs >= 0) & (t_fs <= 550)]
      t_fs = t_fs[(t_fs >= 0) & (t_fs <= 550)]

      results["t_meV"] = t_meV
      results["t_fs"] = t_fs
      results["G_t"] = G_t
      results["E_meV"] = E_meV
      results["L_E"] = L_E
      

    if monte_carlo_simulation:
      (E_photon_monte_carlo, L_monte_carlo), (E_phonons_monte_carlo, S_monte_carlo) = npl.monte_carlo_sampling(franck_condon_factors, energy_phonon_mode, num_samples, zpl, gamma, sigma)
      results["E_photon_monte_carlo"] = E_photon_monte_carlo
      results["L_monte_carlo"] = L_monte_carlo
      results["E_phonons_monte_carlo"] = E_phonons_monte_carlo
      results["S_monte_carlo"] = S_monte_carlo

    return results

