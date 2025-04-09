import numpy as np
from scipy.integrate import quad
from scipy.constants import c as speed_of_light

class Distances:
    """
    Class to comput cosmological distances given an Hubble function
    """

    def __init__(self, hubble_function, h_units=True):
        """
        Parameters
        ----------
        hubble_function : callable
            Hubble function of the form H(z) = H0 * E(z)
        """
        self.hubble_function = hubble_function
        self.h_units = h_units

    def comoving_distance(self, z, quad_kwargs):
        """
        Comoving distance in Mpc
        Parameters
        ----------
        z : float
            Redshift
        quad_kwargs : dict
            Keyword arguments for scipy.integrate.quad
        Returns 
        -------
        float
            Comoving distance in Mpc
        """
        
        inverse_E = lambda zz : 1.0 / self.hubble_function.E(zz)

        if isinstance(z, (list, np.ndarray)):
            integral = np.array([quad(inverse_E, 0, zi, **quad_kwargs)[0] for zi in z])
        else:
            integral = quad(inverse_E, 0, z, **quad_kwargs)[0]

        if self.h_units:
            return integral * speed_of_light/1.e3 / 100
        else:
            return integral * speed_of_light/1.e3 / self.hubble_function.H0
        
    def luminosity_distance(self, z, quad_kwargs):
        """
        Luminosity distance in Mpc
        Parameters
        ----------
        z : float
            Redshift
        quad_kwargs : dict
            Keyword arguments for scipy.integrate.quad
        Returns 
        -------
        float
            Luminosity distance in Mpc
        """
        
        return (1 + z) * self.comoving_distance(z, quad_kwargs)
    
    def transverse_comoving_distance(self, z, quad_kwargs):
        """
        Angular diameter distance in Mpc
        Parameters
        ----------
        z : float
            Redshift
        quad_kwargs : dict
            Keyword arguments for scipy.integrate.quad
        Returns
        -------
        float
            Angular diameter distance in Mpc
        """

        dc = self.comoving_distance(z, quad_kwargs)

        if self.hubble_function.Omega_k == 0:
            return dc
        elif self.hubble_function.Omega_k > 0:
            return dc / np.sqrt(np.abs(self.hubble_function.Omega_k)) * np.sinh(np.abs(self.hubble_function.Omega_k))
        else:
            return dc / np.sqrt(np.abs(self.hubble_function.Omega_k)) * np.sin(np.abs(self.hubble_function.Omega_k))
        
    def angular_diameter_distance(self, z, quad_kwargs):
        """
        Angular diameter distance in Mpc
        Parameters
        ----------
        z : float
            Redshift
        quad_kwargs : dict
            Keyword arguments for scipy.integrate.quad
        Returns
        -------
        float
            Angular diameter distance in Mpc
        """
        return self.transverse_comoving_distance(z, quad_kwargs) / (1 + z)

    def hubble_distance(self, z):
        """
        Hubble distance in Mpc
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Hubble distance in Mpc
        """
        if self.h_units:
            return speed_of_light / 1.e3 / (100 * self.hubble_function.E(z))
        else:
            return speed_of_light / 1.e3 / (self.hubble_function.H(z))

    
    def isotropic_volume_distance(self, z, quad_kwargs):
        """
        Isotropic volume distance in Mpc^3
        Parameters
        ----------
        z : float
            Redshift
        quad_kwargs : dict
            Keyword arguments for scipy.integrate.quad
        Returns
        -------
        float
            Isotropic volume distance in Mpc^3
        """
        
        dm = self.transverse_comoving_distance(z, quad_kwargs)
        dh = self.hubble_distance(z)

        return ( z  * dm**2 * dh)**(1./3)

