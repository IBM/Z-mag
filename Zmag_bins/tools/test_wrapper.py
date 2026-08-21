import numpy as np

class StandardProblem3:

    """ Standard Problem 3: flower and vortex state of a magnetic cube
        ==============================================================

        Parameters
        ----------
        comm: MPI communicator
        L: cube length
        std: noise for the initial magnetization
        init: initialization branch ('F' or 'V')

        'F': flower initialization (Ubermag SP3 style)
        'V': vortex initialization (Ubermag SP3 style)
        
    """

    def __init__(self, comm, L_cube, init, std):
        self.comm = comm
        self.L = L_cube
        self.init = init
        self.std = std

    
    def _coord_scale(self, x):
        """Detect whether callback coordinates are physical ([0,L]) or normalized ([0,1])."""
        d = x.shape[1]
        max_coord = float(
            max(
                np.max(np.abs(x[0])) if d else 0.0,
                np.max(np.abs(x[1])) if d else 0.0,
                np.max(np.abs(x[2])) if d else 0.0,
            )
        )
        return self.L if max_coord > 1.5 else 1.0

    def _base_m_sp3(self, xn, yn, zn):
        """Return base SP3 Cartesian magnetization seed (before optional noise)."""
        if self.init == 'F':
            # Flower branch seed: uniform +Z.
            mx = 0.0
            my = 0.0
            mz = 1.0

            # Previous seed (kept for reference):
            # dx = xn - 0.5
            # dy = yn - 0.5
            # mx = 0.20 * dx
            # my = 0.20 * dy
            # mz = 1.0
            return (mx, my, mz)
        if self.init == 'V':
            mx = 0.0
            my = np.sin(np.pi / 2.0 * (xn - 0.5))
            mz = np.cos(np.pi / 2.0 * (xn - 0.5))
            return (mx, my, mz)
        raise Exception('The selected initialization is not available. Options are F, V')

    def _normalize_m(self, mx, my, mz):
        n = float(np.sqrt(mx * mx + my * my + mz * mz))
        if n <= 1e-15:
            return (1.0, 0.0, 0.0)
        return (mx / n, my / n, mz / n)

    def eval_m_init(self, x):
        """
        Cartesian initialization for stereographic solvers.
        Returns three arrays (mx, my, mz) with shape (N,).
        """
        d = x.shape[1]
        mx = np.zeros(d, dtype=np.float64)
        my = np.zeros(d, dtype=np.float64)
        mz = np.zeros(d, dtype=np.float64)

        coord_scale = self._coord_scale(x)
        np.random.seed(123 * self.comm.rank)
        noise = np.random.normal(0.0, self.std, size=(d, 3))

        for i in range(d):
            xn = x[0][i] / coord_scale
            yn = x[1][i] / coord_scale
            zn = x[2][i] / coord_scale
            m0 = self._base_m_sp3(xn, yn, zn)
            m = self._normalize_m(
                m0[0] + noise[i, 0],
                m0[1] + noise[i, 1],
                m0[2] + noise[i, 2],
            )
            mx[i], my[i], mz[i] = m[0], m[1], m[2]
        return mx, my, mz



class StandardProblem4:

    """ Standard Problem 4: equilibrium of a magnetic cuboid
        ====================================================

        Parameters
        ----------
        comm: MPI communicator 
        Lx: length along the x direction
        de: distance from the edges

        The initial state has to comply with
        0 <= theta[0] <= pi
        0 <= theta[1] <= 2pi
        
    """

    def __init__(self, comm, Lx, de, std):
        self.comm = comm
        self.Lx = Lx
        self.de = de
        self.std = std
    
    def init_s_shape(self,pos):
        xcoord = pos[0]
        if (xcoord>=self.de) and (xcoord<=self.Lx-self.de):
            return (np.pi/2, 0)
        elif (xcoord<self.de):
            return (np.pi/2, np.pi/4*(1-xcoord/self.de))
        else:
            return (np.pi/2, np.pi/4*(1+(xcoord-self.Lx)/self.de))
    
    def eval_theta_init(self,x):
        d = x.shape[1]
        angles = [np.zeros(d), np.zeros(d)]
        # Create noise and perturb only theta[1] so that mz = 0
        np.random.seed(123*self.comm.rank)
        noise = abs(np.random.normal(np.mean(np.ones(d)*0), self.std, size = d))
        for i in range(d):
            init_angles = self.init_s_shape([x[0][i], x[1][i], x[2][i]])
            angles[0][i], angles[1][i] = init_angles[0], init_angles[1]+noise[i]
        
        # Check constraint
        for t0 in angles[0]:
            assert t0>=0 and t0<=np.pi, 'theta0 = {} does not comply with the constraint 0 <= theta0 <= pi'.format(t0)
        for t1 in angles[1]:
            assert t1>=0 and t1<=2*np.pi, 'theta1 = {} does not comply with the constraint 0 <= theta1 <= 2pi'.format(t1)

        return angles