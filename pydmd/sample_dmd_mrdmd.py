import numpy as np
import scipy

from pydmd import DMDBase
from pydmd import MrDMD
from pydmd.dmdoperator import DMDOperator
from pydmd.dmdbase import DMDTimeDict
from pydmd.utils import compute_tlsq
from pydmd.dmd_modes_tuner import select_modes, slow_modes
from scipy import signal
from past.utils import old_div

class Sample_DMD(DMDBase):
    """
    DMD class that samples input.
    """
    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        func=None
    ):
        super().__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact,
            opt=opt, rescale_mode=rescale_mode, forward_backward=forward_backward,
            sorted_eigs=sorted_eigs)
        
    def fit(self, X, sub, decimate=True):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        
        :param X: the input snapshots
        :param sub: downsample rate
        :param decimate: downsample using scipy.signal.decimate default:True
        :type X: numpy.ndarray
        :type sub: int
        :type decimate: bool
        """

        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        
        Z = self._snapshots.copy()
        
        if decimate:
            if sub > 1:
                Z = signal.decimate(Z,sub,ftype = 'fir',axis = 1)
        else:
            Z = Z[:,::sub]
        
        X = Z[:, :-1]
        Y = Z[:, 1:]
        
        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self._svd_modes, _, _ = self.operator.compute_operator(X,Y)
        
        n_samples = self._snapshots.shape[1]
        self._original_time = DMDTimeDict({'t0': 0, 'tend': n_samples - 1, 'dt': sub})
        self._dmd_time = DMDTimeDict({'t0': 0, 'tend': n_samples - 1, 'dt': 1})
        
        self._b = self._compute_amplitudes()

        return self


class Sample_MrDMD(MrDMD):
    """
    Multi-resolution Dynamic Mode Decomposition for sampled data

    :param DMDBase dmd: an instance of a subclass of `DMDBase`, used to
        recursively analyze the dataset.
    :param int max_cycles: the maximum number of mode oscillations in any given
        time scale. Default is 1.
    :param int max_level: the maximum number of levels. Default is 6.
    """

    def __init__(self,
                 dmd,
                 max_level=2,
                 max_cycles=1):

            self.dmd = dmd
            self.max_cycles = max_cycles
            self.max_level = max_level
            self._build_tree()

    def __iter__(self):
        return self.dmd_tree.__iter__()
    
    def __init__(self,
                 dmd,
                 max_level=2,
                 max_cycles=1
    ):
        super().__init__(dmd=dmd,max_cycles=max_cycles,max_level=max_level)
        
 
    def fit(self, X, SAMPLE_FACTOR, decimate = True):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :param SAMPLE_FACTOR: Factor used to sample data
        :param decimate: boolean to use scipy.decimate to downsample the data
        :type X: numpy.ndarray or iterable
        :type SAMPLE_FACTOR: int
        :type decimate: boolean
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # Redefine max level if it is too big.
        lvl_threshold = int(np.log(self._snapshots.shape[1]/4.)/np.log(2.)) + 1
        if self.max_level > lvl_threshold:
            self.max_level = lvl_threshold
            self._build_tree()
            print('Too many levels... '
                  'Redefining `max_level` to {}'.format(self.max_level))

        X = self._snapshots.copy()
        for level in self.dmd_tree.levels:

            n_leaf = 2**level
            Xs = np.array_split(X, n_leaf, axis=1)

            for leaf, x in enumerate(Xs):
               
                rho = old_div(float(self.max_cycles), x.shape[1])
                if SAMPLE_FACTOR is not None:
                    sub = max(round(1/rho/2/SAMPLE_FACTOR/np.pi),1)
                else:
                    sub = 1
                
                current_dmd = self.dmd_tree[level, leaf]
                current_dmd.fit(x,sub,decimate=True)
                
                current_dmd.rho = rho
                current_dmd.sub = sub
                select_modes(current_dmd,slow_modes)
                
            newX = np.hstack([
                self.dmd_tree[level, leaf].reconstructed_data
                for leaf in self.dmd_tree.index_leaves(level)
            ])
            X -= newX


        self._dmd_time = DMDTimeDict(
            dict(t0 = 0, tend = self._snapshots.shape[1], dt= 1))
        self._original_time = DMDTimeDict(
            dict(t0 = 0, tend = self._snapshots.shape[1], dt= 1))

        return self