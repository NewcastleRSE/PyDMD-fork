
import numpy as np
import scipy

from pydmd import DMDBase
from pydmd import MrDMD
from pydmd.dmdoperator import DMDOperator
from pydmd.dmdbase import DMDTimeDict
from pydmd.utils import compute_tlsq
from scipy.linalg import block_diag
from pydmd.dmd_modes_tuner import select_modes, slow_modes
from scipy import signal
from past.utils import old_div


class SampleXY_DMD(DMDBase):
    """
    DMD class that handles explicit X,Y input that has been sampled.
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
        
    def fit(self, X, Y,tend,oX = None,oY = None):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        
        :param X: the input snapshots.
        :param Y: the input snapshots that follow X where Y = AX
        :param tend: end time of snapshots
        :param oX: if data is sampled then pass original X snapshots
        :param oY: if data is sampled then pass original Y snapshots
        :type X: numpy.ndarray
        :type Y: numpy.ndarray
        :type tend: int
        :type oX: numpy.ndarray
        :type oY: numpy.ndarray
        """
        self._snapshotsX, self._snapshotsX_shape = self._col_major_2darray(X)
        self._snapshotsY, self._snapshotsY_shape = self._col_major_2darray(Y)
        
        X = self._snapshotsX.copy()
        Y = self._snapshotsY.copy()
        
        if oX is not None and oY is not None:
            self._snapshots, self._snapshots_shape = self._col_major_2darray(np.concatenate((oX,oY[:,-1,None]),axis = 1))
        else:
            self._snapshots, self._snapshots_shape = self._col_major_2darray(np.concatenate((X,Y[:,-1,None]),axis = 1))

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        self._svd_modes, _, _ = self.operator.compute_operator(X,Y)
        
        self._set_initial_time_dictionary({"t0": 0, "tend": tend-1, "dt": 1})

        self._b = self._compute_amplitudes()

        return self
    
    
    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        .. math::

            \\mathbf{x}(t) \\approx
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\exp \\left( \\omega_{k} t
            \\right) b_{k} = \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\left(
            \\lambda_{k} \\right)^{\\left( t / \\Delta t \\right)} b_{k}

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        temp = np.repeat(
            self.eigs[:, None], self.dmd_timesteps.shape[0], axis=1
        )
        tpow = old_div(
            self.dmd_timesteps - self.original_time["t0"],
            self.original_time["dt"],
        )

        # The new formula is x_(k+j) = \Phi \Lambda^k \Phi^(-1) x_j.
        # Since j is fixed, for a given snapshot "u" we have the following
        # formula:
        # x_u = \Phi \Lambda^{u-j} \Phi^(-1) x_j
        # Therefore tpow must be scaled appropriately.
        tpow = self._translate_eigs_exponent(tpow)
        
        dyna = np.power(temp, tpow) * self.amplitudes[:, None]
        
        if hasattr(self,'tdx') and self.tdx is not None:
            dyna = dyna[:,self.tdx]

        return dyna




class SampleXY_MrDMD(MrDMD):
    """
    Multi-resolution Dynamic Mode Decomposition for explicit and sampled X,Y input

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
        
    def partial_modes(self, level, node=None):
        """
        Return the modes at the specific `level` and at the specific `node`; if
        `node` is not specified, the method returns all the modes of the given
        `level` (all the nodes).

        :param int level: the index of the level from where the modes are
            extracted.
        :param int node: the index of the node from where the modes are
            extracted; if None, the modes are extracted from all the nodes of
            the given level. Default is None.

        :return: the selected modes stored by columns
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]

        modes = np.hstack([
            self.dmd_tree[level, leaf].modes
            for leaf in leaves if self.dmd_tree[level,leaf] is not None
        ])

        return modes
    
    def partial_dynamics(self, level, node=None):
        """
        Return the time evolution of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the time
        evolution of the given `level` (all the nodes). The dynamics are always
        reported to the original time window.

        :param int level: the index of the level from where the time evolution
            is extracted.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.

        :return: the selected dynamics stored by row
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]
        dynamics = block_diag(*tuple(dmd.dynamics
            for dmd in map(lambda leaf: self.dmd_tree[level, leaf], leaves) if dmd is not None))
        return dynamics
    
    def partial_eigs(self, level, node=None):
        """
        Return the eigenvalues of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the eigenvalues
        of the given `level` (all the nodes).

        :param int level: the index of the level from where the eigenvalues is
            extracted.
        :param int node: the index of the node from where the eigenvalues is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.

        :return: the selected eigs
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]
        return np.concatenate([self.dmd_tree[level, leaf].eigs for leaf in leaves if self.dmd_tree[level,leaf] is not None])
    
    def fit(self, X, Y, Tm, SAMPLE_FACTOR: int = 10,decimate: bool = True):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        :param X: the input snapshots.
        :param Y: the input snapshots Y = AX
        :param Tm: tim index of snapshots
        :param SAMPLE_FACTOR: sampling rate
        :param decimate: downsample using scipy.signal.decimate
        :type X: numpy.ndarray or iterable
        :type Y: numpy.ndarray or iterable
        :type Tm: numpy.ndarray of numpy.int64
        :type SAMPLE_FACTOR: int
        :type decimate: bool
        """
        self._snapshotsX, self._snapshots_shapeX = self._col_major_2darray(X)
        self._snapshotsY, self._snapshots_shapeY = self._col_major_2darray(Y)

        # Redefine max level if it is too big.
        lvl_threshold = (
            int(np.log(self._snapshotsX.shape[1] / 4.0) / np.log(2.0)) + 1
        )
        if self.max_level > lvl_threshold:
            self.max_level = lvl_threshold
            self._build_tree()
            print(
                "Too many levels... "
                "Redefining `max_level` to {}".format(self.max_level)
            )
        X = self._snapshotsX.copy()
        Y = self._snapshotsY.copy()
        
        T = np.arange(Tm[-1]+1)

        for level in self.dmd_tree.levels:

            n_leaf = 2 ** level
            Ts = np.array_split(T, n_leaf)

            for leaf, t in enumerate(Ts):
                tm = sorted(list(set(t) & set(Tm)))
                
                if tm:
                    rho = old_div(float(self.max_cycles), len(t))
                    if SAMPLE_FACTOR is not None:
                        sub = max(round(1/rho/2/SAMPLE_FACTOR/np.pi),1)
                    else:
                        sub = 1
                    
                    x = X[:,np.isin(Tm[:-1],tm[:-1])]
                    y = Y[:,np.isin(Tm[1:],tm[1:])]
                    
                    if decimate:
                        if sub > 1:
                            sample_x = signal.decimate(x,sub,ftype = 'fir',axis = 1)
                            sample_y = signal.decimate(y,sub,ftype = 'fir',axis = 1)
                        else:
                            sample_x = x
                            sample_y = y
                    else:
                        sample_x = x[:,::sub]
                        sample_y = y[:,::sub]
                        
                    sample_tm = tm[::sub]
                                   
                    current_dmd = self.dmd_tree[level, leaf]
                    current_dmd.fit(sample_x,sample_y,len(t),x,y)
        
                    current_dmd.tdx = np.array(tm) - t[0]

                    current_dmd.rho = rho
                    current_dmd.sub = sub
                    select_modes(current_dmd,slow_modes)
                   
                else:
                    self.dmd_tree[level,leaf] = None
                
            newX = np.hstack([
                self.dmd_tree[level, leaf].reconstructed_data
                for leaf in self.dmd_tree.index_leaves(level) if self.dmd_tree[level, leaf] is not None
            ]).astype(X.dtype)
            
            X -= newX[:,:-1]
            Y -= newX[:,1:]


        return self
 