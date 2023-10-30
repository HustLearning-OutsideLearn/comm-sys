import numpy as np 

def _calcMMSEFilter(channel: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Calculates the MMSE filter to cancel the inter-stream interference.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            Noise variance.

        Returns
        -------
        W : np.ndarray
            The MMSE receive filter.
        """
        H = channel
        H_H = H.conj().T
        Nt = H.shape[1]
        W = np.linalg.solve(np.dot(H_H, H) + noise_var * np.eye(Nt), H_H)

        return W