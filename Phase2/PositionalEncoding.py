import numpy as np

class Encoding():
    def get_positional_encoding(self, x, L):
        '''
        Positional encodings needeed coz neural networks tend to approximate low frequencies and miss higher ones. So, we convert all the frequencies to higher frequency domain. 
        This is useful when we want to learn high freuency variation.

        Inputs:
            x = scalar value that you want to get positional encodings for
            L = number of frequencies
        Outputs:
            positional encoding
        '''
        exp = [1 << x for x in range(L)]
        print(exp)
        cos = np.cos(exp* np.pi * x)
        sin = np.sin(exp* np.pi * x)
        return np.concatenate([sin, cos], axis=0)

    def get_positional_encoding_coordinate(self, X, L=10):
        '''
        get positional encoding for x,y,z coordinates
        Returns a (3*L*2) size vector. 3 dimensions, L frequencies, 2 (coz of sin and cos values)
        Default val = 10 as mentioned in the NeRF paper
        '''
        return [self.get_positional_encoding(x) for x in X]

    def get_positional_encoding_viewing_dir(self, D, L=4):
        '''
        get positional encoding for the 3 viewing directions
        Returns a (3*L*2) size vector. 3 dimensions, L frequencies, 2 (coz of sin and cos values)
        Default val = 4 as mentioned in the NeRF paper
        '''
        return [self.get_positional_encoding(d) for d in D]