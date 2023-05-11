DEBUG = True
if DEBUG:
    from time import time
import numpy as np
from itertools import combinations

# Threshhold from problem 3e in Part I
entropy_thresh = -3.6
start = time()

P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',')
M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',')
alphabet_arr = np.loadtxt('./data/alphabet.csv', delimiter=',', dtype=str).squeeze()
alphabet = {letter: idx for idx, letter in enumerate(alphabet_arr)}

def fast_decode(ciphertext: str, has_breakpoint: bool) -> str:
    """Part II of project:
    
    1. Vectorize inverse, plaintext, and acceptance-prob computation (target < 120 s. current: 109 s)
    2. Design and implement breakpoint (max likelihood of 2 ciphers, some split letter in the text)

    Args:
        ciphertext (str): _description_
        has_breakpoint (bool): _description_

    Returns:
        str: _description_
    """

    def sample_proposal_1(f:np.ndarray):
        i,j = np.random.choice(len(P), size=2, replace=False)
        f_prime = f.copy()
        f_prime[i] , f_prime[j] = f[j], f[i]
        return f_prime

    N = float(len(ciphertext))

    combs =list(combinations(range(len(P)), 2))
    def sample_proposal(f:np.ndarray):
        """Returns the first nonzero proposal sampled"""
        ij = np.random.choice(len(combs))
        i,j = combs[ij]
        f_i,f_j = f[i], f[j]
        f[i] = f_j
        f[j] = f_i
        while not is_nonzero(f):
            combs.pop(ij) # combs[ij] was zero, so don't try sampling this again
            f[i] = f_i
            f[j] = f_j
            ij = np.random.choice(len(combs))
            i,j = combs[ij]
            f_i,f_j = f[i], f[j]
            f[i] = f_j
            f[j] = f_i
        return f

    def is_nonzero(f):
        if np.isclose(P[inverse(f, ciphertext[0])], 0): return False
        for i in range(len(ciphertext))[1:]:
            if np.isclose(M[inverse(f, ciphertext[i]), inverse(f, ciphertext[i-1])], 0): return False
        return True

    def compute_acceptance(f, f_prime):
        "Vectorize"
        p_y_f, p_y_f_prime = 0,0
        for i in range(len(ciphertext)):
            if i==0:
                p_y_f += np.log2(P[inverse(f, ciphertext[i])])
                p_y_f_prime +=np.log2( P[inverse(f_prime, ciphertext[i])])
            else:
                p_y_f += np.log2( M[inverse(f, ciphertext[i]), inverse(f, ciphertext[i-1])]).squeeze()
                p_y_f_prime += np.log2(M[inverse(f_prime, ciphertext[i]), inverse(f_prime, ciphertext[i-1])]).squeeze()
        
        if  p_y_f_prime > p_y_f:
            a = 1
        else:
            # Round to 10 decimal places
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 10))
        ll = p_y_f/N 
        converged = ll>= entropy_thresh
        return a, converged, ll
    
    def get_plaintext(f, ciphertext):
        """Vectorize with embedding lookup and "".join(inverse(f, ciphertext))"""
        plaintext = ''
        for i in range(len(ciphertext)):
            a = alphabet_arr[inverse(f, ciphertext[i])]
            plaintext += str(a )
        return plaintext
    
    def inverse(f, letter:str):
        """Vectorize lookup"""
        i = alphabet[letter]
        return np.where(f == i)[0][0]
        
    # Do random init
    f = np.random.permutation(len(P))
    while not is_nonzero(f):
        f = np.random.permutation(len(P))

    converged = False
    iter = 0
    while not converged:

        # s = time()

        f_prime = sample_proposal_1(f)
        while not is_nonzero(f_prime):
            f_prime = sample_proposal_1(f)
        # f_prime = sample_proposal(f)
        # print("sample proposal", time() - s)

        # s = time()
        u, converged, ll = compute_acceptance(f, f_prime)
        # print("compute_acceptance", time() - s)

        if np.random.uniform(0, 1) <= u:
            f = f_prime

        # print("log likelihood", ll)
        # if DEBUG and iter % 200 == 0:
        #     print(f"iter #{iter}: log-likelihood (bits): {ll}\ttime (s):{time() - start}")
        # iter += 1

    # s = time()
    plaintext = get_plaintext(f, ciphertext)
    # print("get_plaintext", time() - s)
    return plaintext