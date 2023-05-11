DEBUG = False
if DEBUG:
    from time import time
    start = time()
import numpy as np
# from itertools import combinations

# Threshhold from problem 3e in Part I
entropy_thresh = -3.6

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

    # combs =list(combinations(range(len(P)), 2))

    def is_nonzero(f):
        plain_tok = inverse_vec(f, ciphertext)
        if np.isclose(P[plain_tok[0]], 0): return False
        if np.any(np.isclose(M[plain_tok[1:], plain_tok[:-1]], 0)): return False
        return True

    def compute_acceptance(f, f_prime):
        "Vectorized"
        p_y_f, p_y_f_prime = 0,0
        fN = inverse_vec(f, ciphertext)
        fpN = inverse_vec(f_prime, ciphertext)
        p_y_f += np.log2(P[fN[0]])
        p_y_f_prime +=np.log2( P[fpN[0]])

        p_y_f += np.log2(M[fN[1:], fN[:-1]]).sum()
        p_y_f_prime += np.log2(M[fpN[1:], fpN[:-1]]).sum()
        

        # f_Nm1 = inverse_vec(f, ciphertext)
        # fp_Nm1 = inverse_vec(f_prime, ciphertext)
        # f_N = f_Nm1[1:]
        # f_Nm1 = f_Nm1[:-1]
        # fp_N = fp_Nm1[1:]
        # fp_Nm1 = fp_Nm1[:-1]

        # p_y_f += np.log2(P[f_Nm1[0]])
        # p_y_f_prime +=np.log2( P[fp_Nm1[0]])

        # p_y_f += np.log2(M[f_N, f_Nm1]).sum()
        # p_y_f_prime += np.log2(M[fp_N, fp_Nm1]).sum()
        
        if  p_y_f_prime > p_y_f:
            a = 1
        else:
            # Round 
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 2))
        ll = p_y_f/N 
        converged = ll>= entropy_thresh
        return a, converged, ll
    
    def get_plaintext(f, ciphertext):
        """Vectorize with embedding lookup and "".join(inverse(f, ciphertext))"""
        cipher_tok = inverse_vec(f, ciphertext)
        a = alphabet_arr[cipher_tok]
        return ''.join(a.tolist())
    
    def inverse_vec(f, text:str):
        """Returns tokenized plaintext, given ciphertext and cipher f.
        
        f is a permutation of indices, indices are tokenized letters from 'alphabet'
        """
        ciphertext_tok = np.vectorize(alphabet.__getitem__)(np.array(list(text), dtype=str))
        sorter = np.argsort(f)
        plaintext_tokenized = sorter[np.searchsorted(f, ciphertext_tok, sorter=sorter)]
        return plaintext_tokenized
        
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

        # if iter % 20 == 0:
        #     print(f"iter {iter}\t log likelihood {ll}\t {time() - start}")
        if DEBUG and iter % 200 == 0:
            print(f"iter #{iter}: log-likelihood (bits): {ll}\ttime (s):{time() - start}")
        iter += 1

    # s = time()
    plaintext = get_plaintext(f, ciphertext)
    # print("get_plaintext", time() - s)
    return plaintext