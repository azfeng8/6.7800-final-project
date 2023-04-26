import numpy as np
from tqdm import tqdm

iters = 1200

P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',')
M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',')
alphabet_arr = np.loadtxt('./data/alphabet.csv', delimiter=',', dtype=str).squeeze()
alphabet = {letter: idx for idx, letter in enumerate(alphabet_arr)}

def fast_decode(ciphertext: str, has_breakpoint: bool) -> str:
    """Part II of project:
    
    1. Vectorize inverse, plaintext, and acceptance-prob computation (target < 120 s. current: 109 s)
        - maybe need to change convergence criteria, or tune the iters
    2. Design and implement breakpoint (max likelihood of 2 ciphers, some split letter in the text)

    Args:
        ciphertext (str): _description_
        has_breakpoint (bool): _description_

    Returns:
        str: _description_
    """

    def sample_proposal(f:np.ndarray):
        i,j = np.random.choice(len(P), size=2, replace=False)
        f_prime = f.copy()
        f_prime[i] , f_prime[j] = f[j], f[i]
        return f_prime

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
                p_y_f += np.log(P[inverse(f, ciphertext[i])])
                p_y_f_prime +=np.log( P[inverse(f_prime, ciphertext[i])])
            else:
                p_y_f += np.log( M[inverse(f, ciphertext[i]), inverse(f, ciphertext[i-1])]).squeeze()
                p_y_f_prime += np.log(M[inverse(f_prime, ciphertext[i]), inverse(f_prime, ciphertext[i-1])]).squeeze()
        
        if  p_y_f_prime > p_y_f:
            a = 1
        else:
            # Round to 10 decimal places
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 10))
        return a
    
    def get_plaintext(f, ciphertext):
        """Vectorize with embedding lookup and "".join(inverse(f, ciphertext))"""
        plaintext = ''
        for i in range(len(ciphertext)):
            a = alphabet_arr[inverse(f, ciphertext[i])]
            plaintext += str(a )
        return plaintext
    
    def inverse(f, letter:str):
        """Vectorize lookup with torch.Embedding"""
        i = alphabet[letter]
        return np.where(f == i)[0][0]
        
    # Do random init
    f = np.random.permutation(len(P))
    while not is_nonzero(f):
        f = np.random.permutation(len(P))

    for _ in (range(iters)):

        f_prime = sample_proposal(f)
        while not is_nonzero(f_prime):
            f_prime = sample_proposal(f)

        u  = compute_acceptance(f, f_prime)

        if np.random.uniform(0, 1) <= u:
            f = f_prime

    plaintext = get_plaintext(f, ciphertext)
    return plaintext