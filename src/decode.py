import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

plot = True

P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',')
M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',')
alphabet_arr = np.loadtxt('./data/alphabet.csv', delimiter=',', dtype=str).squeeze()
alphabet = {letter: idx for idx, letter in enumerate(alphabet_arr)}

plot_dir = './plot_data'

def decode(ciphertext: str, has_breakpoint: bool) -> str:

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
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 10))
        return a, p_y_f, p_y_f_prime
    
    def get_plaintext(f, ciphertext):
        plaintext = ''
        for i in range(len(ciphertext)):
            a = alphabet_arr[inverse(f, ciphertext[i])]
            plaintext += str(a )
        return plaintext
    
    def inverse(f, letter:str):
        i = alphabet[letter]
        return np.where(f == i)[0][0]
        
    # 5000 iters had good result
    iters = 100
    state_log_likelihoods = []

    f = np.random.permutation(len(P))
    while not is_nonzero(f):
        f = np.random.permutation(len(P))
    for _ in tqdm(range(iters)):

        f_prime = sample_proposal(f)
        while not is_nonzero(f_prime):
            f_prime = sample_proposal(f)

        u, p_y_f, p_y_f_prime = compute_acceptance(f, f_prime)

        if np.random.uniform(0, 1) <= u:
            f = f_prime
            p_y_f = p_y_f_prime

        if plot:
            state_log_likelihoods.append(p_y_f)

    plaintext = get_plaintext(f, ciphertext)

    print("Iters:", iters)
    if plot:
        np.savetxt(plot_dir + '/log_likelihoods.txt', np.array(state_log_likelihoods), fmt="%.4f")
        plt.plot(np.arange(iters), state_log_likelihoods)
        plt.title("Log likelihoods")
        plt.xlabel("Iteration #")
        plt.ylabel("Log(P(accepted state))")
        plt.savefig(plot_dir + '/3a.png')

    return plaintext
