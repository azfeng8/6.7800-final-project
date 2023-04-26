import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

plot = True
iters = 3000
T = 150
# iters = 100
# gt_path = './data/sample/short_plaintext.txt' 
gt_path = './data/sample/plaintext.txt'

plot_dir = './plot_data'

with open(gt_path, 'r') as f:
    gt_arr = np.array(list(''.join(f.readlines())), dtype=str)
N = len(gt_arr)

window = np.zeros(T)
P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',')
M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',')
alphabet_arr = np.loadtxt('./data/alphabet.csv', delimiter=',', dtype=str).squeeze()
alphabet = {letter: idx for idx, letter in enumerate(alphabet_arr)}


def decode(ciphertext: str, has_breakpoint: bool, gt_arr=gt_arr, ret_decode_corr_counts=False, plot_dir=plot_dir) -> str:

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
            # Round to 10 decimal places
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 10))
        return a, p_y_f, p_y_f_prime
    
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
        
    state_log_likelihoods = []
    acceptance_rates = []
    decode_acc = []

    # Do random init
    f = np.random.permutation(len(P))
    while not is_nonzero(f):
        f = np.random.permutation(len(P))

    for i in tqdm(range(iters)):

        f_prime = sample_proposal(f)
        while not is_nonzero(f_prime):
            f_prime = sample_proposal(f)

        u, p_y_f, p_y_f_prime = compute_acceptance(f, f_prime)

        if np.random.uniform(0, 1) <= u:
            f = f_prime
            p_y_f = p_y_f_prime
            window[i % T] =  1
        else:
            f = f
            window[i % T] = 0

        if plot:
            # 3a
            state_log_likelihoods.append(p_y_f)
            # 3b
            acceptance_rates.append(window.sum() / float(min(i + 1, T)))
            # 3c
            decode_acc.append(np.sum(gt_arr == np.array(list(get_plaintext(f, ciphertext)), dtype=str)) / N)


    print("Iters:", iters)
    if plot:
        # np.savetxt(plot_dir + '/log_likelihoods.txt', np.array(state_log_likelihoods), fmt="%.8f")
        # plt.figure()
        # plt.plot(np.arange(iters), state_log_likelihoods)
        # plt.title("Log likelihoods")
        # plt.xlabel("Iteration #")
        # plt.ylabel("Log(P(accepted state))")
        # plt.savefig(plot_dir + '/3a.png')

        # np.savetxt(plot_dir + '/acceptance_rates.txt', np.array(acceptance_rates), fmt="%.8f")
        # plt.figure()
        # plt.plot(np.arange(iters), acceptance_rates)
        # plt.title(f"Acceptance Rates, sliding window avg of {T} iterations")
        # plt.xlabel("Iteration #")
        # plt.ylabel("Acceptance rate")
        # plt.savefig(plot_dir + '/3b.png')

        np.savetxt(plot_dir + '/decode_acc.txt', np.array(decode_acc), fmt="%.8f")
        plt.figure()
        plt.plot(np.arange(iters), decode_acc)
        plt.title("Decoding accuracy")
        plt.xlabel("# iterations")
        plt.ylabel("# chars correct / (# chars in plaintext)")
        plt.savefig(plot_dir +'/3c.png')

    plaintext = get_plaintext(f, ciphertext)
    if ret_decode_corr_counts: return plaintext, np.array(decode_acc) * N
    return plaintext

def decode_segments_indep(ciphertext):
    import os

    num_segments = 17 * 2
    segment_length = len(ciphertext) // num_segments
    plaintext = ''
    decode_accs_segments = []
    for i in range(0, len(ciphertext), int(segment_length)):
        segment_dir = os.path.join(plot_dir, 'segments', f'len_{segment_length}', f"segment_{i}_{i+segment_length}")
        os.makedirs(segment_dir, exist_ok=True)
        pt, decode_correct = decode(ciphertext[i: i + segment_length], False, ret_decode_corr_counts=True, plot_dir=segment_dir, gt_arr=gt_arr[i:i+segment_length])
        plaintext += pt
        # N = min(segment_length, len(ciphertext) - i)
        decode_accs_segments.append(decode_correct / N)
        print(pt)
    print(plaintext)
    run_dir = os.path.join(plot_dir, "segments", f"len_{segment_length}")
    overall_decode_acc = np.array(decode_accs_segments).sum(axis=0) / N
    np.savetxt(os.path.join(run_dir, "overall_decode_acc.txt"), overall_decode_acc, fmt="%.8f")
    plt.figure()
    plt.plot(np.arange(iters), overall_decode_acc)
    plt.title(f"Decode segments independently: {len(decode_accs_segments)} segments, length ~{segment_length}")
    plt.xlabel("# iterations")
    plt.ylabel("decode accuracy")
    plt.savefig(run_dir + '/decode_acc.png')
    
    
if __name__ == "__main__":
    with open('./data/sample/ciphertext.txt', 'r') as f:
        ciphertext = f.readlines()[0]
    decode_segments_indep(ciphertext)