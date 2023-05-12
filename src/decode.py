DEBUG = True
if DEBUG:
    from time import time
    from itertools import combinations
    start = time()
import numpy as np

entropy_lower_thresh = -4.2
entropy_upper_thresh = -2.3
english_entropy = -3.23

P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',')
M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',')
alphabet_arr = np.loadtxt('./data/alphabet.csv', delimiter=',', dtype=str).squeeze()
alphabet = {letter: idx for idx, letter in enumerate(alphabet_arr)}

def get_plaintext_bp(f1, f2, ciphertext, breakpoint):
    return get_plaintext(f1, ciphertext[:breakpoint]) + get_plaintext(f2, ciphertext[breakpoint:])

def get_plaintext(f, ciphertext):
    """Vectorize with embedding lookup and "".join(inverse(f, ciphertext))"""
    cipher_tok = inverse_vec(f, ciphertext)
    a = alphabet_arr[cipher_tok]
    return ''.join(a.tolist())
 
def sample_proposal_1(f:np.ndarray):
        i,j = np.random.choice(len(P), size=2, replace=False)
        f_prime = f.copy()
        f_prime[i] , f_prime[j] = f[j], f[i]
        return f_prime

def inverse_vec(f, text:str):
    """Returns tokenized plaintext, given ciphertext and cipher f.
    
    f is a permutation of indices, indices are tokenized letters from 'alphabet'
    """
    ciphertext_tok = np.vectorize(alphabet.__getitem__)(np.array(list(text), dtype=str))
    sorter = np.argsort(f)
    plaintext_tokenized = sorter[np.searchsorted(f, ciphertext_tok, sorter=sorter)]
    return plaintext_tokenized

def decode_bp(ciphertext:str) -> str:
    """Part II.
    
    2. Design and implement breakpoint (max likelihood of 2 ciphers, some split letter in the text)
    
    start guess at middle. then move left or right depending on which likelihood larger.
        l_left > l_right: move left
        l_right > l_left: move right
    repeat until both likelihoods are larger than some threshold
    may wait some iterations in between moves, or try every iteration


    Args:
        ciphertext (str): _description_

    Returns:
        str: _description_
    """

    def is_nonzero(f, text, verbose=False):
        plain_tok = inverse_vec(f, text)
        if np.isclose(P[plain_tok[0]], 0): 
            return False
        if np.any(np.isclose(M[plain_tok[1:], plain_tok[:-1]], 0)): 
            if verbose:
                for i in range(1, len(plain_tok)):
                    if np.isclose(M[plain_tok[i], plain_tok[i-1]], 0):
                        print('->'.join(alphabet_arr[plain_tok[i:i+2]]))
                        print(f"{''.join(alphabet_arr[plain_tok[i:min(len(plain_tok), i+10)]])}")
                        print(f"{text_gt[i:min(len(plain_tok), i+10)]}")
                        print()
                        break
            return False

        return True

    def compute_acceptance(f, f_prime, text):
        "Vectorized"
        fN = inverse_vec(f, text)
        fpN = inverse_vec(f_prime, text)
        p_y_f = np.log2(P[fN[0]])
        p_y_f_prime =np.log2( P[fpN[0]])
        p_y_f += np.log2(M[fN[1:], fN[:-1]]).sum()
        p_y_f_prime += np.log2(M[fpN[1:], fpN[:-1]]).sum()
        
        if  p_y_f_prime > p_y_f:
            a = 1
        else:
            # Round 
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 2))
        ll = p_y_f/N 
        converged = (ll >= entropy_lower_thresh) and (ll <= entropy_upper_thresh) # set thresholds based on entropy of English from problem 3e in Part I
        return a, converged, ll

    with open('./data/sample/short_plaintext.txt', 'r') as f:
        text_gt = f.readlines()[0]

    N = float(len(ciphertext))
    # Get a nonzero init for both ciphers
    breakpoints = list(range(1, int(N) - 1))
    # If sampling proposal has this many times get a nonzero f, choose another breakpoint b/c this one unlikely
    thresh = 1500
    thresh1 = 2000 # same reason as for `thresh`, but used in the nonzero checking when sampling the distr

    # Rank best breakpoints by number of iterations they got
    best_breakpoint_info = (-1, (-1, -1, -1), 999)# max iter:  (bp number, f1, f2), (l2 loss)
    # # breakpts left when to stop
    STOP = N - N//4
    while True:

        # Init at new breakpoint
        if len(breakpoints) <= STOP:
            return get_plaintext_bp(best_breakpoint_info[1][1], best_breakpoint_info[1][2], ciphertext, breakpoint=best_breakpoint_info[1][0])
        breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
        while True:
            if DEBUG:
                print(breakpoint)
            retry = False
            f1 = np.random.permutation(len(P))
            i = 0
            while not is_nonzero(f1, ciphertext[:breakpoint]):
                f1 = np.random.permutation(len(P))
                i+=1
                if i >= thresh:
                    retry = True
                    break
                    
            if retry:
                if len(breakpoints) <= STOP:
                    return get_plaintext_bp(best_breakpoint_info[1][1], best_breakpoint_info[1][2], ciphertext, breakpoint=best_breakpoint_info[1][0])
                breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
                continue
            f2 = np.random.permutation(len(P))
            j=0
            while not is_nonzero(f2, ciphertext[breakpoint:], verbose=False):
                f2 = np.random.permutation(len(P))
                if j>= thresh:
                    retry = True
                    break
                j+=1

            if retry:
                if len(breakpoints) <= STOP:
                    return get_plaintext_bp(best_breakpoint_info[1][1], best_breakpoint_info[1][2], ciphertext, breakpoint=best_breakpoint_info[1][0])
                breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
                continue
            break
        # End Init code

        converged = False
        iters = 0
        # MCMC
        while not converged:

            f1p = sample_proposal_1(f1)
            x = 0
            while not is_nonzero(f1p, ciphertext[:breakpoint]):
                f1p = sample_proposal_1(f1)
                if x >= thresh1:
                    retry = True
                    break
                x += 1

            if retry:
                break # init at new breakpoint

            f2p = sample_proposal_1(f2)
            y=0
            while not is_nonzero(f2p, ciphertext[breakpoint:]):
                f2p = sample_proposal_1(f2)
                if y >= thresh1:
                    retry = True
                    break
                y+=1

            if retry:
                break # init at new breakpoint

            u1, converged_left, ll1 = compute_acceptance(f1, f1p, ciphertext[:breakpoint])
            u2, converged_right, ll2 = compute_acceptance(f2, f2p, ciphertext[breakpoint:])


            if np.random.uniform(0, 1) <= u1:
                f1 = f1p
            if np.random.uniform(0,1) <= u2:
                f2 = f2p


            loss = (ll1 - english_entropy)**2 + (ll2 - english_entropy)**2
            if loss < best_breakpoint_info[2]:
                best_breakpoint_info = (iters, (breakpoint, f1, f2), loss)
                print(f"[BEST] iter {iters}: breakpoint {breakpoint}/{N}\t Log likelihoods: left {ll1} right: {ll2} bp_left: {len(breakpoints)}")
            elif DEBUG:
                print(f"iter {iters}: breakpoint {breakpoint}/{N}\t Log likelihoods: left {ll1} right: {ll2} bp_left: {len(breakpoints)}")
            # if best_breakpoint_info[0] < iters:
            #     best_breakpoint_info = (iters, (breakpoint, f1, f2))

            converged = converged_left and converged_right

            if iters >= 1300 or ll1 >= entropy_lower_thresh or ll2 >= entropy_lower_thresh: # entropy of english is -3.23 so shouldn't be greater than that
                break # init at new breakpoint

            iters += 1
        if converged:
            return get_plaintext_bp(best_breakpoint_info[1][1], best_breakpoint_info[1][2], ciphertext, breakpoint=best_breakpoint_info[1][0])


def decode_no_bp(ciphertext: str) -> str:
    """Part II of project:
    problem: ll gets stuck sometimes. Maybe need to keep track of f's done and not do those

    Args:
        ciphertext (str): _description_
        has_breakpoint (bool): _description_

    Returns:
        str: _description_
    """
    N = float(len(ciphertext))

    def is_nonzero(f):
        plain_tok = inverse_vec(f, ciphertext)
        if np.isclose(P[plain_tok[0]], 0): return False
        if np.any(np.isclose(M[plain_tok[1:], plain_tok[:-1]], 0)): return False
        return True

    def compute_acceptance(f, f_prime):
        "Vectorized"
        fN = inverse_vec(f, ciphertext)
        fpN = inverse_vec(f_prime, ciphertext)
        p_y_f = np.log2(P[fN[0]])
        p_y_f_prime =np.log2( P[fpN[0]])
        p_y_f += np.log2(M[fN[1:], fN[:-1]]).sum()
        p_y_f_prime += np.log2(M[fpN[1:], fpN[:-1]]).sum()
        
        if  p_y_f_prime > p_y_f:
            a = 1
        else:
            # Round 
            a = min(1, np.round(np.exp(p_y_f_prime - p_y_f), 2))
        ll = p_y_f/N 
        converged = ll>= entropy_lower_thresh
        return a, converged, ll
   
    # Do random init
    f = np.random.permutation(len(P))
    while not is_nonzero(f):
        f = np.random.permutation(len(P))

    converged = False
    iter = 0
    while not converged:

        f_prime = sample_proposal_1(f)
        while not is_nonzero(f_prime):
            f_prime = sample_proposal_1(f)

        u, converged, ll = compute_acceptance(f, f_prime)

        if np.random.uniform(0, 1) <= u:
            f = f_prime

        # if DEBUG and iter % 200 == 0:
        #     print(f"iter #{iter}: log-likelihood (bits): {ll}\ttime (s):{time() - start}\tu:{u}")
        if iter == 3000:
            break
        # if DEBUG:
        # print(f"iter {iter} ll: {ll}")
        iter += 1

    plaintext = get_plaintext(f, ciphertext)
    return plaintext

    def find_f1_f2_bp_random():
        while True:
            print(breakpoint)
            retry = False
            f1 = np.random.permutation(len(P))
            i = 0
            while not is_nonzero(f1, ciphertext[:breakpoint]):
                f1 = np.random.permutation(len(P))
                i+=1
                if i >= thresh:
                    retry = True
                    break
                    
            if retry:
                breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
                continue
            f2 = np.random.permutation(len(P))
            j=0
            while not is_nonzero(f2, ciphertext[breakpoint:], verbose=False):
                f2 = np.random.permutation(len(P))
                if j>= thresh:
                    retry = True
                    break
                j+=1

            if retry:
                breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
                continue
            break
        return breakpoint, f1, f2
# while True:
#     print(breakpoint)
#     retry = False
#     f1p = sample_proposal_1(f1)
#     i = 0
#     while not is_nonzero(f1p, ciphertext[:breakpoint]):
#         f1p = sample_proposal_1(f1)
#         if i>= thresh1:
#             retry = True
#             break
#         i+=1
#     if retry:
#         breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
#         continue
#     j=0
#     f2p = sample_proposal_1(f2)
#     while not is_nonzero(f2p, ciphertext[breakpoint:]):
#         f2p = sample_proposal_1(f2)
#         if j >= thresh1:
#             retry = True
#             break
#         j+=1
#     if retry:
#         breakpoint = breakpoints.pop(np.random.choice(len(breakpoints)))
#         continue
#     break