from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.state_dict2 = {}
        for k, v in state_dict.items():
            self.state_dict2[v] = k

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        # Initial values
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                alpha[s, t] = sum(self.A[s_prime, s] * alpha[s_prime, t-1]
                                  for s_prime in range(S))
                alpha[s, t] *= self.B[s, self.obs_dict[Osequence[t]]]
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for s in range(S):
            beta[s, L-1] = 1

        for t in reversed(range(L-1)):
            for s in range(S):
                beta[s, t] = sum(self.A[s, s_prime] * self.B[s_prime, self.obs_dict[Osequence[t+1]]]
                                 * beta[s_prime, t+1] for s_prime in range(S))
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        # sum up all alpha's in last column
        prob = sum(alpha[:, -1])
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        seq_prob = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        for i in range(S):
            for t in range(L):
                prob[i, t] = alpha[i, t] * beta[i, t] / seq_prob
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        for s in range(S):
            for s_prime in range(S):
                for t in range(L-1):
                    prob[s, s_prime, t] = alpha[s, t] * self.A[s, s_prime] * \
                        self.B[s_prime, self.obs_dict[Osequence[t+1]]] * \
                        beta[s_prime, t+1] / seq_prob
        return prob

    def viterbi(self, Osequence):
        """
        Viterbi algorithm
        Inputs:
        - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - O: A list of observation sequence (in terms of index, not the actual symbol)
        Returns:
        - path: A list of the most likely hidden state path k* (in terms of the state index)
        argmax_k P(s_k1:s_kT | x_1:x_T)
        """
        path = []
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])    # delta[s, t]'s
        Delta = {}                  # Dictionary of optimal paths ending at each state
        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]
            Delta[s] = [s]

        for t in range(1, L):
            # Build up best path for each state iteratively from t=2 to T
            best_path = {}
            for s in range(S):
                deltas = [self.B[s, self.obs_dict[Osequence[t]]] *
                          self.A[s_prime, s] * delta[s_prime, t-1] for s_prime in range(S)]
                delta[s, t] = max(deltas)
                state = np.argmax(deltas)
                best_path[s] = Delta[state] + [s]
            Delta = best_path

        state = np.argmax(delta[:, -1])
        path_indices = Delta[state]
        for i in path_indices:
            path.append(self.state_dict2[i])
        return path
