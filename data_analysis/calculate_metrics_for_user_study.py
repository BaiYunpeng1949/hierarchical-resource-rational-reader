import os
import re
import json
import glob
import numpy as np
import pandas as pd
import datetime
import constants as const
# import editdistance
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import directed_hausdorff
from itertools import product
# from fastdtw import fastdtw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ScanpathMetrics:
    
    def __init__(self, human_sequence, simulated_sequence, mode=const.MODE_COORD):
        """
        Initialize the ScanpathMetrics with coordinate data.

        Parameters:
        - human_sequence_coords: List of (x, y) tuples from human data.
        - simulated_sequence_coords: List of (x, y) tuples from simulated data.
        - mode: 'coord' for coordinate sequences, 'word_index' for word index sequences.
        - height: Height of the stimulus/display (int). Required if mode='coord'.
        - width: Width of the stimulus/display (int). Required if mode='coord'.
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).

        Reference: Yue, EyeFormer, 2024, https://github.com/YueJiang-nj/EyeFormer-UIST2024/blob/main/evaluation/eval_scanpaths.py
        """
        self.human_sequence = human_sequence
        self.simulated_sequence = simulated_sequence

        self.mode = mode

        if not self.human_sequence or not self.simulated_sequence:
            raise ValueError("Coordinate sequences are required for spatial metrics.")
    
    def map_coords_to_states(self, coords_sequence, height, width, x_bins=10, y_bins=10):
        """
        Map coordinates to a sequence of states based on grid bins.

        Parameters:
        - coords_sequence: List of (x, y) tuples.
        - height: Height of the stimulus/display (int).
        - width: Width of the stimulus/display (int).
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).

        Returns:
        - state_sequence: List of state indices.
        """
        state_sequence = []
        x_step = width / x_bins
        y_step = height / y_bins
        for x, y in coords_sequence:
            x_bin = int(min(x // x_step, x_bins - 1))
            y_bin = int(min(y // y_step, y_bins - 1))
            state = y_bin * x_bins + x_bin
            state_sequence.append(int(state))
        return state_sequence

    def levenshtein_distance(self, height, width, x_bins=10, y_bins=10):
        """
        Compute the Levenshtein distance between human and simulated sequences of coordinates.

        Parameters:
        - height: Height of the stimulus/display (int).
        - width: Width of the stimulus/display (int).
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).

        Returns:
        - dist: The Levenshtein distance (int).
        """
        if self.mode == const.MODE_COORD:
            s1 = self.map_coords_to_states(self.human_sequence, height, width, x_bins, y_bins)
            s2 = self.map_coords_to_states(self.simulated_sequence, height, width, x_bins, y_bins)
        elif self.mode == const.MODE_WORD_INDEX:
            s1 = self.human_sequence
            s2 = self.simulated_sequence
        
        len_s1 = len(s1)
        len_s2 = len(s2)

        # Initialize matrix
        dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j

        # Compute distances
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                               dp[i][j - 1] + 1,      # Insertion
                               dp[i - 1][j - 1] + cost)  # Substitution

        return dp[len_s1][len_s2]
    
    def normalized_levenshtein_distance(self, height, width, x_bins=10, y_bins=10):
        """
        Compute the Normalized Levenshtein Distance between sequences of coordinates.

        Parameters:
        - height: Height of the stimulus/display (int).
        - width: Width of the stimulus/display (int).
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).

        Returns:
        - nld: Normalized Levenshtein distance (float between 0 and 1).
        """
        lev_dist = self.levenshtein_distance(height, width, x_bins, y_bins)
        max_len = max(len(self.human_sequence), len(self.simulated_sequence))
        if max_len == 0:
            return 0.0  # Both sequences are empty
        nld = lev_dist / max_len
        return nld
    
    # def dynamic_time_warping(self):
    #     """
    #     Compute the Dynamic Time Warping distance between coordinate sequences.

    #     Returns:
    #     - dtw_distance: The DTW distance (float).
    #     """
    #     P = np.array(self.human_sequence_coords)
    #     Q = np.array(self.simulated_sequence_coords)

    #     dtw_distance, _ = fastdtw(P, Q, dist=euclidean)
    #     return dtw_distance
    
    def dynamic_time_warping(self):
        """
        Compute the Dynamic Time Warping distance between sequences of coordinates.

        Returns:
        - dtw_distance: The DTW distance (float).
        """
        # s1 = np.array(self.human_sequence)
        # s2 = np.array(self.simulated_sequence)
        if self.mode == const.MODE_COORD:
            s1 = np.array(self.human_sequence)
            s2 = np.array(self.simulated_sequence)

            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            s1 = np.array(self.human_sequence, dtype=float)
            s2 = np.array(self.simulated_sequence, dtype=float)

            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")
        
        len_s1 = len(s1)
        len_s2 = len(s2)

        dtw_matrix = np.full((len_s1 + 1, len_s2 + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                # cost = euclidean(s1[i - 1], s2[j - 1])
                cost = distance(s1[i - 1], s2[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                              dtw_matrix[i, j - 1],    # Deletion
                                              dtw_matrix[i - 1, j - 1])  # Match

        dtw_distance = dtw_matrix[len_s1, len_s2]
        return dtw_distance

    
    def longest_common_subsequence(self, height, width, x_bins=10, y_bins=10):
        """
        Compute the length of the Longest Common Subsequence (LCS) between human and simulated sequences of coordinates.

        Parameters:
        - height: Height of the stimulus/display (int).
        - width: Width of the stimulus/display (int).
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).

        Returns:
        - lcs_length: Length of the LCS (int).
        """
        if self.mode == const.MODE_COORD:
            s1 = self.map_coords_to_states(self.human_sequence, height, width, x_bins, y_bins)
            s2 = self.map_coords_to_states(self.simulated_sequence, height, width, x_bins, y_bins)
        elif self.mode == const.MODE_WORD_INDEX:
            s1 = self.human_sequence
            s2 = self.simulated_sequence
        else:
            raise ValueError("Unknown mode.")
        
        len_s1 = len(s1)
        len_s2 = len(s2)
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        for i in range(len_s1):
            for j in range(len_s2):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        lcs_length = dp[len_s1][len_s2]
        return lcs_length
    
    def frechet_distance(self):
        """
        Compute the Frechet distance between two sequences of coordinates.

        Returns:
        - frechet_dist: The Frechet distance (float).
        """
        P = self.human_sequence  # List of (x, y) tuples
        Q = self.simulated_sequence  # List of (x, y) tuples

        if self.mode == const.MODE_COORD:
            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")

        len_P = len(P)
        len_Q = len(Q)
        ca = np.full((len_P, len_Q), -1.0)

        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                # ca[i, j] = euclidean(P[0], Q[0])
                ca[i, j] = distance(P[0], Q[0])
            elif i > 0 and j == 0:
                # ca[i, j] = max(c(i - 1, 0), euclidean(P[i], Q[0]))
                ca[i, j] = max(c(i - 1, 0), distance(P[i], Q[0]))
            elif i == 0 and j > 0:
                # ca[i, j] = max(c(0, j - 1), euclidean(P[0], Q[j]))
                ca[i, j] = max(c(0, j - 1), distance(P[0], Q[j]))
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                    # euclidean(P[i], Q[j])
                    distance(P[i], Q[j])
                )
            else:
                ca[i, j] = float('inf')
            return ca[i, j]

        frechet_dist = c(len_P - 1, len_Q - 1)
        return frechet_dist
    
    def hausdorff_distance(self):
        """
        Compute the Hausdorff distance between two sequences of coordinates.

        Returns:
        - hausdorff_dist: The Hausdorff distance (float).
        """
        P = np.array(self.human_sequence)
        Q = np.array(self.simulated_sequence)

        if self.mode == const.MODE_COORD:
            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")
        
        # Compute the directed Hausdorff distances
        def directed_hausdorff_distance(A, B):
            max_min_distance = 0
            for a in A:
                min_distance = np.inf
                for b in B:
                    d = distance(a, b)
                    if d < min_distance:
                        min_distance = d
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
            return max_min_distance

        # hausdorff_dist = max(
        #     directed_hausdorff(P, Q)[0],
        #     directed_hausdorff(Q, P)[0]
        # )
        # return hausdorff_dist

        d_AB = directed_hausdorff_distance(P, Q)
        d_BA = directed_hausdorff_distance(Q, P)
        hausdorff_dist = max(d_AB, d_BA)
        return hausdorff_dist

    def mannan_distance(self):
        """
        Compute the Mannan distance between two sequences of coordinates.

        Returns:
        - mannan_dist: The Mannan distance (float).
        """
        P = np.array(self.human_sequence)
        Q = np.array(self.simulated_sequence)

        if self.mode == const.MODE_COORD:
            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")

        dist_matrix = np.zeros((len(P), len(Q)))

        for i in range(len(P)):
            for j in range(len(Q)):
                # dist_matrix[i, j] = euclidean(P[i], Q[j])
                dist_matrix[i, j] = distance(P[i], Q[j])

        min_dist_P = dist_matrix.min(axis=1).sum()
        min_dist_Q = dist_matrix.min(axis=0).sum()

        mannan_dist = (1 / (len(P) + len(Q))) * (min_dist_P**2 + min_dist_Q**2)
        return mannan_dist
    
    def eyenalysis_distance(self):
        """
        Compute the Eyenalysis distance between two sequences of coordinates.

        Returns:
        - eyenalysis_dist: The Eyenalysis distance (float).
        """
        P = np.array(self.human_sequence)
        Q = np.array(self.simulated_sequence)

        if self.mode == const.MODE_COORD:
            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")

        dist_matrix = np.zeros((len(P), len(Q)))

        for i in range(len(P)):
            for j in range(len(Q)):
                # dist_matrix[i, j] = euclidean(P[i], Q[j])
                dist_matrix[i, j] = distance(P[i], Q[j])

        min_dist_P = dist_matrix.min(axis=1).sum()
        min_dist_Q = dist_matrix.min(axis=0).sum()

        eyenalysis_dist = (1 / (len(P) + len(Q))) * (min_dist_P + min_dist_Q)
        return eyenalysis_dist

    def scan_match(self, height, width, x_bins=10, y_bins=10, gap_value=-1, threshold=1):
        """
        Compute the ScanMatch similarity score between human and simulated scanpaths.

        Parameters:
        - height: Height of the stimulus/display (int).
        - width: Width of the stimulus/display (int).
        - x_bins: Number of bins along the x-axis (int).
        - y_bins: Number of bins along the y-axis (int).
        - gap_value: Gap penalty for alignment (int).
        - threshold: Threshold for substitution matrix (float).

        Returns:
        - similarity_score: The ScanMatch similarity score (float between 0 and 1).
        """
        # def create_substitution_matrix(x_bins, y_bins, threshold):
        #     num_states = x_bins * y_bins
        #     sub_matrix = np.zeros((num_states, num_states))

        #     for i in range(num_states):
        #         y1, x1 = divmod(i, x_bins)
        #         for j in range(num_states):
        #             y2, x2 = divmod(j, x_bins)
        #             distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        #             # Substitution score decreases with distance
        #             sub_matrix[i, j] = max(threshold - distance, 0)
        #     return sub_matrix

        # # Map fixations to grid states
        # P_states = self.map_coords_to_states(self.human_sequence, height, width, x_bins, y_bins)
        # Q_states = self.map_coords_to_states(self.simulated_sequence, height, width, x_bins, y_bins)
        # sub_matrix = create_substitution_matrix(x_bins, y_bins, threshold)

        # # Perform global alignment using the class's static method
        # score = self.global_align(
        #     P_states,
        #     Q_states,
        #     SubMatrix=sub_matrix,
        #     gap=gap_value
        # )
        # # Normalize score to be between 0 and 1
        # max_score = max(len(P_states), len(Q_states)) * threshold
        # similarity_score = score / max_score if max_score != 0 else 0
        # return similarity_score

        if self.mode == const.MODE_COORD:
            if height is None or width is None:
                raise ValueError("Height and width must be provided when mode is 'coord'.")

            def create_substitution_matrix(x_bins, y_bins, threshold):
                num_states = x_bins * y_bins
                sub_matrix = np.zeros((num_states, num_states))

                for i in range(num_states):
                    y1, x1 = divmod(i, x_bins)
                    for j in range(num_states):
                        y2, x2 = divmod(j, x_bins)
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        # Substitution score decreases with distance
                        sub_matrix[i, j] = max(threshold - distance, 0)
                return sub_matrix

            # Map fixations to grid states
            P_states = self.map_coords_to_states(self.human_sequence, height, width, x_bins, y_bins)
            Q_states = self.map_coords_to_states(self.simulated_sequence, height, width, x_bins, y_bins)
            sub_matrix = create_substitution_matrix(x_bins, y_bins, threshold)

        elif self.mode == const.MODE_WORD_INDEX:
            # For word indices, create a substitution matrix where identical indices have maximum score
            P_states = self.human_sequence
            Q_states = self.simulated_sequence
            unique_states = list(set(P_states + Q_states))
            num_states = len(unique_states)
            state_to_index = {state: idx for idx, state in enumerate(unique_states)}
            P_states = [state_to_index[s] for s in P_states]
            Q_states = [state_to_index[s] for s in Q_states]
            sub_matrix = np.full((num_states, num_states), fill_value=0.0)
            np.fill_diagonal(sub_matrix, threshold)
        else:
            raise ValueError("Unknown mode.")

        # Perform global alignment using the class's static method
        score = self.global_align(
            P_states,
            Q_states,
            SubMatrix=sub_matrix,
            gap=gap_value
        )
        # Normalize score to be between 0 and 1
        max_score = max(len(P_states), len(Q_states)) * threshold
        similarity_score = score / max_score if max_score != 0 else 0
        return similarity_score

    @staticmethod
    def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
        """
        Compute alignment between two sequences using global alignment.

        Parameters:
        - P: Sequence 1 (list or array of symbols/states).
        - Q: Sequence 2 (list or array of symbols/states).
        - SubMatrix: Substitution matrix (2D array) for alignment scoring.
        - gap: Gap penalty (int or float).
        - match: Score for a match (int or float).
        - mismatch: Score for a mismatch (int or float).

        Returns:
        - score_max: The maximum alignment score (float).
        """
        UP, LEFT, DIAG, NONE = range(4)
        max_p = len(P)
        max_q = len(Q)
        score = np.zeros((max_p + 1, max_q + 1), dtype='f')
        pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

        pointer[0, 0] = NONE
        score[0, 0] = 0.0
        pointer[0, 1:] = LEFT
        pointer[1:, 0] = UP

        score[0, 1:] = gap * np.arange(1, max_q + 1)
        score[1:, 0] = gap * np.arange(1, max_p + 1)

        for i in range(1, max_p + 1):
            ci = P[i - 1]
            for j in range(1, max_q + 1):
                cj = Q[j - 1]
                if SubMatrix is None:
                    diag_score = score[i - 1, j - 1] + (match if ci == cj else mismatch)
                else:
                    diag_score = score[i - 1, j - 1] + SubMatrix[ci][cj]
                up_score = score[i - 1, j] + gap
                left_score = score[i, j - 1] + gap

                if diag_score >= up_score and diag_score >= left_score:
                    score[i, j] = diag_score
                    pointer[i, j] = DIAG
                elif up_score > left_score:
                    score[i, j] = up_score
                    pointer[i, j] = UP
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

        # The score matrix has been filled; now backtrack to compute the alignment
        # In this implementation, we're only returning the maximum score
        score_max = score[max_p, max_q]
        return score_max

    def time_delay_embedding(self, k=2, distance_mode='mean'):
        """
        Compute the Time Delay Embedding (TDE) distance between sequences.

        Parameters:
        - k: Embedding dimension (int).
        - distance_mode: 'mean' or 'hausdorff' (str).

        Returns:
        - tde_distance: The TDE distance (float).
        """
        P = self.human_sequence
        Q = self.simulated_sequence

        if self.mode == const.MODE_COORD:
            def distance(a, b):
                return euclidean(a, b)
        elif self.mode == const.MODE_WORD_INDEX:
            def distance(a, b):
                return abs(a - b)
        else:
            raise ValueError("Unknown mode.")

        if len(P) < k or len(Q) < k:
            raise ValueError("Sequences are too short for the embedding dimension k.")

        # Create time-delay embeddings
        P_vectors = [np.array(P[i:i + k]).flatten() for i in range(len(P) - k + 1)]
        Q_vectors = [np.array(Q[i:i + k]).flatten() for i in range(len(Q) - k + 1)]

        distances = []
        for q_vec in Q_vectors:
            min_dist = min(
                np.linalg.norm(p_vec - q_vec)
                for p_vec in P_vectors
            )
            distances.append(min_dist / k)

        if distance_mode == 'mean':
            tde_distance = np.mean(distances)
        elif distance_mode == 'hausdorff':
            tde_distance = np.max(distances)
        else:
            raise ValueError("Unknown distance mode.")

        return tde_distance

    def _compute_coincidence_matrix(self, threshold):
        """
        Compute coincidence matrix between human and simulated coordinate sequences.

        Parameters:
        - threshold: Distance threshold for defining coincidence.

        Returns:
        - c: Coincidence matrix (2D numpy array).
        """
        P = np.array(self.human_sequence)
        Q = np.array(self.simulated_sequence)

        min_len = min(len(P), len(Q))
        P = P[:min_len]
        Q = Q[:min_len]

        c = np.zeros((min_len, min_len))

        for i in range(min_len):
            for j in range(min_len):
                if euclidean(P[i], Q[j]) < threshold:
                    c[i, j] = 1

        return c

    def recurrence_rate(self, threshold=0.1):
        """
        Compute the recurrence rate between two sequences.

        Parameters:
        - threshold: Distance threshold for defining recurrence (float).

        Returns:
        - rr: Recurrence rate (float).
        """
        recurrence_matrix = self._compute_recurrence_matrix(threshold)
        min_len = len(recurrence_matrix)
        recurrence_points = np.sum(recurrence_matrix) - min_len  # Exclude diagonal
        total_possible = min_len * (min_len - 1)
        rr = (recurrence_points / total_possible) * 100
        return rr
    
    def determinism(self, threshold=0.1, l_min=2):
        """
        Compute determinism of the recurrence plot.

        Parameters:
        - threshold: Distance threshold for defining recurrence (float).
        - l_min: Minimum length of diagonal lines considered (int).

        Returns:
        - det: Determinism (float).
        """
        recurrence_matrix = self._compute_recurrence_matrix(threshold)
        diagonal_counts = self._count_diagonals(recurrence_matrix, l_min)
        total_recursions = np.sum(recurrence_matrix) - len(recurrence_matrix)  # Exclude diagonal
        det = (diagonal_counts / total_recursions) * 100 if total_recursions > 0 else 0
        return det
    
    def laminarity(self, threshold=0.1, v_min=2):
        """
        Compute laminarity of the recurrence plot.

        Parameters:
        - threshold: Distance threshold for defining recurrence (float).
        - v_min: Minimum length of vertical and horizontal lines considered (int).

        Returns:
        - laminarity: Laminarity (float).
        """
        recurrence_matrix = self._compute_recurrence_matrix(threshold)
        s = len(recurrence_matrix)
        R = np.triu(recurrence_matrix, 1).sum() + np.finfo(float).eps

        HL = 0
        HV = 0

        # Horizontal lines
        for i in range(s):
            row = recurrence_matrix[i, :]
            ones_runs = self._get_runs_of_ones(row)
            HL += sum(run_length for run_length in ones_runs if run_length >= v_min)

        # Vertical lines
        for j in range(s):
            column = recurrence_matrix[:, j]
            ones_runs = self._get_runs_of_ones(column)
            HV += sum(run_length for run_length in ones_runs if run_length >= v_min)

        laminarity = 100 * ((HL + HV) / (2 * R))
        return laminarity

    def center_of_recurrence_mass(self, threshold=0.1):
        """
        Compute Center of Recurrence Mass (CORM) of the recurrence plot.

        Parameters:
        - threshold: Distance threshold for defining recurrence (float).

        Returns:
        - corm: Center of Recurrence Mass (float).
        """
        recurrence_matrix = self._compute_recurrence_matrix(threshold)
        s = len(recurrence_matrix)
        R = np.triu(recurrence_matrix, 1).sum() + np.finfo(float).eps
        counter = 0
        for i in range(s - 1):
            for j in range(i + 1, s):
                counter += (j - i) * recurrence_matrix[i, j]
        corm = 100 * (counter / ((s - 1) * R))
        return corm
    
    def _compute_recurrence_matrix(self, threshold):
        P = np.array(self.human_sequence)
        Q = np.array(self.simulated_sequence)
        min_len = min(len(P), len(Q))
        P = P[:min_len]
        Q = Q[:min_len]

        recurrence_matrix = np.zeros((min_len, min_len))
        for i in range(min_len):
            for j in range(min_len):
                if np.linalg.norm(P[i] - Q[j]) < threshold:
                    recurrence_matrix[i, j] = 1
        return recurrence_matrix
    
    def _count_diagonals(self, recurrence_matrix, l_min):
        min_len = len(recurrence_matrix)
        diagonal_counts = 0
        for k in range(-min_len + 1, min_len):
            diag = recurrence_matrix.diagonal(k)
            ones_runs = self._get_runs_of_ones(diag)
            diagonal_counts += sum(run_length for run_length in ones_runs if run_length >= l_min)
        return diagonal_counts

    def _count_verticals(self, recurrence_matrix, v_min):
        s = recurrence_matrix.shape[0]
        vertical_counts = 0
        for j in range(s):
            column = recurrence_matrix[:, j]
            ones_runs = self._get_runs_of_ones(column)
            vertical_counts += sum(run_length for run_length in ones_runs if run_length >= v_min)
        return vertical_counts

    def _get_runs_of_ones(self, array):
        runs = []
        run_length = 0
        for val in array:
            if val == 1:
                run_length += 1
            else:
                if run_length > 0:
                    runs.append(run_length)
                    run_length = 0
        if run_length > 0:
            runs.append(run_length)
        return runs
    
def calc_scanpath_similarity(human_data_path, sim_data_path, folder_dir, num_fixations=None):
     
    with open(human_data_path, 'r') as f:
        human_data = json.load(f)

    with open(sim_data_path, 'r') as f:
        simulated_data = json.load(f)

    # Organize human data by (stimulus_index, time_constraint)
    human_data_by_stimulus_time = {}
    for trial in human_data:
        stimulus_index = trial['stimulus_index']
        time_constraint = trial['time_constraint']
        key = (stimulus_index, time_constraint)
        if key not in human_data_by_stimulus_time:
            human_data_by_stimulus_time[key] = []
            print(f"Human data for stimulus {key[0]} with time constraint {key[1]} found.")
        human_data_by_stimulus_time[key].append(trial)
    
    # Organize simulated data by (stimulus_index, time_constraint)
    sim_data_by_stimulus_time = {}
    for trial in simulated_data:
        stimulus_index = trial['stimulus_index']
        time_constraint = trial['time_constraint']
        key = (stimulus_index, time_constraint)
        if key not in sim_data_by_stimulus_time:
            sim_data_by_stimulus_time[key] = []
            print(f"Simulated data for stimulus {key[0]} with time constraint {key[1]} found.")
        sim_data_by_stimulus_time[key].append(trial)

    # For each stimulus_index, compute metrics
    results = []

    # Iterate over all keys (stimulus_index, time_constraint) present in human data
    for key in human_data_by_stimulus_time:
        if key not in sim_data_by_stimulus_time:
            print(f"No simulated data for stimulus {key[0]} with time constraint {key[1]}")
            continue
        
        # Print out the progress
        print(f"****************************************************************************************")
        print(f"Processing stimulus {key[0]} with time constraint {key[1]}...\n")

        human_trials = human_data_by_stimulus_time[key]
        sim_trials = sim_data_by_stimulus_time[key]

        # For each pair of human and simulated trials, compute metrics
        # Compare each human trial with each simulated trial
        for human_trial, sim_trial in product(human_trials, sim_trials):
            # Extract sequences and coordinates
            human_fixations = human_trial['fixation_data']
            sim_fixations = sim_trial['fixation_data']

            # Extract word index sequences
            human_word_indices = [fix['word_index'] for fix in human_fixations if fix['word_index'] != -1]
            sim_word_indices = [fix['word_index'] for fix in sim_fixations if fix['word_index'] != -1]

            # Extract coordinate sequences, handling missing data
            human_coords_sequence = []
            for fix in human_fixations:
                if fix['word_index'] != -1 and fix.get('fix_x') is not None and fix.get('fix_y') is not None:
                    human_coords_sequence.append((fix['fix_x'], fix['fix_y']))

            sim_coords_sequence = []
            for fix in sim_fixations:
                if fix['word_index'] != -1 and fix.get('fix_x') is not None and fix.get('fix_y') is not None:
                    sim_coords_sequence.append((fix['fix_x'], fix['fix_y']))

            # Extract normalized coordinate sequences, handling missing data
            human_norm_coords_sequence = []
            for fix in human_fixations:
                if fix['word_index'] != -1 and fix.get('norm_fix_x') is not None and fix.get('norm_fix_y') is not None:
                    human_norm_coords_sequence.append((fix['norm_fix_x'], fix['norm_fix_y']))

            sim_norm_coords_sequence = []
            for fix in sim_fixations:
                if fix['word_index'] != -1 and fix.get('norm_fix_x') is not None and fix.get('norm_fix_y') is not None:
                    sim_norm_coords_sequence.append((fix['norm_fix_x'], fix['norm_fix_y']))

            # Limit sequences to the specified number of fixations
            if num_fixations is not None:
                human_word_indices = human_word_indices[:num_fixations]
                sim_word_indices = sim_word_indices[:num_fixations]
                human_coords_sequence = human_coords_sequence[:num_fixations]
                sim_coords_sequence = sim_coords_sequence[:num_fixations]
                human_norm_coords_sequence = human_norm_coords_sequence[:num_fixations]
                sim_norm_coords_sequence = sim_norm_coords_sequence[:num_fixations]

            # Initialize result dict with basic info  
            result_base = {
                'stimulus_index': key[0],
                'time_constraint': key[1],
                'num_fixations': num_fixations if num_fixations is not None else 'All',
                'human_participant_id': human_trial.get('participant_index', None),
                'simulated_episode_index': sim_trial.get('episode_index', None),
                'simulated_trial_index': sim_trial.get('sim_trial_index', None),
            }

            # Compute metrics using word indices
            result = result_base.copy()
            if human_word_indices and sim_word_indices:
                # Initialize ScanpathMetrics
                metrics_word = ScanpathMetrics(
                    human_sequence=human_word_indices,
                    simulated_sequence=sim_word_indices,
                    mode=const.MODE_WORD_INDEX
                )
                # Compute metrics
                lev_dist = metrics_word.levenshtein_distance(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                nld = metrics_word.normalized_levenshtein_distance(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                dtw_dist = metrics_word.dynamic_time_warping()
                lcs_length = metrics_word.longest_common_subsequence(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                frechet_dist = metrics_word.frechet_distance()
                hausdorff_dist = metrics_word.hausdorff_distance()
                mann_distance = metrics_word.mannan_distance()
                eyeanalysis_distance = metrics_word.eyenalysis_distance()
                # determinism = metrics_word.determinism()
                # laminarity = metrics_word.laminarity()
                scanmatch_score = metrics_word.scan_match(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                # tde_distance = metrics_word.time_delay_embedding(k=2)
                # recurrence_rate = metrics_word.recurrence_rate()
                # Store the results
                result.update({
                    'mode': 'word_index',
                    const.LEV: lev_dist,
                    const.NLD: nld,
                    const.DTW: dtw_dist,
                    const.LCS: lcs_length,
                    const.FRECT: frechet_dist,
                    const.HAUSDF: hausdorff_dist,
                    const.ScanMatchScore: scanmatch_score,
                    # const.TDE: tde_distance,
                    # const.RR: recurrence_rate, 
                    const.MD: mann_distance,
                    const.EYEALS: eyeanalysis_distance,
                    # const.DET: determinism,
                    # const.LAM: laminarity
                })
            else:
                # Can't compute metrics using word indices
                result.update({
                    'mode': 'word_index',
                    const.LEV: 0,
                    const.NLD: 0,
                    const.DTW: 0,
                    const.LCS: 0,
                    const.FRECT: 0,
                    const.HAUSDF: 0,
                    const.ScanMatchScore: 0,
                    const.TDE: 0,
                    const.RR: 0, 
                    const.MD: 0,
                    const.EYEALS: 0,
                    const.DET: 0,
                    const.LAM: 0
                })
            results.append(result.copy())

            # Compute metrics using coordinates
            result = result_base.copy()
            if human_coords_sequence and sim_coords_sequence:
                # Initialize ScanpathMetrics
                metrics_coords = ScanpathMetrics(
                    human_sequence=human_coords_sequence,
                    simulated_sequence=sim_coords_sequence,
                    mode=const.MODE_COORD
                )
                # Compute metrics
                lev_dist = metrics_coords.levenshtein_distance(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                nld = metrics_coords.normalized_levenshtein_distance(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                dtw_dist = metrics_coords.dynamic_time_warping()
                lcs_length = metrics_coords.longest_common_subsequence(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                frechet_dist = metrics_coords.frechet_distance()
                hausdorff_dist = metrics_coords.hausdorff_distance()
                mann_distance = metrics_coords.mannan_distance()
                eyeanalysis_distance = metrics_coords.eyenalysis_distance()
                # determinism = metrics_coords.determinism()
                # laminarity = metrics_coords.laminarity()
                scanmatch_score = metrics_coords.scan_match(height=const.SCREEN_RESOLUTION_HEIGHT_PX, width=const.SCREEN_RESOLUTION_WIDTH_PX)
                # tde_distance = metrics_coords.time_delay_embedding(k=2)
                # recurrence_rate = metrics_coords.recurrence_rate()
                # Store the results
                result.update({
                    'mode': 'coords',
                    const.LEV: lev_dist,
                    const.NLD: nld,
                    const.DTW: dtw_dist,
                    const.LCS: lcs_length,
                    const.FRECT: frechet_dist,
                    const.HAUSDF: hausdorff_dist,
                    const.ScanMatchScore: scanmatch_score,
                    # const.TDE: tde_distance,
                    # const.RR: recurrence_rate, 
                    const.MD: mann_distance,
                    const.EYEALS: eyeanalysis_distance,
                    # const.DET: determinism,
                    # const.LAM: laminarity
                })
            else:
                # Can't compute metrics using coordinates
                result.update({
                    'mode': 'coords',
                    const.LEV: 0,
                    const.NLD: 0,
                    const.DTW: 0,
                    const.LCS: 0,
                    const.FRECT: 0,
                    const.HAUSDF: 0,
                    const.ScanMatchScore: 0,
                    # const.TDE: 0,
                    # const.RR: 0, 
                    const.MD: 0,
                    const.EYEALS: 0,
                    # const.DET: 0,
                    # const.LAM: 0
                })
            results.append(result.copy())

            # Compute metrics using normalized coordinates
            result = result_base.copy()
            if human_norm_coords_sequence and sim_norm_coords_sequence:
                # Initialize ScanpathMetrics
                metrics_norm_coords = ScanpathMetrics(
                    human_sequence=human_norm_coords_sequence,
                    simulated_sequence=sim_norm_coords_sequence,
                    mode=const.MODE_COORD
                )
                # Compute metrics
                lev_dist = metrics_norm_coords.levenshtein_distance(height=1, width=1)
                nld = metrics_norm_coords.normalized_levenshtein_distance(height=1, width=1)
                dtw_dist = metrics_norm_coords.dynamic_time_warping()
                lcs_length = metrics_norm_coords.longest_common_subsequence(height=1, width=1)
                frechet_dist = metrics_norm_coords.frechet_distance()
                hausdorff_dist = metrics_norm_coords.hausdorff_distance()
                mann_distance = metrics_norm_coords.mannan_distance()
                eyeanalysis_distance = metrics_norm_coords.eyenalysis_distance()
                # determinism = metrics_norm_coords.determinism()
                # laminarity = metrics_norm_coords.laminarity()
                scanmatch_score = metrics_norm_coords.scan_match(height=1, width=1)
                # tde_distance = metrics_norm_coords.time_delay_embedding(k=2)
                # recurrence_rate = metrics_norm_coords.recurrence_rate()
                # Store the results
                result.update({
                    'mode': 'norm_coords',
                    const.LEV: lev_dist,
                    const.NLD: nld,
                    const.DTW: dtw_dist,
                    const.LCS: lcs_length,
                    const.FRECT: frechet_dist,
                    const.HAUSDF: hausdorff_dist,
                    const.ScanMatchScore: scanmatch_score,
                    # const.TDE: tde_distance,
                    # const.RR: recurrence_rate, 
                    const.MD: mann_distance,
                    const.EYEALS: eyeanalysis_distance,
                    # const.DET: determinism,
                    # const.LAM: laminarity
                })
            else:
                # Can't compute metrics using normalized coordinates
                result.update({
                    'mode': 'norm_coords',
                    const.LEV: 0,
                    const.NLD: 0,
                    const.DTW: 0,
                    const.LCS: 0,
                    const.FRECT: 0,
                    const.HAUSDF: 0,
                    const.ScanMatchScore: 0,
                    # const.TDE: 0,
                    # const.RR: 0, 
                    const.MD: 0,
                    const.EYEALS: 0,
                    # const.DET: 0,
                    # const.LAM: 0
                })
            results.append(result.copy())

    # After the loops, save the results
    # Save data to files in a folder
    # Save the metadata of these comparisons into a JSON file
    metadata = {
        'human_data_path': human_data_path,
        'sim_data_path': sim_data_path,
        'results_dir': folder_dir,
        'num_fixations': num_fixations,
    }
    with open(os.path.join(folder_dir, "scanpath_sim_and_human_data_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save the results to a csv file
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(folder_dir, "scanpath_metrics_results.csv"), index=False)

    # Now, group and summarize metrics
    # Function to group and summarize metrics
    def summarize_metrics(df, metrics):
        grouped = df.groupby(['time_constraint', 'num_fixations', 'mode'])[metrics].agg(['mean', 'std'])
        # Flatten MultiIndex columns
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        return grouped

    # List of metric columns
    metric_columns = [
        const.LEV, const.NLD, const.DTW, const.LCS, const.FRECT, const.HAUSDF, const.ScanMatchScore,
        #   const.TDE, const.RR, const.MD, 
          const.EYEALS, 
        #   const.DET, const.LAM
    ]

    summary = summarize_metrics(df_results, metric_columns)

    # Save the summary to a CSV file
    csv_path = os.path.join(folder_dir, "scanpath_metrics_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Summary statistics saved to {csv_path}")


class AggregatedFixationMetrics:       
    def __init__(self, fixation_data, time_constraint, model_name=None, word_metadata=None):
        """
        Initialize the FixationAggregatedMetrics class.

        Parameters:
        - fixation_data: List of fixation dictionaries from the JSON data.
        - time_constraint: Duration of the reading trial in seconds.
        - word_metadata: Dictionary mapping word_index to word information (e.g., center coordinates).
        """
        self.fixation_data = fixation_data
        self.time_constraint = time_constraint
        self.word_metadata = word_metadata
        self.model_name = model_name

        # Convert fixation data to DataFrame for easier processing
        self.df_fixations = pd.DataFrame(fixation_data)
        # Ensure that 'fix_x' and 'fix_y' columns are present
        if 'fix_x' not in self.df_fixations.columns or 'fix_y' not in self.df_fixations.columns:
            raise ValueError("Fixation data must contain 'fix_x' and 'fix_y'.")

        # Get word indexes
        self.df_fixations['word_index'] = self.df_fixations['word_index']

        # # Calculate saccade lengths between fixations
        # self.df_fixations['prev_fix_x'] = self.df_fixations['fix_x'].shift(1)
        # self.df_fixations['prev_fix_y'] = self.df_fixations['fix_y'].shift(1)
        # self.df_fixations['saccade_length'] = np.sqrt(
        #     (self.df_fixations['fix_x'] - self.df_fixations['prev_fix_x']) ** 2 +
        #     (self.df_fixations['fix_y'] - self.df_fixations['prev_fix_y']) ** 2
        # )

        # Assign 'fix_x' and 'fix_y' if missing, using word centers
        if 'fix_x' not in self.df_fixations.columns or self.df_fixations['fix_x'].isnull().all():
            if self.word_metadata is not None:
                # Assign 'fix_x' and 'fix_y' from word centers
                fix_x_list = []
                fix_y_list = []
                for idx, row in self.df_fixations.iterrows():
                    word_idx = row['word_index']
                    if word_idx != -1 and word_idx in self.word_metadata:
                        word_info = self.word_metadata[word_idx]
                        word_center_x = word_info['word_center'][0]     # TODO there is not a key called word_center in the word metadata
                        word_center_y = word_info['word_center'][1]
                        fix_x_list.append(word_center_x)
                        fix_y_list.append(word_center_y)
                    else:
                        fix_x_list.append(np.nan)
                        fix_y_list.append(np.nan)
                self.df_fixations['fix_x'] = fix_x_list
                self.df_fixations['fix_y'] = fix_y_list
            else:
                # Cannot compute 'fix_x' and 'fix_y' without word metadata
                self.df_fixations['fix_x'] = np.nan
                self.df_fixations['fix_y'] = np.nan
        
        # Calculate saccade lengths if possible
        if 'fix_x' in self.df_fixations.columns and 'fix_y' in self.df_fixations.columns and not self.df_fixations['fix_x'].isnull().all():
            # Calculate saccade lengths between fixations
            self.df_fixations['prev_fix_x'] = self.df_fixations['fix_x'].shift(1)
            self.df_fixations['prev_fix_y'] = self.df_fixations['fix_y'].shift(1)
            self.df_fixations['saccade_length'] = np.sqrt(
                (self.df_fixations['fix_x'] - self.df_fixations['prev_fix_x']) ** 2 +
                (self.df_fixations['fix_y'] - self.df_fixations['prev_fix_y']) ** 2
            )
        else:
            # Cannot compute saccade lengths
            self.df_fixations['saccade_length'] = np.nan

    def calculate_saccade_lengths(self):
        """
        Calculate saccade lengths between consecutive fixations.

        Returns:
        - saccade_lengths: List of saccade lengths in pixels.
        """
        saccade_lengths = self.df_fixations['saccade_length'].dropna().tolist()
        return saccade_lengths

    def calculate_average_saccade_length(self):
        """
        Calculate the average saccade length.

        Returns:
        - avg_saccade_length: Average saccade length in pixels.
        """
        # saccade_lengths = self.calculate_saccade_lengths()
        # if saccade_lengths:
        #     avg_saccade_length = np.mean(saccade_lengths)
        # else:
        #     avg_saccade_length = 0
        # return avg_saccade_length
        saccade_lengths = self.calculate_saccade_lengths()
        if saccade_lengths:
            avg_saccade_length = np.nanmean(saccade_lengths)
        else:
            avg_saccade_length = np.nan  # Return NaN if saccade lengths cannot be computed
        return avg_saccade_length
    
    def calculate_number_of_fixations(self):
        """
        Calculate the number of fixations.

        Returns:
        - num_fixations: Number of fixations.
        """
        num_fixations = len(self.df_fixations)
        return num_fixations

    def calculate_word_skip_percentage(self):
        """
        Calculate the word skip percentage based on fixation data and word bounding boxes in metadata.

        Returns:
        - word_skip_percentage_by_reading_progress: Skip percentage by reading progress (word-based)
        - word_skip_percentage_by_saccades: Skip percentage by total number of saccades
        """
        df_fixations = self.df_fixations

        # Using word indices from df_fixations
        fixated_word_indices = df_fixations['word_index'].tolist()

        # Proceed with computation using word indices
        total_num_skip_saccades = 0
        for i in range(len(fixated_word_indices) - 1):
            current_word_idx = fixated_word_indices[i]
            next_word_idx = fixated_word_indices[i + 1]
            if current_word_idx == -1 or next_word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            skipped = next_word_idx - current_word_idx - 1
            if skipped > 0:
                total_num_skip_saccades += 1

        last_read_word_index = max([idx for idx in fixated_word_indices if idx != -1], default=-1)
        total_words_to_last_read_word = last_read_word_index + 1  # Assuming word indices start from 0

        total_saccades = len(fixated_word_indices) - 1

        # Calculate skip percentage by reading progress (word-based)
        if total_words_to_last_read_word > 0:
            word_skip_percentage_by_reading_progress = (
                total_num_skip_saccades / total_words_to_last_read_word
            ) * 100
        else:
            word_skip_percentage_by_reading_progress = 0  # Avoid division by zero

        # Calculate skip percentage by total number of saccades
        if total_saccades > 0:
            word_skip_percentage_by_saccades = (
                total_num_skip_saccades / total_saccades
            ) * 100
        else:
            word_skip_percentage_by_saccades = 0  # Avoid division by zero

        return word_skip_percentage_by_reading_progress, word_skip_percentage_by_saccades

    def calculate_revisit_percentage(self):
        """
        Calculate the regression/revisit percentage based on fixation data and word bounding boxes in metadata.

        Returns:
        - revisit_percentage_by_reading_progress: Revisit percentage by reading progress (word-based)
        - revisit_percentage_by_fixations: Revisit percentage by total number of fixations
        """
        df_fixations = self.df_fixations
        fixated_word_indices = df_fixations['word_index'].tolist()

        last_read_word_index = -1
        num_revisit_words = 0

        for word_idx in fixated_word_indices:
            if word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            if word_idx < last_read_word_index:
                num_revisit_words += 1
            else:
                last_read_word_index = word_idx

        total_words_to_last_read_word = last_read_word_index + 1
        total_fixations = len([idx for idx in fixated_word_indices if idx != -1])

        if total_words_to_last_read_word > 0:
            revisit_percentage_by_reading_progress = (
                num_revisit_words / total_words_to_last_read_word
            ) * 100
        else:
            revisit_percentage_by_reading_progress = 0

        if total_fixations > 0:
            revisit_percentage_by_fixations = (
                num_revisit_words / total_fixations
            ) * 100
        else:
            revisit_percentage_by_fixations = 0

        return revisit_percentage_by_reading_progress, revisit_percentage_by_fixations

    # def calculate_reading_speed(self):
    #     """
    #     Calculate the reading speed based on the total number of words read over the time constraint.

    #     Returns:
    #     - reading_speed: Number of words read per minute.
    #     """
    #     df_fixations = self.df_fixations
    #     fixated_word_indices = df_fixations['word_index'].tolist()

    #     total_words_read = 0
    #     previous_word_idx = None

    #     for word_idx in fixated_word_indices:
    #         if word_idx == -1:
    #             continue  # Skip if word index is -1 (not mapped)
    #         if word_idx != previous_word_idx:
    #             # New word read
    #             total_words_read += 1
    #         previous_word_idx = word_idx

    #     # Calculate reading speed (words per minute)
    #     if self.time_constraint > 0:
    #         reading_speed = (total_words_read / self.time_constraint) * 60  # words per minute
    #     else:
    #         reading_speed = 0  # Avoid division by zero

    #     return reading_speed

    def calculate_reading_speed(self, use_total_fixation_duration=False):
        """
        Calculate the reading speed based on the total number of words read over the time constraint or total fixation duration.

        Parameters:
        - use_total_fixation_duration: If True, use total fixation duration instead of time_constraint.

        Returns:
        - reading_speed: Number of words read per minute.
        """
        df_fixations = self.df_fixations
        fixated_word_indices = df_fixations['word_index'].tolist()

        total_words_read = 0
        previous_word_idx = None

        for word_idx in fixated_word_indices:
            if word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            if word_idx != previous_word_idx:
                # New word read
                total_words_read += 1
            previous_word_idx = word_idx

        if use_total_fixation_duration:
            total_time = df_fixations['fix_duration'].sum() / 1000.0  # Assuming fix_duration is in milliseconds
        else:
            total_time = self.time_constraint

        # Calculate reading speed (words per minute)
        if total_time > 0:
            reading_speed = (total_words_read / total_time) * 60  # words per minute
        else:
            reading_speed = 0  # Avoid division by zero

        return reading_speed


    # def compute_all_metrics(self):
    #     """
    #     Compute all metrics and return as a dictionary.

    #     Returns:
    #     - metrics: Dictionary containing all computed metrics.
    #     """
    #     word_skip_reading_progress, word_skip_saccades = self.calculate_word_skip_percentage()
    #     revisit_reading_progress, revisit_fixations = self.calculate_revisit_percentage()

    #     metrics = {
    #         const.NUM_FIXATIONS: self.calculate_number_of_fixations(),  # Added line
    #         const.AVG_SACCADE_LENGTH_PX: self.calculate_average_saccade_length(),
    #         const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2: word_skip_saccades,
    #         const.REVISIT_PERCENTAGE_BY_SACCADES_V2: revisit_fixations,
    #         const.READING_SPEED: self.calculate_reading_speed()
    #     }
    #     return metrics

    def compute_all_metrics(self):
        """
        Compute all metrics and return as a dictionary.

        Returns:
        - metrics: Dictionary containing all computed metrics.
        """
        word_skip_reading_progress, word_skip_saccades = self.calculate_word_skip_percentage()
        revisit_reading_progress, revisit_fixations = self.calculate_revisit_percentage()

        avg_saccade_length = self.calculate_average_saccade_length()
        if np.isnan(avg_saccade_length):
            avg_saccade_length = 0  # Handle NaN values

        # Decide whether to use total fixation duration
        if self.model_name in [const.EZREADER, const.SWIFT, const.SCANDL, const.EYETTENTION]: 
            use_total_fixation_duration = True
        else:
            use_total_fixation_duration = False

        reading_speed = self.calculate_reading_speed(use_total_fixation_duration=use_total_fixation_duration)

        metrics = {
            const.NUM_FIXATIONS: self.calculate_number_of_fixations(),
            const.AVG_SACCADE_LENGTH_PX: avg_saccade_length,
            const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2: word_skip_saccades,
            const.REVISIT_PERCENTAGE_BY_SACCADES_V2: revisit_fixations,
            const.READING_SPEED: reading_speed
        }
        return metrics


def calc_aggregated_fixation_metrics(human_data_path, sim_data_path, folder_dir, word_metadata_path):
    # Load data
    with open(human_data_path, 'r') as f:
        human_data = json.load(f)

    with open(sim_data_path, 'r') as f:
        sim_data = json.load(f)

    # Load word metadata
    with open(word_metadata_path, 'r') as f:
        bbox_metadata = json.load(f)
    
    # Prepare word metadata mapping from stimulus_index to word_metadata
    word_metadata_by_stimulus = {}
    for image_meta in bbox_metadata['images']:
        stimulus_index = image_meta['image index']
        word_metadata = {}
        for idx, word_info in enumerate(image_meta['words metadata']):
            word_index = idx  # Use the index in the list as the word_index
            word_center_x = (word_info['word_bbox'][0] + word_info['word_bbox'][2]) / 2
            word_center_y = (word_info['word_bbox'][1] + word_info['word_bbox'][3]) / 2
            word_metadata[word_index] = {
                'word_center': (word_center_x, word_center_y),
                'word_bbox': word_info['word_bbox']
            }
        word_metadata_by_stimulus[stimulus_index] = word_metadata


    # ================================= Process human data =================================
    human_results = []
    for trial in human_data:
        participant_id = trial.get('participant_index', 'Unknown')
        stimulus_index = trial['stimulus_index']
        time_constraint = trial['time_constraint']
        fixation_data = trial['fixation_data']
        trial_condition = time_constraint  # Trial condition is time constraint

        # Initialize metrics calculator
        metrics_calculator = AggregatedFixationMetrics(
            fixation_data=fixation_data,
            time_constraint=time_constraint
        )

        # Compute metrics
        metrics = metrics_calculator.compute_all_metrics()

        # Prepare result
        result = {
            const.PID: participant_id,
            const.STIM_ID: stimulus_index,
            const.TRIAL_COND: trial_condition,
            const.AVG_SACCADE_LENGTH_PX: metrics[const.AVG_SACCADE_LENGTH_PX],
            const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2: metrics[const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2],
            const.REVISIT_PERCENTAGE_BY_SACCADES_V2: metrics[const.REVISIT_PERCENTAGE_BY_SACCADES_V2],
            const.READING_SPEED: metrics[const.READING_SPEED],
            const.NUM_FIXATIONS: metrics[const.NUM_FIXATIONS]
        }
        human_results.append(result)

    # ================================= Process simulation data =================================
    sim_results = []
    for trial in sim_data:
        episode_id = trial.get('episode_index', 'Unknown')
        stimulus_index = trial['stimulus_index']
        time_constraint = trial['time_constraint']
        fixation_data = trial['fixation_data']
        baseline_model_name = trial.get('baseline_model_name', None)
        trial_condition = time_constraint  # Trial condition is time constraint

        # Get word metadata for this stimulus
        word_metadata = word_metadata_by_stimulus.get(stimulus_index, None)

        # # Initialize metrics calculator
        # metrics_calculator = AggregatedFixationMetrics(
        #     fixation_data=fixation_data,
        #     time_constraint=time_constraint
        # )
        # Initialize metrics calculator
        metrics_calculator = AggregatedFixationMetrics(
            fixation_data=fixation_data,
            time_constraint=time_constraint,
            model_name=baseline_model_name,
            word_metadata=word_metadata
        )

        # Compute metrics
        metrics = metrics_calculator.compute_all_metrics()

        # Prepare result
        result = {
            const.EPSID: episode_id,
            const.STIM_ID: stimulus_index,
            const.TRIAL_COND: trial_condition,
            const.AVG_SACCADE_LENGTH_PX: metrics[const.AVG_SACCADE_LENGTH_PX],
            const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2: metrics[const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2],
            const.REVISIT_PERCENTAGE_BY_SACCADES_V2: metrics[const.REVISIT_PERCENTAGE_BY_SACCADES_V2],
            const.READING_SPEED: metrics[const.READING_SPEED],
            const.NUM_FIXATIONS: metrics[const.NUM_FIXATIONS]
        }
        sim_results.append(result)

    # Save results to CSV files
    human_df = pd.DataFrame(human_results)
    sim_df = pd.DataFrame(sim_results)

    human_csv_path = os.path.join(folder_dir, 'human_aggregated_fixation_metrics.csv')
    sim_csv_path = os.path.join(folder_dir, 'simulation_aggregated_fixation_metrics.csv')

    human_df.to_csv(human_csv_path, index=False)
    sim_df.to_csv(sim_csv_path, index=False)

    # Save metadata
    metadata = {
        'human_data_path': human_data_path,
        'sim_data_path': sim_data_path,
        'results_dir': folder_dir,
    }
    with open(os.path.join(folder_dir, "aggregated_fixation_metrics_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Generate summary files
    def generate_summary(df, group_by_column, summary_type):
        metrics = [const.AVG_SACCADE_LENGTH_PX, const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2, const.REVISIT_PERCENTAGE_BY_SACCADES_V2, const.READING_SPEED, const.NUM_FIXATIONS]
        summary = df.groupby(group_by_column)[metrics].agg(['mean', 'std']).reset_index()
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
        summary['Type'] = summary_type
        return summary

    human_summary = generate_summary(human_df, 'Trial Condition', 'Human')
    sim_summary = generate_summary(sim_df, 'Trial Condition', 'Simulation')

    combined_summary = pd.concat([human_summary, sim_summary], ignore_index=True)

    summary_csv_path = os.path.join(folder_dir, 'aggregated_fixation_metrics_summary.csv')
    combined_summary.to_csv(summary_csv_path, index=False)

    print(f"Results saved to {folder_dir}")


class MCQFreeRecallProcessor:

    def __init__(self, reference_answers):
        """
        Initialize the processor with reference answers for free recall scoring.

        Parameters:
        - reference_answers: A dictionary mapping stimulus_index to reference text.
        """
        self.reference_answers = reference_answers

    def compute_mcq_accuracy(self, mcq_logs):
        """
        Compute the MCQ accuracy for a given list of MCQ logs.

        Parameters:
        - mcq_logs: A list of dictionaries containing MCQ questions and answers.

        Returns:
        - accuracy: The proportion of correct answers (float between 0 and 1).
        """
        total_questions = len(mcq_logs)
        if total_questions == 0:
            return 0.0
        correct_answers = sum(
            1 for mcq in mcq_logs
            if mcq.get('answer', '').strip().upper() == mcq.get('correct_answer', '').strip().upper()
        )
        accuracy = correct_answers / total_questions
        return accuracy

    def compute_free_recall_score(self, agent_answer, reference_answer):
        """
        Compute the similarity score between the participant's free recall answer and the reference answer.

        Parameters:
        - participant_answer: The participant's free recall response (string).
        - reference_answer: The reference text to compare against (string).

        Returns:
        - cosine_sim: The cosine similarity score (float between 0 and 1).
        """
        # Preprocess the texts
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
            return text.strip()

        agent_text = preprocess(agent_answer)
        reference_text = preprocess(reference_answer)

        # Use CountVectorizer to create term-frequency vectors
        vectorizer = CountVectorizer().fit([agent_text, reference_text])
        vectors = vectorizer.transform([agent_text, reference_text])

        # Compute cosine similarity
        cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return cosine_sim
    
    def process_episode(self, episode):
        """
        Process a single episode to compute MCQ accuracy and free recall score.

        Parameters:
        - episode: A dictionary containing the episode data.

        Returns:
        - result: A dictionary containing the computed metrics.
        """
        episodic_info = episode.get('episodic_info', {})
        stimulus_info = episodic_info.get('stimulus', {})
        stimulus_index = stimulus_info.get('stimulus_index')
        episode_index = episodic_info.get('episode_index', None)
        
        # Extract time_constraint from the correct location
        # For simulation data, it might be under episodic_info['task']['time_constraint']
        time_constraint = episode.get('time_constraint', None)
        if time_constraint is None:
            task_info = episodic_info.get('task', {})
            time_constraint = task_info.get('time_constraint', None)

        mcq_logs = episodic_info.get('mcq_logs', [])
        mcq_accuracy = self.compute_mcq_accuracy(mcq_logs)

        agent_answer = episodic_info.get('free_recall_answer', '')
        reference_answer = stimulus_info.get('words_in_section', '')

        if agent_answer and reference_answer:
            free_recall_score = self.compute_free_recall_score(agent_answer, reference_answer)
        else:
            free_recall_score = 0.0  # Assign zero if either answer is missing

        result = {
            'episode_index': episode_index,
            'stimulus_index': stimulus_index,
            'time_constraint': time_constraint,
            'MCQ Accuracy': mcq_accuracy,
            'Free Recall Score': free_recall_score
        }
        return result
    
    def process_all_episodes(self, data):
        """
        Process all episodes in the data to compute free recall scores.

        Parameters:
        - data: A list of episodes (each is a dictionary).

        Returns:
        - results: A list of dictionaries containing computed metrics for each episode.
        """
        results = []
        for episode in data:
            result = self.process_episode(episode)
            results.append(result)
        return results


def calc_comprehension_metrics(sim_json_data_path, human_json_data_path, folder_dir, sim_output_csv='simulation_comprehension_results.csv', human_output_csv='human_comprehension_results.csv', 
                               report_human_baseline_results=True):
    """
    Process the simulation JSON data and the human CSV data to compute and summarize MCQ accuracy and free recall scores.

    Parameters:
    - sim_json_data_path: Path to the simulation JSON data file.
    - human_json_data_path: Path to the human JSON data file (for reference answers).
    - folder_dir: Directory to save the output CSV files.
    - sim_output_csv: The filename for the simulation output CSV file.
    - human_output_csv: The filename for the human output CSV file.

    Returns:
    - sim_df: Pandas DataFrame containing the simulation results.
    - human_df: Pandas DataFrame containing the human results.
    """

    # Process simulation data
    # ------------------------------------------------------------
    # Load simulation JSON data
    with open(sim_json_data_path, 'r') as f:
        sim_json_data = json.load(f)

    # Load or define reference answers for free recall
    # We'll assume the 'words_in_section' is the reference answer
    reference_sim_answers = {}
    for episode in sim_json_data:
        episodic_info = episode.get('episodic_info', {})
        stimulus_info = episodic_info.get('stimulus', {})
        stimulus_index = stimulus_info.get('stimulus_index')
        reference_text = stimulus_info.get('words_in_section', '')
        reference_sim_answers[stimulus_index] = reference_text

    sim_processor = MCQFreeRecallProcessor(reference_sim_answers)

    # Get the configuration for the simulation from one of the episodes
    param1_text_similarity_threshold = sim_json_data[0].get('episodic_info', {}).get('memory_retrieve_metadata', {}).get('text_similarity_threshold', None)
    param2_exploration_rate = sim_json_data[0].get('episodic_info', {}).get('memory_retrieve_metadata', {}).get('exploration_rate', None)
    
    sim_results = []
    for episode in sim_json_data:
        result = sim_processor.process_episode(episode)
        result['Type'] = 'Simulation'
        result['text_similarity_threshold'] = param1_text_similarity_threshold
        result['exploration_rate'] = param2_exploration_rate
        sim_results.append(result)

    # Convert simulation results to DataFrame
    sim_df = pd.DataFrame(sim_results)

    # Ensure that 'MCQ Accuracy' and 'Free Recall Score' are in percentages
    # If they are in decimal form (less than or equal to 1.0), multiply by 100
    if sim_df['MCQ Accuracy'].max() <= 1.0:
        sim_df['MCQ Accuracy'] = sim_df['MCQ Accuracy'] * 100
    if sim_df['Free Recall Score'].max() <= 1.0:
        sim_df['Free Recall Score'] = sim_df['Free Recall Score'] * 100

    # Save simulation results to CSV
    updated_unique_sim_output_csv = sim_output_csv.split('.')[0] + f"_text_similarity_threshold_{param1_text_similarity_threshold}_exploration_rate_{param2_exploration_rate}" + ".csv"
    sim_save_path = os.path.join(folder_dir, updated_unique_sim_output_csv)
    sim_df.to_csv(sim_save_path, index=False)
    print(f"Simulation results saved to {sim_save_path}")

    # Process human data
    # ------------------------------------------------------------
    if report_human_baseline_results:
        # Load human JSON data
        with open(human_json_data_path, 'r') as f:
            human_json_data = json.load(f)
        
        reference_human_answers = {}
        for episode in human_json_data:
            episodic_info = episode.get('episodic_info', {})
            stimulus_info = episodic_info.get('stimulus', {})
            stimulus_index = stimulus_info.get('stimulus_index')
            reference_text = stimulus_info.get('words_in_section', '')
            reference_human_answers[stimulus_index] = reference_text
        
        human_processor = MCQFreeRecallProcessor(reference_human_answers)

        human_results = []
        for episode in human_json_data:
            result = human_processor.process_episode(episode)
            result['Type'] = 'Human'
            human_results.append(result)
        
        # Convert human results to DataFrame
        human_df = pd.DataFrame(human_results)

        # Ensure that 'MCQ Accuracy' and 'Free Recall Score' are in percentages
        # If they are in decimal form (less than or equal to 1.0), multiply by 100
        if human_df['MCQ Accuracy'].max() <= 1.0:
            human_df['MCQ Accuracy'] = human_df['MCQ Accuracy'] * 100
        if human_df['Free Recall Score'].max() <= 1.0:
            human_df['Free Recall Score'] = human_df['Free Recall Score'] * 100

        # Add 'Type' column to human data
        human_df['Type'] = 'Human'
        human_df['text_similarity_threshold'] = None
        human_df['exploration_rate'] = None

        # Save human results to CSV (optional, since data is already loaded from CSV)
        human_save_path = os.path.join(folder_dir, human_output_csv)
        human_df.to_csv(human_save_path, index=False)
        print(f"Human results saved to {human_save_path}")
    else:
        human_df = None

    # Generate summary files
    def generate_summary(df, group_by_column, summary_type):
        metrics = ['MCQ Accuracy', 'Free Recall Score']
        summary = df.groupby(group_by_column)[metrics].agg(['mean', 'std']).reset_index()
        # Flatten MultiIndex columns
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns.values]
        summary['Type'] = summary_type
        summary['text_similarity_threshold'] = df['text_similarity_threshold'].iloc[0]
        summary['exploration_rate'] = df['exploration_rate'].iloc[0]
        return summary

    # Generate summaries
    sim_summary = generate_summary(sim_df, 'time_constraint', 'Simulation')
    # human_summary = generate_summary(human_df, 'time_constraint', 'Human')
    if report_human_baseline_results:
        human_summary = generate_summary(human_df, 'time_constraint', 'Human')
        # Combine summaries
        combined_summary = pd.concat([human_summary, sim_summary], ignore_index=True)
    else:
        combined_summary = sim_summary
    # # Combine summaries
    # combined_summary = pd.concat([human_summary, sim_summary], ignore_index=True)

    # Save summary to CSV
    summary_csv_path = os.path.join(folder_dir, f'comprehension_metrics_summary_text_similarity_threshold_{param1_text_similarity_threshold}_exploration_rate_{param2_exploration_rate}.csv')
    combined_summary.to_csv(summary_csv_path, index=False)
    print(f"Comprehension summary statistics saved to {summary_csv_path}")

    return sim_df, human_df


# Plotting ******************************************************************************************************************** 

class MetricsPlotter:
    def __init__(self, folder_dir, csv_filename_pattern, trial_condition_column='Trial Condition', type_column='Type'):
        """
        Initialize the MetricsPlotter class.

        Parameters:
        - folder_dir: Directory where the CSV file is located.
        - csv_filename: Name of the CSV file containing the metrics summary.
        - trial_condition_column: Column name for trial conditions (default is 'Trial Condition').
        - type_column: Column name for the data type (e.g., 'Human', 'Simulation') (default is 'Type').
        """
        self.folder_dir = folder_dir
        self.csv_filename_pattern = csv_filename_pattern
        self.trial_condition_column = trial_condition_column
        self.type_column = type_column
        self.df = None
        self.load_data()

    def load_data(self):
        """
        Load and prepare the data from the CSV files matching the pattern.
        """
        # Find all CSV files matching the pattern
        file_pattern = os.path.join(self.folder_dir, self.csv_filename_pattern)
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files matching pattern '{self.csv_filename_pattern}' found in directory '{self.folder_dir}'.")

        # Load and concatenate data from all CSV files
        data_frames = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            data_frames.append(df)

        # Concatenate all DataFrames
        self.df = pd.concat(data_frames, ignore_index=True)

        # Clean and prepare the data
        self.df.columns = self.df.columns.str.strip()

        # Replace spaces in column names with underscores for easier access
        self.df.columns = self.df.columns.str.replace(' ', '_')

        # Ensure that trial condition and type columns are present
        if self.trial_condition_column not in self.df.columns:
            raise ValueError(f"Column '{self.trial_condition_column}' not found in the data.")
        if self.type_column not in self.df.columns:
            raise ValueError(f"Column '{self.type_column}' not found in the data.")

    def plot_metric(self, metric_mean_col, metric_std_col, ylabel, title, filename, custom_colors=None):
        """
        Plot a metric with error bars and save the plot.

        Parameters:
        - metric_mean_col: Column name for the metric mean values.
        - metric_std_col: Column name for the metric standard deviation values.
        - ylabel: Label for the Y-axis.
        - title: Title of the plot.
        - filename: Name of the file to save the plot.
        - custom_colors: Dictionary with 'Human' and 'Simulation' as keys and color tuples as values.
        """
        plt.figure(figsize=(10, 6))
        trial_conditions = sorted(self.df[self.trial_condition_column].unique())
        x = np.arange(len(trial_conditions))
        width = 0.35
        human_df = self.df[self.df[self.type_column] == 'Human']
        sim_df = self.df[self.df[self.type_column] == 'Simulation']
        human_means = human_df.set_index(self.trial_condition_column)[metric_mean_col].reindex(trial_conditions)
        human_stds = human_df.set_index(self.trial_condition_column)[metric_std_col].reindex(trial_conditions)
        sim_means = sim_df.set_index(self.trial_condition_column)[metric_mean_col].reindex(trial_conditions)
        sim_stds = sim_df.set_index(self.trial_condition_column)[metric_std_col].reindex(trial_conditions)

        # Define default colors if not provided
        if custom_colors is None:
            human_color = const.HUMAN_DATA_COLOR
            simulation_color = const.SIMULATION_RESULTS_COLOR
        else:
            human_color = custom_colors.get('Human', (0.0, 0.4470, 0.7410))
            simulation_color = custom_colors.get('Simulation', (0.8500, 0.3250, 0.0980))

        # Plot the bars without edge colors
        plt.bar(x - width/2, human_means, width, yerr=human_stds, capsize=5,
                label='Human', color=human_color, edgecolor='none')
        plt.bar(x + width/2, sim_means, width, yerr=sim_stds, capsize=5,
                label='Simulation', color=simulation_color, edgecolor='none')

        # Add annotations on top of the bars
        offset = max(human_stds.max(), sim_stds.max()) * 0.05  # Adjust the offset as needed
        for i in range(len(x)):
            # Handle possible NaN values
            human_mean = human_means.iloc[i]
            human_std = human_stds.iloc[i]
            sim_mean = sim_means.iloc[i]
            sim_std = sim_stds.iloc[i]

            if not np.isnan(human_mean) and not np.isnan(human_std):
                plt.text(x[i] - width/2, human_mean + human_std + offset,
                         f"{human_mean:.1f}±{human_std:.1f}",
                         ha='center', va='bottom', fontsize=9)
            if not np.isnan(sim_mean) and not np.isnan(sim_std):
                plt.text(x[i] + width/2, sim_mean + sim_std + offset,
                         f"{sim_mean:.1f}±{sim_std:.1f}",
                         ha='center', va='bottom', fontsize=9)
        plt.xlabel('Trial Condition (seconds)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x, trial_conditions)
        plt.legend()
        plt.tight_layout()
        # Save the plot
        file_path = os.path.join(self.folder_dir, filename)
        plt.savefig(file_path)
        plt.close()
        print(f"Plot '{title}' saved to {file_path}")

def plot_aggregated_fixation_metrics(folder_dir):
    """
    Plot aggregated fixation metrics using the MetricsPlotter class.

    Parameters:
    - folder_dir: Directory where the CSV file and plots are located.
    """
    csv_filename = 'aggregated_fixation_metrics_summary.csv'
    plotter = MetricsPlotter(folder_dir, csv_filename, trial_condition_column='Trial_Condition', type_column='Type')

    # Map the relevant columns to simpler names
    df = plotter.df
    df = df.rename(columns={
        'Average_Saccade_Length_(px)_mean': 'AvgSaccadeLength_Mean',
        'Average_Saccade_Length_(px)_std': 'AvgSaccadeLength_STD',
        'Word_Skip_Percentage_by_Saccades_V2_With_Word_Index_Correction_mean': 'WordSkipRate_Mean',
        'Word_Skip_Percentage_by_Saccades_V2_With_Word_Index_Correction_std': 'WordSkipRate_STD',
        'Revisit_Percentage_by_Saccades_V2_With_Word_Index_Correction_mean': 'RevisitRate_Mean',
        'Revisit_Percentage_by_Saccades_V2_With_Word_Index_Correction_std': 'RevisitRate_STD',
        'Average_Reading_Speed_(wpm)_mean': 'ReadingSpeed_Mean',
        'Average_Reading_Speed_(wpm)_std': 'ReadingSpeed_STD',
    })
    plotter.df = df  # Update the DataFrame in the plotter

    # Generate the plots
    plotter.plot_metric(
        metric_mean_col='AvgSaccadeLength_Mean',
        metric_std_col='AvgSaccadeLength_STD',
        ylabel='Average Saccade Length (px)',
        title='Average Saccade Length by Trial Condition',
        filename='avg_saccade_length_by_trial_condition.png'
    )

    plotter.plot_metric(
        metric_mean_col='WordSkipRate_Mean',
        metric_std_col='WordSkipRate_STD',
        ylabel='Word Skip Rate (%)',
        title='Word Skip Rate by Trial Condition',
        filename='word_skip_rate_by_trial_condition.png'
    )

    plotter.plot_metric(
        metric_mean_col='RevisitRate_Mean',
        metric_std_col='RevisitRate_STD',
        ylabel='Revisit Rate (%)',
        title='Revisit Rate by Trial Condition',
        filename='revisit_rate_by_trial_condition.png'
    )

    plotter.plot_metric(
        metric_mean_col='ReadingSpeed_Mean',
        metric_std_col='ReadingSpeed_STD',
        ylabel='Reading Speed (wpm)',
        title='Average Reading Speed by Trial Condition',
        filename='reading_speed_by_trial_condition.png'
    )

# def plot_comprehension_metrics(folder_dir):
#     """
#     Plot comprehension metrics using the MetricsPlotter class.

#     Parameters:
#     - folder_dir: Directory where the CSV file and plots are located.
#     """
#     csv_filename = 'comprehension_metrics_summary.csv'
#     plotter = MetricsPlotter(folder_dir, csv_filename, trial_condition_column='time_constraint', type_column='Type')

#     # Map the relevant columns to simpler names
#     df = plotter.df
#     # Since the comprehension metrics may have different column names, adjust accordingly
#     df = df.rename(columns={
#         'MCQ_Accuracy_mean': 'MCQAccuracy_Mean',
#         'MCQ_Accuracy_std': 'MCQAccuracy_STD',
#         'Free_Recall_Score_mean': 'FreeRecallScore_Mean',
#         'Free_Recall_Score_std': 'FreeRecallScore_STD',
#     })
#     plotter.df = df  # Update the DataFrame in the plotter

#     # Generate the plots
#     plotter.plot_metric(
#         metric_mean_col='MCQAccuracy_Mean',
#         metric_std_col='MCQAccuracy_STD',
#         ylabel='MCQ Accuracy (%)',
#         title='MCQ Accuracy by Time Constraint',
#         filename='mcq_accuracy_by_time_constraint.png'
#     )

#     plotter.plot_metric(
#         metric_mean_col='FreeRecallScore_Mean',
#         metric_std_col='FreeRecallScore_STD',
#         ylabel='Free Recall Score (%)',
#         title='Free Recall Score by Time Constraint',
#         filename='free_recall_score_by_time_constraint.png'
#     )
    
def plot_comprehension_metrics(folder_dir):
    """
    Plot comprehension metrics using the MetricsPlotter class.

    Parameters:
    - folder_dir: Directory where the CSV files and plots are located.
    """
    csv_filename_pattern = 'comprehension_metrics_*.csv'  # Use the filename pattern
    plotter = MetricsPlotter(folder_dir, csv_filename_pattern, trial_condition_column='time_constraint', type_column='Type')

    # Map the relevant columns to simpler names
    df = plotter.df
    # Since the comprehension metrics may have different column names, adjust accordingly
    df = df.rename(columns={
        'MCQ_Accuracy_mean': 'MCQAccuracy_Mean',
        'MCQ_Accuracy_std': 'MCQAccuracy_STD',
        'Free_Recall_Score_mean': 'FreeRecallScore_Mean',
        'Free_Recall_Score_std': 'FreeRecallScore_STD',
    })
    plotter.df = df  # Update the DataFrame in the plotter

    # Generate the plots
    plotter.plot_metric(
        metric_mean_col='MCQAccuracy_Mean',
        metric_std_col='MCQAccuracy_STD',
        ylabel='MCQ Accuracy (%)',
        title='MCQ Accuracy by Time Constraint',
        filename='mcq_accuracy_by_time_constraint.png'
    )

    plotter.plot_metric(
        metric_mean_col='FreeRecallScore_Mean',
        metric_std_col='FreeRecallScore_STD',
        ylabel='Free Recall Score (%)',
        title='Free Recall Score by Time Constraint',
        filename='free_recall_score_by_time_constraint.png'
    )


# Example usage:
if __name__ == "__main__":
    
    # ========================================================== Read Data ==========================================================
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/processed_data/11_05_19_13/processed_human_scanpath_wo_p1_to_p4.json"  # With word index tuning
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_13_12_36_trial_wise_scanpaths_corrected/integrated_corrected_human_scanpath.json"   # p30, data corrected by fix8
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_13_17_17_warps/integrated_corrected_human_scanpath.json"    # p5-p11, warping
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_13_17_39_warps_plus_attach/integrated_corrected_human_scanpath.json"    # p5-p11, warping + attach
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_13_20_18/integrated_corrected_human_scanpath.json"  # p5-p32, warping + attach
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_13_28_warping/integrated_corrected_human_scanpath.json"  # p5-p32, warping only
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_14_13_baselines_p5_to_p32_stim0/integrated_corrected_human_scanpath.json"    # p5-p32, baselines without filtering
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_15_17_warping_plus_attach/integrated_corrected_human_scanpath.json"  # p5-p32, warping + attach
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_17_47_attach/integrated_corrected_human_scanpath.json"  # p5-p32, attach only
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_original/integrated_corrected_human_scanpath.json"   # p5-p32, original data, stim1
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_warp_attaching/integrated_corrected_human_scanpath.json"   # p5-p32, warping + attach, stim1
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_warp/integrated_corrected_human_scanpath.json"   # p5-p32, warping only, stim1
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_stretch/integrated_corrected_human_scanpath.json"   # p5-p32, stretching only, stim1
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_20_00_stimid0/integrated_corrected_human_scanpath.json"   # p5-p32, warp, chain, warp+chain, stretch combinations, stimid0
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_13_47_stimid1/integrated_corrected_human_scanpath.json"   # p5-p32, warp, chain, warp+chain, stretch combinations, stimid1
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_13_45_stimid1_warp_attach/integrated_corrected_human_scanpath.json"   # p5-p32, warp+attach only, stimid1,
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli/integrated_corrected_human_scanpath.json"  # p5-p32, all stimuli, all corrections
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_15_12_stimid2/integrated_corrected_human_scanpath.json"   # p5-p32, warp, chain, warp+chain, stretch combinations, stimid2
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_16_15_stimid3/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid3
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_10_29_stimid4/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid4
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_13_54_stimid5/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid5
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_14_55_stimid6/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid6
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_15_30_stimid7/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid7
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_16_11_stimid8/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid8
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_16_54_stimid0/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid0
    # human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_17_30_stimid1/integrated_corrected_human_scanpath.json"  # p5-p32, warp, chain, warp+chain, stretch combinations, stimid1
    human_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli/11_18_17_40_integrated_corrected_human_scanpath.json"  # p5-p32, all stimuli, all corrections

    # ------------------------------------------------------------ My Simulator ------------------------------------------------------------
    # sim_data_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s/sim_processed_scanpath.json"    
    # sim_data_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s/sim_processed_scanpath_merge_filter.json"
    # sim_data_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s/corrected_simulation_data/11_23_21_41_corrected_simulation_trial_wise_scanpaths_35dot3px/integrated_corrected_human_scanpath.json"
    # # Merged the adjacent ones (<1.0 degrees), and use fix8 to correct the data, see whether the results can be better --> scanpath sim does not seem to be better
    sim_data_path = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_16_54_simulation_data_correction_attach/integrated_corrected_human_scanpath.json"      # Merged the adjacent ones (<0.5 degrees), and use fix8 to correct the data

    # ------------------------------------------------------------ Baseline Models ------------------------------------------------------------
    scanpath_vqa = "/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/formated_model_predictions/converted_simulation_data.json"
    ezreader = "/home/baiy4/reading-model/baseline_models/ezreader/ezreader_output_data.json"
    swift = "/home/baiy4/reading-model/baseline_models/swift/swift_output_data.json"
    # sim_data_path = swift        # NOTE comment this later

    # ============================================================ Calculate Metrics ============================================================
    # Create a folder
    metrics_folder_dir = os.path.join(const.USER_STUDY_DIR, f"calculated_metrics_{datetime.datetime.now().strftime('%m_%d_%H_%M')}")
    if not os.path.exists(metrics_folder_dir):
        os.makedirs(metrics_folder_dir)  
    # ------------------------------------------------------------ Scanpath Similarity ------------------------------------------------------------
    # Calculate the metrics
    num_fixations_to_specify = 2     # NOTE check and change this every time!!!
    calc_scanpath_similarity(human_data_path, sim_data_path, metrics_folder_dir, num_fixations=num_fixations_to_specify)
    # ------------------------------------------------------------ Aggregated Fixation Metrics ------------------------------------------------------------       
    # Calculate aggregated fixation metrics
    calc_aggregated_fixation_metrics(human_data_path, sim_data_path, metrics_folder_dir, word_metadata_path=const.BBOX_METADATA_DIR)
    # ------------------------------------------------------------ Comprehension Metrics ------------------------------------------------------------
    # Call the function
    # sim_json_comprehension_file_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s/simulate_xep_text_level.json"
    # sim_json_comprehension_file_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/simulate_xep_text_level_w_Kintsch.json"
    # sim_json_comprehension_file_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_11_29_13_18/simulate_xep_text_level_w_Kintsch.json_text_similarity_threshold_0dot6_exploration_rate_1dot0.json"
    # sim_json_comprehension_file_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_12_01_19_12/simulate_xep_text_level_w_Kintsch_text_similarity_threshold_1.0_exploration_rate_1.0.json"
    sim_json_comprehension_file_path = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_12_02_09_59/simulate_xep_text_level_w_Kintsch_text_similarity_threshold_0.2_exploration_rate_1.0.json"
    human_json_comprehension_file_path = '/home/baiy4/reading-model/data_analysis/human_data/comprehension_data/processed_human_comprehension_data_p1_to_p32.json'
    sim_df, human_df = calc_comprehension_metrics(
        sim_json_data_path=sim_json_comprehension_file_path,
        # human_csv_data_path=human_json_comprehension_file_path,
        human_json_data_path=human_json_comprehension_file_path,
        folder_dir=metrics_folder_dir,
        sim_output_csv='simulation_comprehension_results.csv',
        human_output_csv='human_comprehension_results.csv'
    )
    # ============================================================ Plotting ============================================================
    # Plot the metrics
    plot_aggregated_fixation_metrics(metrics_folder_dir)
    plot_comprehension_metrics(metrics_folder_dir)