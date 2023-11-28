#!/bin/python3

import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from scipy.stats import invgamma, chi2, norm
from scipy.linalg import sqrtm, expm

def norm_dist():

    return 1

def EKF(bearing_mat: np.array, range_mat: np.array, init_state: np.array):

    sigma_x = 10                    # m
    sigma_y = 10                    # m
    sigma_x_dot = 5                 # m/s
    sigma_y_dot = 5                 # m/s
    sigma_omega = np.deg2rad(2)     # deg/s
    sigma_r = 10                    # m
    sigma_theta = np.deg2rad(2)     # deg

    P = np.diagflat([sigma_x, sigma_y, sigma_x_dot, sigma_y_dot, sigma_omega])
    Q_k_minus_1 = np.diagflat([sigma_x**2, sigma_x_dot**2, sigma_omega**2])
    L_k_minus_1 = np.array([[0, 0, 0], \
                            [0, 0, 0], \
                            [1, 0, 0], \
                            [0, 1, 0], \
                            [0, 0, 1]])
    R_k = np.diagflat([sigma_r**2, sigma_theta**2])

    state_hist = np.atleast_2d(init_state)

    for i, (ranges, bearings) in enumerate(zip(range_mat, bearing_mat)):
 
        x = state_hist[i][0]
        y = state_hist[i][1]
        x_dot = state_hist[i][2]
        y_dot = state_hist[i][3]
        omega = state_hist[i][4]
        
        cos_plus_sin = (omega * np.cos(omega) - np.sin(omega)) / omega**2
        sin_plus_cos = (omega * np.sin(omega) - 1 + np.cos(omega)) / omega**2
        F_k_minus_1 = np.array([[1, 0, np.sin(omega) / omega, \
                                 -(1 - np.cos(omega)) / omega, \
                                 cos_plus_sin * x_dot - sin_plus_cos * y_dot], \
                                [0, 1, (1 - np.cos(omega)) / omega, \
                                 np.sin(omega) / omega, \
                                 sin_plus_cos * x_dot + cos_plus_sin * y_dot], \
                                [0, 0, np.cos(omega), -np.sin(omega), \
                                 -np.sin(omega) * x_dot - np.cos(omega) * y_dot], \
                                [0, 0, np.sin(omega), np.cos(omega), \
                                 np.cos(omega) * x_dot - np.sin(omega * y_dot)], \
                                [0, 0, 0, 0, 1]])

        H_k = np.array([[x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0, 0, 0], \
                        [y / (x**2 + y**2), -x / (x**2 + y**2), 0, 0, 0]])
        P = F_k_minus_1 @ P @ np.transpose(F_k_minus_1) \
            + L_k_minus_1 @ Q_k_minus_1 @ np.transpose(L_k_minus_1)
        S_k = H_k @ P @ np.transpose(H_k) + np.eye(2) @ R_k @ np.eye(2)
        K_k = P @ np.transpose(H_k) @ nplin.inv(S_k)
        y_curr_col = np.array([[np.sqrt(x**2 + y**2)], [np.arctan2(x, y)]])

        if np.ndim(bearings) != 0:

            mac_dist_vec = np.array([[0], [0], [0]])

            df = (len(bearings) - 1) * (2 - 1)
            q = .05
            gate_thres = chi2.ppf(q, df)
            
            for (bearing, range) in zip(bearings, ranges):

                if not np.isnan(bearing) and not np.isnan(range):

                    y_meas = np.array([[range], [bearing]])
                    y_k = y_meas - y_curr_col
                    mac_dist = np.transpose(y_k) \
                                            @ nplin.inv(S_k) \
                                            @ (y_k)
                    
                    if mac_dist[0][0] < gate_thres:

                        mac_dist_vec = np.hstack((mac_dist_vec, \
                                                  np.block([[mac_dist], [y_k]])))

            if np.array_equal(mac_dist_vec, np.array([[0], [0], [0]])):
                y_k_min = y_curr_col - y_curr_col

            else:
                mac_dist_vec = np.delete(mac_dist_vec, 0, 1)
                idx = np.argmin(mac_dist_vec[0])
                y_k_min = np.array([[mac_dist_vec[1][idx]], \
                                    [mac_dist_vec[2][idx]]])

            state_hist_new = np.transpose(np.atleast_2d(state_hist[i])) \
                             + K_k @ y_k_min
            
        else:
            y_meas = np.array([[ranges], [bearings]])
            y_k = y_meas - y_curr_col
            state_hist_new = np.transpose(np.atleast_2d(state_hist[i])) + K_k @ y_k
        
        state_hist = np.vstack((state_hist, np.transpose(np.atleast_2d(state_hist_new))))
        P = P - K_k @ S_k @ np.transpose(K_k)

    return state_hist

def PDAF(bearing_mat: np.array, range_mat: np.array, init_state: np.array):

    sigma_x = 10                    # m
    sigma_y = 10                    # m
    sigma_x_dot = 5                 # m/s
    sigma_y_dot = 5                 # m/s
    sigma_omega = np.deg2rad(2)     # deg/s
    sigma_r = 10                    # m
    sigma_theta = np.deg2rad(2)     # deg
    P_D = 0.9
    lambda_c = 0.0032

    P = np.diagflat([sigma_x, sigma_y, sigma_x_dot, sigma_y_dot, sigma_omega])
    Q_k_minus_1 = np.diagflat([sigma_x**2, sigma_x_dot**2, sigma_omega**2])
    L_k_minus_1 = np.array([[0, 0, 0], \
                            [0, 0, 0], \
                            [1, 0, 0], \
                            [0, 1, 0], \
                            [0, 0, 1]])
    R_k = np.diagflat([sigma_r**2, sigma_theta**2])

    state_hist = np.atleast_2d(init_state)

    for i, (ranges, bearings) in enumerate(zip(range_mat, bearing_mat)):
 
        x = state_hist[i][0]
        y = state_hist[i][1]
        x_dot = state_hist[i][2]
        y_dot = state_hist[i][3]
        omega = state_hist[i][4]
        
        cos_plus_sin = (omega * np.cos(omega) - np.sin(omega)) / omega**2
        sin_plus_cos = (omega * np.sin(omega) - 1 + np.cos(omega)) / omega**2
        F_k_minus_1 = np.array([[1, 0, np.sin(omega) / omega, \
                                 -(1 - np.cos(omega)) / omega, \
                                 cos_plus_sin * x_dot - sin_plus_cos * y_dot], \
                                [0, 1, (1 - np.cos(omega)) / omega, \
                                 np.sin(omega) / omega, \
                                 sin_plus_cos * x_dot + cos_plus_sin * y_dot], \
                                [0, 0, np.cos(omega), -np.sin(omega), \
                                 -np.sin(omega) * x_dot - np.cos(omega) * y_dot], \
                                [0, 0, np.sin(omega), np.cos(omega), \
                                 np.cos(omega) * x_dot - np.sin(omega * y_dot)], \
                                [0, 0, 0, 0, 1]])

        H_k = np.array([[x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0, 0, 0], \
                        [y / (x**2 + y**2), -x / (x**2 + y**2), 0, 0, 0]])
        P = F_k_minus_1 @ P @ np.transpose(F_k_minus_1) \
            + L_k_minus_1 @ Q_k_minus_1 @ np.transpose(L_k_minus_1)
        S_k = H_k @ P @ np.transpose(H_k) + np.eye(2) @ R_k @ np.eye(2)
        K_k = P @ np.transpose(H_k) @ nplin.inv(S_k)
        y_curr_col = np.array([[np.sqrt(x**2 + y**2)], [np.arctan2(x, y)]])

        df = (len(bearings) - 1) * (2 - 1)
        alpha = 0.05
        P_G = 1 - alpha
        gate_thres = chi2.ppf(P_G, df)
        vol_gate = np.pi * gate_thres**(df / 2) * sqrtm(S_k)
        y_k_vec = np.array([[0], [0]])
        L_list = []
        y_meas_list = []

        for (bearing, range) in zip(bearings, ranges):

            if not np.isnan(bearing) and not np.isnan(range):

                y_meas = np.array([[range], [bearing]])
                y_k = y_meas - y_curr_col
                mac_dist = np.transpose(y_k) \
                                        @ nplin.inv(S_k) \
                                        @ (y_k)

                if mac_dist[0][0] < gate_thres:
                    y_k_vec = np.hstack((y_k_vec, y_k))
                    L_k_i = 1 / lambda_c * norm.pdf(y_meas, loc=y_curr_col, scale=S_k) * P_D
                    L_list.append(L_k_i)
                    y_meas_list.append(y_meas)

        if L_list == []:
            
            Prob = 1

        else:

            a = 1
        
        y_k_vec = np.delete(y_k_vec, 0, 1)


        pass

    return 1

def main():

    bearings_clean = np.genfromtxt('bearings_clean.csv', delimiter=",", dtype=float)
    ranges_clean = np.genfromtxt('ranges_clean.csv', delimiter=",", dtype=float)
    bearings_clutter = np.genfromtxt('bearings_clutter.csv', delimiter=",", dtype=float)
    ranges_clutter = np.genfromtxt('ranges_clutter.csv', delimiter=",", dtype=float)

    truth = np.genfromtxt('truth.csv', delimiter=",", dtype=float)

    init_state = np.array([500, 500, 7.5, 7.5, 2])

    state_hist_clean_EKF = EKF(bearings_clean, ranges_clean, init_state)
    state_hist_clutter_EKF = EKF(bearings_clutter, ranges_clutter, init_state)

    # state_hist_clean_PDAF = PDAF(bearings_clean, ranges_clean, init_state)
    state_hist_clutter_PDAF = PDAF(bearings_clutter, ranges_clutter, init_state)

    plt.plot(state_hist_clean_EKF[:, 1], state_hist_clean_EKF[:, 0])
    plt.plot(truth[:, 2], truth[:, 0])
    plt.plot(state_hist_clutter_EKF[:, 1], state_hist_clutter_EKF[:, 0])

    plt.show()

    return 1

if __name__=="__main__":
    main()