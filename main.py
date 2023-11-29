#!/bin/python3

import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from scipy.stats import chi2, multivariate_normal
from scipy.linalg import sqrtm

def prob_calc(L_and_y_k: np.array, P_D:float, P_G:float):
    
    prob_vec = np.array([0])
    sum_of_like = np.sum(L_and_y_k[0])
    prob_vec = np.append(prob_vec, (1 - P_G * P_D) \
                                   / (1 - P_G * P_D + sum_of_like))

    for col in L_and_y_k.T:

        prob_vec = np.append(prob_vec, (col[0]) \
                                       / (1 - P_G * P_D + sum_of_like))

    prob_vec = np.delete(prob_vec, 0)

    return prob_vec

def prob_weight_avg(prob_vec:np.array, L_and_y_k:np.array):

    r_k_tilde = np.array([[0], [0]], dtype=float)
    prob_vec = np.delete(prob_vec, 0)

    for (prob, col) in zip(prob_vec, L_and_y_k.T):

        r_k_tilde += prob * np.array([[col[1]], [col[2]]])

    return r_k_tilde

def EKF(bearing_mat: np.array, range_mat: np.array, init_state: np.array):

    sigma_x = 10                    # m
    sigma_y = 10                    # m
    sigma_x_dot = 5                 # m/s
    sigma_y_dot = 5                 # m/s
    sigma_omega = np.deg2rad(2)     # rad/s
    sigma_r = 10                    # m
    sigma_theta = np.deg2rad(2)     # rad

    P = np.diagflat([sigma_x**2, sigma_y**2, sigma_x_dot**2, sigma_y_dot**2, sigma_omega**2])
    Q_k_minus_1 = np.diagflat([sigma_x_dot**2, sigma_y_dot**2, sigma_omega**2])
    R_k = np.diagflat([sigma_r**2, sigma_theta**2])
    L_k_minus_1 = np.array([[0.5, 0, 0], \
                            [0, 0.5, 0], \
                            [1, 0, 0], \
                            [0, 1, 0], \
                            [0, 0, 1]])

    state_hist = np.atleast_2d(init_state)

    for i, (ranges, bearings) in enumerate(zip(range_mat, bearing_mat)):
 
        # PREDICITON STEP
        x = state_hist[i][0]
        y = state_hist[i][1]
        x_dot = state_hist[i][2]
        y_dot = state_hist[i][3]
        omega = state_hist[i][4]

        cos_sin = (omega * np.cos(omega) - np.sin(omega)) / omega**2
        sin_cos = (omega * np.sin(omega) - 1 + np.cos(omega)) / omega**2
        F_k_minus_1 = np.array([[1, 0, np.sin(omega) / omega, \
                                 -(1 - np.cos(omega)) / omega, \
                                 cos_sin * x_dot - sin_cos * y_dot], \
                                [0, 1, (1 - np.cos(omega)) / omega, \
                                 np.sin(omega) / omega, \
                                 sin_cos * x_dot + cos_sin * y_dot], \
                                [0, 0, np.cos(omega), -np.sin(omega), \
                                 -np.sin(omega) * x_dot - np.cos(omega) * y_dot], \
                                [0, 0, np.sin(omega), np.cos(omega), \
                                 np.cos(omega) * x_dot - np.sin(omega) * y_dot], \
                                [0, 0, 0, 0, 1]])
        P = F_k_minus_1 @ P @ F_k_minus_1.T \
            + L_k_minus_1 @ Q_k_minus_1 @ L_k_minus_1.T
        F_nonlinear = np.array([[1, 0, np.sin(omega) / omega, \
                         -(1 - np.cos(omega)) / omega, 0], \
                        [0, 1, (1 - np.cos(omega)) / omega, \
                         np.sin(omega) / omega, 0], \
                        [0, 0, np.cos(omega), \
                         -np.sin(omega), 0], \
                        [0, 0, np.sin(omega), \
                         np.cos(omega), 0], \
                        [0, 0, 0, 0, 1]])
        x_k_k_minus_1 = F_nonlinear @ np.atleast_2d(state_hist[i]).T #add noise
        x = x_k_k_minus_1[0][0]
        y = x_k_k_minus_1[1][0]
        
        # CORRECTION STEP
        H_k = np.array([[x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0, 0, 0], \
                        [y / (x**2 + y**2), -x / (x**2 + y**2), 0, 0, 0]])
        S_k = H_k @ P @ H_k.T + np.eye(2) @ R_k @ np.eye(2)
        K_k = P @ H_k.T @ nplin.inv(S_k)
        y_curr_col = np.array([[np.sqrt(x**2 + y**2)], [np.arctan2(x, y)]])

        # FIND NEAREST NEIGHBOR - CLUTTER
        if np.ndim(bearings) != 0:

            dist_min = 1e8
            
            for (bearing, range) in zip(bearings, ranges):

                if not np.isnan(bearing) and not np.isnan(range):

                    y_meas = np.array([[range], [bearing]])
                    x_meas = np.array([[range * np.sin(bearing)], \
                                       [range * np.cos(bearing)]])
                    y_k = y_meas - y_curr_col
                    diff = x_meas - x_k_k_minus_1[0:2]
                    dist = nplin.norm(diff)
                    pass

                    if dist < dist_min:

                        dist_min = dist
                        y_k_min = y_k

            state_hist_new = x_k_k_minus_1 + K_k @ y_k_min

        # FIND NEAREST NEIGHBOR - CLEAN DATA  
        else:

            y_meas = np.array([[ranges], [bearings]])
            y_k = y_meas - y_curr_col
            state_hist_new = x_k_k_minus_1 + K_k @ y_k
        
        state_hist = np.vstack((state_hist, np.atleast_2d(state_hist_new).T))
        P = P - (K_k @ S_k @ K_k.T)

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
        
        # PREDICTION STEP
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
        P = F_k_minus_1 @ P @ F_k_minus_1.T \
            + L_k_minus_1 @ Q_k_minus_1 @ L_k_minus_1.T
        F_nonlinear = np.array([[1, 0, np.sin(omega) / omega, \
                         -(1 - np.cos(omega)) / omega, 0], \
                        [0, 1, (1 - np.cos(omega)) / omega, \
                         np.sin(omega) / omega, 0], \
                        [0, 0, np.cos(omega), \
                         -np.sin(omega), 0], \
                        [0, 0, np.sin(omega), \
                         np.cos(omega), 0], \
                        [0, 0, 0, 0, 1]])
        x_k_k_minus_1 = F_nonlinear @ np.atleast_2d(state_hist[i]).T #add noise
        x = x_k_k_minus_1[0][0]
        y = x_k_k_minus_1[1][0]

        # CORRECTION STEP
        H_k = np.array([[x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0, 0, 0], \
                        [y / (x**2 + y**2), -x / (x**2 + y**2), 0, 0, 0]])
        S_k = H_k @ P @ H_k.T + np.eye(2) @ R_k @ np.eye(2)
        K_k = P @ H_k.T @ nplin.inv(S_k)
        y_curr_col = np.array([[np.sqrt(x**2 + y**2)], [np.arctan2(x, y)]])

        df = 1 #(np.count_nonzero(~np.isnan(bearings)) - 1)
        alpha = 0.05
        P_G = 1 - alpha
        gate_thres = chi2.ppf(P_G, df)
        vol_gate = np.pi * gate_thres**(df / 2) * nplin.det(S_k)**(1 / 2)
        L_k_i_and_y_k = np.array([[0], [0], [0]])
        y_meas_list = []

        for (bearing, range) in zip(bearings, ranges):

            if not np.isnan(bearing) and not np.isnan(range):

                y_meas = np.array([[range], [bearing]])
                y_k = y_meas - y_curr_col
                mac_dist = y_k.T \
                           @ nplin.inv(S_k) \
                           @ (y_k)

                if mac_dist[0][0] < gate_thres:
                    
                    y_meas_list.append(y_meas)

        for y_meas in y_meas_list:
 
            if y_meas_list:

                L_k_i = vol_gate / len(y_meas_list) \
                        * multivariate_normal.pdf(y_meas.T, \
                                                    mean=y_curr_col.T[0], \
                                                    cov=S_k, \
                                                    allow_singular=True) \
                        * P_D
                y_k = y_meas - y_curr_col
                L_k_i_and_y_k = np.hstack((L_k_i_and_y_k, \
                                            np.block([[L_k_i], [y_k]])))

        L_k_i_and_y_k = np.delete(L_k_i_and_y_k, 0, 1)
        prob_vec = prob_calc(L_k_i_and_y_k, P_D, P_G)

        if len(prob_vec) <= 1:

            print(f'''PDAF: Missed detection on index {i}''')
            state_hist = np.vstack((state_hist, x_k_k_minus_1.T))
            continue

        r_k_tilde = prob_weight_avg(prob_vec, L_k_i_and_y_k)
        state_hist_new = x_k_k_minus_1 + K_k @ r_k_tilde
        state_hist = np.vstack((state_hist, np.atleast_2d(state_hist_new).T))
        P = (1 - prob_vec[0]) * (P - K_k @ S_k @ K_k.T) + prob_vec[0] * P \
            + K_k @ ((sum(prob_vec) - prob_vec[0]) \
                     * (L_k_i_and_y_k[0:3][1:]) \
                        @ (L_k_i_and_y_k[0:3][1:]).T \
                        - r_k_tilde @ r_k_tilde.T) \
            @ K_k.T

    return state_hist

def main():

    bearings_clean = np.genfromtxt('bearings_clean.csv', \
                                     delimiter=",", \
                                     dtype=float)
    ranges_clean = np.genfromtxt('ranges_clean.csv', \
                                   delimiter=",", \
                                   dtype=float)
    bearings_clutter = np.genfromtxt('bearings_clutter.csv', \
                                     delimiter=",", \
                                     dtype=float)
    ranges_clutter = np.genfromtxt('ranges_clutter.csv', \
                                   delimiter=",", \
                                   dtype=float)
    truth = np.genfromtxt('truth.csv', \
                          delimiter=",", \
                          dtype=float)

    init_state = np.array([500, 500, 7.5, 7.5, np.deg2rad(2)])
    time_truth = np.linspace(0, 199, 200)
    time = np.linspace(0, 200, 201)

    state_hist_clean_EKF = EKF(bearings_clean, ranges_clean, init_state)
    state_hist_clutter_EKF = EKF(bearings_clutter, ranges_clutter, init_state)
    state_hist_clutter_PDAF = PDAF(bearings_clutter, ranges_clutter, init_state)

    fig1, axs1 = plt.subplots(1, 2)
    axs1[0].set_title('Y vs X')
    axs1[0].plot(truth[:, 2], truth[:, 0])
    axs1[0].plot(state_hist_clean_EKF[:, 1], state_hist_clean_EKF[:, 0])
    # axs1[0].plot(state_hist_clutter_EKF[:, 1], state_hist_clutter_EKF[:, 0])
    axs1[0].plot(state_hist_clutter_PDAF[:, 1], state_hist_clutter_PDAF[:, 0])

    axs1[1].set_title('Omega vs Time')
    axs1[1].plot(time_truth, truth[:, 4])
    axs1[1].plot(time, state_hist_clean_EKF[:, 4])
    # axs1[1].plot(time, state_hist_clutter_EKF[:, 4])
    axs1[1].plot(time, state_hist_clutter_PDAF[:, 4])

    fig2, axs2 = plt.subplots(1, 2)
    axs2[0].set_title('X_dot vs Time')
    axs2[0].plot(time_truth, truth[:, 1])
    axs2[0].plot(time, state_hist_clean_EKF[:, 2])
    # axs2[0].plot(time, state_hist_clutter_EKF[:, 2])
    axs2[0].plot(time, state_hist_clutter_PDAF[:, 2])

    axs2[1].set_title('Y_dot vs Time')
    axs2[1].plot(time_truth, truth[:, 3])
    axs2[1].plot(time, state_hist_clean_EKF[:, 3])
    # axs2[1].plot(time, state_hist_clutter_EKF[:, 3])
    axs2[1].plot(time, state_hist_clutter_PDAF[:, 3])

    plt.show()

    return 1

if __name__=="__main__":
    main()