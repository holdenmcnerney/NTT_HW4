#!/bin/python3

import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt

def EKF(bearing_mat: np.array, range_mat:np.array, init_state:np.array):

    sigma_x = 10        # m
    sigma_y = 10        # m
    sigma_x_dot = 5     # m/s
    sigma_y_dot = 5     # m/s
    sigma_omega = 2     # deg/s
    sigma_r = 10        # 10 m
    sigma_theta = 2     # deg

    P = np.diagflat([sigma_x, sigma_y, sigma_x_dot, sigma_y_dot, sigma_omega])
    # Q_k_minus_1 = np.array([[1 / 2, 0, 0], \
    #                         [0, 1 / 2, 0], \
    #                         [1, 0, 0], \
    #                         [0, 1, 0], \
    #                         [0, 0, 1]])
    Q_k_minus_1 = np.diagflat([sigma_x**2, sigma_x_dot**2, sigma_omega**2])
    L_k_minus_1 = np.array([[0, 0, 0], \
                            [0, 0, 0], \
                            [1, 0, 0], \
                            [0, 1, 0], \
                            [0, 0, 1]])
    R_k = np.diagflat([sigma_r, sigma_theta])

    state_hist = np.atleast_2d(init_state)

    for i, (ranges, bearings) in enumerate(zip(range_mat, bearing_mat)):
 
        x = state_hist[i][0]
        y = state_hist[i][1]
        x_dot = state_hist[i][2]
        y_dot = state_hist[i][3]
        omega = state_hist[i][4]

        cos_plus_sin = (omega * np.cos(omega) - np.sin(omega)) / omega**2
        sin_plus_cos = (omega * np.sin(omega) - 1 + np.cos(omega)) / omega**2
        # F_k_minus_1_jacob = np.array([[1, 0, np.sin(omega) / omega, \
        #                          -(1 - np.cos(omega)) / omega, \
        #                          cos_plus_sin * x_dot - sin_plus_cos * y_dot], \
        #                         [0, 1, (1 - np.cos(omega)) / omega, \
        #                          np.sin(omega) / omega, \
        #                          sin_plus_cos * x_dot + cos_plus_sin * y_dot], \
        #                         [0, 0, np.cos(omega), -np.sin(omega), \
        #                          -np.sin(omega) * x_dot - np.cos(omega) * y_dot], \
        #                         [0, 0, np.sin(omega), np.cos(omega), \
        #                          np.cos(omega) * x_dot - np.sin(omega * y_dot)], \
        #                         [0, 0, 0, 0, 1]])
        F_k_minus_1 = np.array([[1, 0, np.sin(omega) / omega, \
                                 -(1 - np.cos(omega)) / omega, 0], \
                                [0, 1, (1 - np.cos(omega)) / omega, \
                                 np.sin(omega) / omega, 0], \
                                [0, 0, np.cos(omega), -np.sin(omega), 0], \
                                [0, 0, np.sin(omega), np.cos(omega), 0], \
                                [0, 0, 0, 0, 1]])
        
        H_k = np.array([[x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0, 0, 0], \
                        [y / (x**2 + y**2), -x / (x**2 + y**2), 0, 0, 0]])
        
        # P = F_k_minus_1_jacob @ P @ np.transpose(F_k_minus_1_jacob) + L_k_minus_1 @ Q_k_minus_1 @ np.transpose(L_k_minus_1)
        P = F_k_minus_1 @ P @ np.transpose(F_k_minus_1) + L_k_minus_1 @ Q_k_minus_1 @ np.transpose(L_k_minus_1)
        S_k = H_k @ P @ np.transpose(H_k) + np.eye(2) @ R_k @ np.eye(2)
        K_k = P @ np.transpose(H_k) @ nplin.inv(S_k)
        y_curr = H_k @ state_hist[i]

        if np.ndim(bearings) != 0:
            
            mac_dist_min = 1e8

            for (bearing, range) in zip(bearings, ranges):

                y_meas = np.array([[range * np.sin(bearing)], [range * np.cos(bearing)]])
                mac_dist = np.transpose(y_meas - y_curr) @ S_k @ (y_meas - y_curr)

                if mac_dist < mac_dist_min:

                    mac_dist_min = mac_dist
                    y_k = y_meas - y_curr

                state_hist_new = np.transpose(np.atleast_2d(state_hist[i])) + K_k @ y_k
            
        else:
            y_meas = np.array([[ranges * np.sin(bearings)], [ranges * np.cos(bearings)]])
            y_k = y_meas - y_curr
            state_hist_new = np.transpose(np.atleast_2d(state_hist[i])) + K_k @ y_k
        
        state_hist = np.vstack((state_hist, np.transpose(np.atleast_2d(state_hist_new))))
        P = P - K_k @ S_k @ np.transpose(K_k)

    return state_hist

def main():

    bearings_clean = np.genfromtxt('bearings_clean.csv', delimiter=",", dtype=float)
    ranges_clean = np.genfromtxt('ranges_clean.csv', delimiter=",", dtype=float)
    bearings_clutter = np.genfromtxt('bearings_clutter.csv', delimiter=",", dtype=float)
    ranges_clutter = np.genfromtxt('ranges_clutter.csv', delimiter=",", dtype=float)

    truth = np.genfromtxt('truth.csv', delimiter=",", dtype=float)

    init_state = np.array([500, 500, 7.5, 7.5, 2])
    
    P_D = 0.9
    lambda_c = 0.0032

    state_hist_clean = EKF(bearings_clean, ranges_clean, init_state)
    # state_hist_clutter = EKF(bearings_clutter, ranges_clutter, init_state)

    fig1, axs1 = plt.subplots(1, 2)
    axs1[0].plot(state_hist_clean[:, 1], state_hist_clean[:, 0])
    # axs1[0].set_xlabel('')
    # axs1[0].set_ylabel('')
    # axs1[0].set_title('')

    axs1[1].plot(truth[:, 2], truth[:, 0])
    # axs1[1].set_xlabel('')
    # axs1[1].set_ylabel('')
    # axs1[1].set_title('')

    plt.show()

    return 1

if __name__=="__main__":
    main()