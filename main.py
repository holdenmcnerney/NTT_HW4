#!/bin/python3

import numpy as np

def EKF():



    return 1

def main():

    bearings_clean = np.genfromtxt('bearings_clean.csv', delimiter=",", dtype=float)
    ranges_clean = np.genfromtxt('ranges_clean.csv', delimiter=",", dtype=float)

    init_state = np.array([500, 7.5, 500, 7.5, 2])

    sigma_x = 10            # m
    sigma_y = 10            # m
    sigma_x_dot = 5         # m/s
    sigma_x_dot = 5         # m/s
    sigma_omega = 2         # deg/s
    
    sigma_r = 10            # 10 m
    sigma_theta = 2         # deg
    P_D = 0.9
    lambda_c = 0.0032



    return 1

if __name__=="__main__":
    main()