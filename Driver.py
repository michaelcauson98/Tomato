#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:53:20 2024

@author: michaelcauson
"""
import numpy as np
from tomato import Tomato

tomato = Tomato(room_dims=[30,30,10],audience_dims=[20,5],speaker_loc=[30,5.657196,1.7768])

tomato.set_truth(np.array([6,18]),np.array([20,-10.28567,4.45]))
data = tomato.generate_data(np.array([0.01,0.01,0.01,0.1,0.1,0.1]))
traj = tomato.plot_trajectory(np.array([6,18]),np.array([20,-10.28567,4.45]))

mcmc = tomato.MCMC(its=250_000,rw_std=[0.5,0.5,0.5,0.5,0.5])
tomato.plot_MCMC()
tomato.plot_MCMC_samples()
tomato.plot_histograms()
#tomato.plot_sample_trajectory(20)
tomato.plot_MCMC_samples_fwd()