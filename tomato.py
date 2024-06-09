#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:12:43 2024

@author: michaelcauson
"""
import numpy as np
import matplotlib.pyplot as plt

class Tomato:
    
    def __init__(self,room_dims,audience_dims,speaker_loc,sigmax=5,sigmay=5,sigmaz=5):
        self.room_dims = room_dims # [x,y,z] lengths
        self.audience_dims = audience_dims # [x,z] lengths, width is assumed to be room_dims[1]
        self.speaker_loc = speaker_loc # [x_s,y_s,z_s]
        self.refinement = 100 # refinement used in trajectory plots
        self.audience = self._generate_audience() # generate audience meshgrid
        self.sigmax = sigmax # std of prior u_x
        self.sigmay = sigmay # std of prior u_y
        self.sigmaz = sigmaz # std of prior u_z
        self.truth_set = 0 # logical for truth set
    
        
    def fwd_map(self,x0,u):
        x0 = np.concatenate((x0,[self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*x0[0]]))
        tof = self.assess_tof(x0, u)
        xf = x0 + u*tof + 0.5*np.array([0,0,-9.81])*tof**2
        vf = u + np.array([0,0,-9.81])*tof
        return np.concatenate((xf,vf))
    
    def set_truth(self,x0,u):
        x0_z = np.concatenate((x0,[self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*x0[0]]))
        self.true_x0 = x0
        self.true_u = u
        self.true_tf = self.assess_tof(x0_z, u)
        self.truth_set = 1
    
    def assess_tof(self,x0,u):
        assert u[0] > 0, "Initial velocity in x-direction must be positive"
        assert x0[0] > 0 and x0[0] < self.audience_dims[0], "x coordinate must be in audience"
        assert x0[1] > 0 and x0[1] < self.room_dims[1], "y coordinate must be in audience"
        
        a = 0.5 * -9.81
        b = u[2]
        c1 = x0[2]
        c2 = x0[2]-self.room_dims[2]
        
        # Case 1: tomato hits the floor
        tof_floor = (-b - np.sqrt(b**2 - 4*a*c1))/(2*a)
        
        # Case 2: tomato hits the ceiling (if at all)
        if b**2 - 4*a*c2 > 0:
            tof_ceil = (-b + np.sqrt(b**2 - 4*a*c2))/(2*a) 
        else:
            tof_ceil = 1000 # tomato never hits ceiling (set sufficiently large)
            
        # Case 3: tomato hits the back wall
        tofx = (self.speaker_loc[0]-x0[0])/u[0] # speed = dist/time
        
        # Case 4: Tomato hits either of the side walls
        if u[1] != 0:
            tofy = (self.room_dims[1]-x0[1])/np.abs(u[1]) if u[1]>=0 else x0[1]/np.abs(u[1])
        else:
            tofy = 1000
            
        # Find which TOF is satisfied first
        tof = np.min([tof_floor,tof_ceil,tofx,tofy])
        return tof
    
    def compute_trajectory(self,x0,u):
        x0 = np.concatenate((x0,[self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*x0[0]]))
        tof = self.assess_tof(x0, u)
        traj = [x0 + u*i + 0.5*np.array([0,0,-9.81])*i**2 for i in list(np.linspace(0,tof,self.refinement))]
        traj = np.array(traj).T
        
        return traj
    
    def _generate_audience(self):
        x = np.linspace(0,self.audience_dims[0],self.refinement)
        y = np.linspace(0,self.room_dims[1],self.refinement)
        X, Y = np.meshgrid(x,y)
        Z = self.audience_dims[1] - (self.audience_dims[1]/self.audience_dims[0])*X
        return X, Y, Z
    
    def plot_scene(self):
        fig = plt.figure(figsize=(17,5))

        ax = fig.add_subplot(131, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.view_init(0, -90)
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
    
        ax = fig.add_subplot(132, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.view_init(45, 45)
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)

        ax = fig.add_subplot(133, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.view_init(90, 0)
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        plt.show()
        
    def plot_trajectory(self,x0,u):
        
        traj = self.compute_trajectory(x0, u)
        zero_point = sum(traj[2]>0)
        
        fig = plt.figure(figsize=(17,5))

        ax = fig.add_subplot(131, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(0, -90)
        ax.plot(traj[0], traj[1], traj[2],color='red',linestyle='--')
        ax.scatter(*traj.T[0], color = 'red')
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        if zero_point< self.refinement and traj[0][zero_point]<self.room_dims[0] and traj[1][zero_point]<self.room_dims[1]:
            ax.scatter(*[traj[0][zero_point],traj[1][zero_point],0],s=3,marker='*',color='red')

    
        ax = fig.add_subplot(132, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(45, 45)
        ax.plot(traj[0], traj[1], traj[2],color='red',linestyle='--')
        ax.scatter(*traj.T[0], color = 'red')
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        if zero_point< self.refinement and traj[0][zero_point]<self.room_dims[0] and traj[1][zero_point]<self.room_dims[1]:
            ax.scatter(*[traj[0][zero_point],traj[1][zero_point],0],s=3,marker='*',color='red')

        ax = fig.add_subplot(133, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(90, 0)
        ax.plot(traj[0], traj[1], traj[2],color='red',linestyle='--')
        ax.scatter(*traj.T[0], color = 'red')
        ax.scatter(*self.speaker_loc, color = 'blue')
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        if zero_point< self.refinement and traj[0][zero_point]<self.room_dims[0] and traj[1][zero_point]<self.room_dims[1]:
            ax.scatter(*[traj[0][zero_point],traj[1][zero_point],0],s=3,marker='*',color='red')
        plt.show()
        return traj
    
    
    def generate_data(self,noise_cov):
        fwdmap = self.fwd_map(self.true_x0,self.true_u)
        noise = np.random.multivariate_normal([0,0,0,0,0,0],np.diag(noise_cov))
        data = fwdmap + noise
        self.data = data
        self.noise_cov = noise_cov
        return fwdmap, noise, data
    
    def likelihood(self,x0,u):
        return np.exp(-0.5*np.sum((self.data - self.fwd_map(x0,u))**2 * (1/self.noise_cov)))
    
    def prior(self,x0,u):
        x0_prob = 1/self.audience_dims[0]
        y0_prob = 1/self.room_dims[1]
        ux_prob = np.exp(-0.5*(u[0]-self.true_u[0])**2/(self.sigmax**2))
        uy_prob = np.exp(-0.5*(u[1]-self.true_u[1])**2/(self.sigmay**2))
        uz_prob = np.exp(-0.5*(u[2]-self.true_u[2])**2/(self.sigmaz**2))
        
        return x0_prob*y0_prob*ux_prob*uy_prob*uz_prob
    
    def MCMC(self,its=1_000_000,
             rw_std=[0.5,0.5,0.5,0.5,0.5]):
        u0=[self.true_x0[0],self.true_x0[1],self.true_u[0],self.true_u[1],self.true_u[2]]
        u_list = np.zeros( (its,5) )
        u = np.array(u0)
        p = np.log(self.prior(u[:2], u[2:])) + np.log(self.likelihood(u[:2], u[2:]))
        for i in range(its):
            new_u = np.array([-1,-1,-1])
            while new_u[0] < 0 or new_u[0] > self.audience_dims[0] or new_u[1] < 0 or new_u[1] > self.room_dims[1] or new_u[2] < 0:
                new_u = np.random.multivariate_normal(u, np.diag(rw_std))
            new_p = np.log(self.prior(new_u[:2], new_u[2:])) + np.log(self.likelihood(new_u[:2], new_u[2:]))
            r = np.minimum(0,new_p - p)
            alpha = np.random.uniform(0,1)
            if np.log(alpha) < r:
                u = new_u
                p = new_p
            u_list[i] = u
            
            self.mcmc_chain = u_list[round(0.1*its):]
        return u_list[round(0.1*its):]
    
    def plot_MCMC(self):
        plt.hist2d(self.mcmc_chain[:,1],self.mcmc_chain[:,0],bins=20,range=[[0, self.room_dims[1]], [0, self.audience_dims[0]]])
        plt.plot(self.true_x0[1],self.true_x0[0],'r*')
        plt.show()
        
    def plot_MCMC_samples(self):
        
        samples_xy = self.mcmc_chain.T[:2]
        samples_z = [self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*samples_xy[0]]
        samples = np.r_[samples_xy,samples_z]
        alpha = 0.0075
        
        fig = plt.figure(figsize=(17,5))

        ax = fig.add_subplot(131, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(0, -90)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)
        
        
        ax = fig.add_subplot(132, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(45, 45)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)

        ax = fig.add_subplot(133, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(90, 0)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)
        plt.tight_layout()
        plt.show()

    def plot_histograms(self):
        alpha=0.3
        n_bin=20
        plt.figure(figsize=(10,4))
        plt.subplot(2,5,1)
        plt.hist(np.random.uniform(0,self.audience_dims[0],len(self.mcmc_chain)),density=True,color='red',alpha=0.1,bins=n_bin)
        plt.hist(self.mcmc_chain[:,0],density=True,color='blue',alpha=alpha,bins=n_bin)
        plt.xlim([0,self.audience_dims[0]])
        plt.title(r'$x_0$')
        plt.axvline(x=self.true_x0[0],color='r',linestyle='--')
        
        plt.subplot(2,5,2)
        plt.hist(np.random.uniform(0,self.room_dims[1],len(self.mcmc_chain)),density=True,color='red',alpha=0.1,bins=n_bin)
        plt.hist(self.mcmc_chain[:,1],density=True,color='blue',alpha=alpha,bins=n_bin)
        plt.title(r'$y_0$')
        plt.xlim([0,self.room_dims[1]])
        plt.axvline(x=self.true_x0[1],color='r',linestyle='--')
        
        plt.subplot(2,5,3)
        plt.hist(np.random.normal(self.true_u[0],self.sigmax,len(self.mcmc_chain)),density=True,color='red',alpha=0.1,bins=n_bin)
        plt.hist(self.mcmc_chain[:,2],density=True,color='blue',alpha=alpha,bins=n_bin)
        plt.title(r'$u_x$')
        plt.xlim([self.true_u[0]-2*self.sigmax,self.true_u[0]+2*self.sigmax])
        plt.axvline(x=self.true_u[0],color='r',linestyle='--')
        
        plt.subplot(2,5,4)
        plt.hist(np.random.normal(self.true_u[1],self.sigmay,len(self.mcmc_chain)),density=True,color='red',alpha=0.1,bins=n_bin)
        plt.hist(self.mcmc_chain[:,3],density=True,color='blue',alpha=alpha,bins=n_bin)
        plt.title(r'$u_y$')
        plt.xlim([self.true_u[1]-2*self.sigmay,self.true_u[1]+2*self.sigmay])
        plt.axvline(x=self.true_u[1],color='r',linestyle='--')
        
        plt.subplot(2,5,5)
        plt.hist(np.random.normal(self.true_u[2],self.sigmaz,len(self.mcmc_chain)),density=True,color='red',alpha=0.1,bins=n_bin)
        plt.hist(self.mcmc_chain[:,4],density=True,color='blue',alpha=alpha,bins=n_bin)
        plt.title(r'$u_z$')
        plt.xlim([self.true_u[2]-2*self.sigmaz,self.true_u[2]+2*self.sigmaz])
        plt.axvline(x=self.true_u[2],color='r',linestyle='--')

        plt.subplot(2,5,6)
        plt.plot(self.mcmc_chain[:,0])
        
        plt.subplot(2,5,7)
        plt.plot(self.mcmc_chain[:,1])
        
        plt.subplot(2,5,8)
        plt.plot(self.mcmc_chain[:,2])
        
        plt.subplot(2,5,9)
        plt.plot(self.mcmc_chain[:,3])
        
        plt.subplot(2,5,10)
        plt.plot(self.mcmc_chain[:,4])
        
        plt.tight_layout()
        plt.show()
        
    def plot_sample_trajectory(self,i):
        
        # random feature
        # rand_numbers = np.random.choice(range(0,len(mcmc)),20)
        
        traj = tomato.plot_trajectory(mcmc[i,:2],mcmc[i,2:])
        
    def push_forward(self):
        return [self.fwd_map(self.mcmc_chain[i,:2], self.mcmc_chain[i,2:]) for i in range(len(self.mcmc_chain))]
    
    def plot_MCMC_samples_fwd(self):
        fwd_pushes = np.array(self.push_forward())
        
        samples_xy = self.mcmc_chain.T[:2]
        samples_z = [self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*samples_xy[0]]
        samples = np.r_[samples_xy,samples_z]
        alpha = 0.0075
        
        fig = plt.figure(figsize=(17,5))

        ax = fig.add_subplot(131, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(0, -90)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)
        ax.scatter(*fwd_pushes.T[:3], color = 'black',s=3,alpha=alpha)

        
        ax = fig.add_subplot(132, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10])
        ax.view_init(45, 45)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)
        ax.scatter(*fwd_pushes.T[:3], color = 'black',s=3,alpha=alpha)

        ax = fig.add_subplot(133, projection = '3d')
        surf = ax.plot_surface(self.audience[0],
                               self.audience[1],
                               self.audience[2],alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0,self.room_dims[0]])
        ax.set_ylim([0,self.room_dims[1]])
        ax.set_zlim([0,self.room_dims[2]])
        ax.set_xticks([0,10,20,30])
        ax.set_yticks([0,10,20,30])
        ax.set_zticks([0,5,10,30])
        ax.view_init(90, 0)
        ax.scatter(*self.speaker_loc, color = 'blue',s=70)
        ax.scatter(*[self.speaker_loc[0],self.speaker_loc[1],0], color = 'black',s=3)
        ax.scatter(*samples, color = 'black',alpha=alpha,s=3)
        ax.scatter(*[self.true_x0[0],self.true_x0[1],self.audience_dims[1]-(self.audience_dims[1]/self.audience_dims[0])*self.true_x0[0]], color = 'red',s=70)
        ax.scatter(*fwd_pushes.T[:3], color = 'black',s=3,alpha=alpha)
        plt.tight_layout()
        plt.show()

