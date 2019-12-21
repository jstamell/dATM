# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:15:02 2019

@author: jakes
"""
### This file runs the algorithm ###

#%%
import os
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, gamma, poisson
from scipy.special import digamma, loggamma
import seaborn as sns
import matplotlib as plt
import datetime
from scipy import sparse
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#%%
os.chdir('C:\\Users\\jakes\\Documents\\Programming_projects\\CU_F19\\graph\\Project')

#%% Helper functions for algorithm

def E_log_g(a,b):
    return digamma(a) - np.log(b)

def E_g(a,b):
    return a/b

def update_phi(omega_s, omega_r, kappa_s, kappa_r, x):
    N_sub = omega_s.shape[0]
    V_sub = kappa_s.shape[0]
    
    phi_part_a = E_log_g(omega_s,omega_r)
    phi_part_b = E_log_g(kappa_s,kappa_r)
    
    phi = np.repeat(np.expand_dims(phi_part_a,axis=1), repeats=V_sub,axis=1) \
        + np.repeat(np.expand_dims(phi_part_b,axis=0), repeats=N_sub,axis=0)
    
    phi_exp = np.exp(phi)
    
    phi_norm = phi_exp  / np.sum(phi_exp ,axis=2,keepdims=True)
    z = phi_norm * np.expand_dims(x,axis=2)
    nzi = np.nonzero(z)
    
    ELBO_contribution = np.sum( z[nzi[0],nzi[1],nzi[2]]*(phi[nzi[0],nzi[1],nzi[2]] - np.log(phi_norm[nzi[0],nzi[1],nzi[2]])) ) - np.sum( np.repeat(np.expand_dims(omega_s/omega_r,axis=1), repeats=V_sub,axis=1) \
        * np.repeat(np.expand_dims(kappa_s/kappa_r,axis=0), repeats=N_sub,axis=0) )
    return z, ELBO_contribution
    
def update_mu(a_mu, z, tau_s, tau_r, kappa_s, kappa_r):
#    shape = a_mu + np.sum(z, axis=1)
#    rate = a_mu * (tau_s/tau_r) + np.sum(kappa_s/kappa_r, axis=0, keepdims=True)
#    return shape, rate
    return (a_mu + np.sum(z, axis=1)), (a_mu * (tau_s/tau_r) + np.sum(kappa_s/kappa_r, axis=0, keepdims=True))

def update_rho(a_rho, a_mu, nu_s, nu_r, omega_s, omega_r):
#    shape = a_rho + a_mu
#    rate = a_rho * (nu_s/nu_r) + a_mu * (omega_s/omega_r)
#    return shape, rate
    return (a_rho * (nu_s/nu_r) + a_mu * (omega_s/omega_r))

def update_beta(a_beta, a_gamma, z, eta_s, eta_r, omega_s, omega_r):
#    shape = a_beta + a_gamma + np.sum(z, axis=0)
#    rate = a_beta * (eta_s[:,:,0]/eta_r[:,:,0]) + a_gamma * (eta_s[:,:,1]/eta_r[:,:,1]) + np.sum(omega_s/omega_r, axis=0, keepdims=True)
#    return shape, rate
    return (a_beta + a_gamma + np.sum(z, axis=0)), (a_beta * (eta_s[:,:,0]/eta_r[:,:,0]) + a_gamma * (eta_s[:,:,1]/eta_r[:,:,1]) + np.sum(omega_s/omega_r, axis=0, keepdims=True))

def update_beta_first(a_beta, b_beta, a_gamma, z, eta_s, eta_r, omega_s, omega_r):
#    shape = a_beta + a_gamma + np.sum(z, axis=0)
#    rate = a_beta * b_beta + a_gamma * (eta_s/eta_r) + np.sum(omega_s/omega_r, axis=0, keepdims=True)
#    return shape, rate
    return (a_beta + a_gamma + np.sum(z, axis=0)), (a_beta * b_beta + a_gamma * (eta_s/eta_r) + np.sum(omega_s/omega_r, axis=0, keepdims=True))

def update_beta_last(a_beta, z, eta_s, eta_r, omega_s, omega_r):
#    shape = a_beta + np.sum(z, axis=0)
#    rate = a_beta * (eta_s/eta_r) + np.sum(omega_s/omega_r, axis=0, keepdims=True)
#    return shape, rate
    return (a_beta + np.sum(z, axis=0)), (a_beta * (eta_s/eta_r) + np.sum(omega_s/omega_r, axis=0, keepdims=True))

def update_gamma(a_gamma, a_beta, kappa_s, kappa_r):
#    shape = a_gamma + a_beta
#    rate = a_gamma * (kappa_s[:,:,0]/kappa_r[:,:,0]) + a_beta * (kappa_s[:,:,1]/kappa_r[:,:,1])
#    return shape, rate
    return (a_gamma * (kappa_s[:,:,0]/kappa_r[:,:,0]) + a_beta * (kappa_s[:,:,1]/kappa_r[:,:,1]))

def update_theta(a_theta, b_theta, a_rho, rho_a):
#    shape = a_theta + a_rho * N_a # N_a removed from args for now
#    rate = a_theta * b_theta + a_rho * rho_a
#    return shape, rate
    return (a_theta * b_theta + a_rho * rho_a)

#%% Reading in data

#train_info = pd.read_csv('./data/train-track-index.csv',index_col=0)
train_meta = pd.read_csv('./data/train-track-meta.csv',index_col=0)
word_index = np.load('./data/word-index.npy')
w_sparse = sparse.load_npz('./data/word-counts-sparse.npz')

#%% Removing stop words
vocab_list = pd.DataFrame(word_index)
vocab_list.rename(columns={0:"vocab"},inplace=True)
vocab_list["stop_word"] = 0
stops = set(stopwords.words("english"))
for i in range(len(vocab_list)):
    if vocab_list.loc[i,"vocab"] in stops:
        vocab_list.loc[i,"stop_word"] = 1
        
stop_mask = vocab_list['stop_word'] == 0

#%% Selecting time period
year_start = 1970
year_end = 1995
interval = 1
T = int((year_end-year_start+1)/interval)

song_year_mask = np.logical_and(train_meta.year >= year_start, train_meta.year <= year_end)
song_year_mask_num = song_year_mask[song_year_mask].index


#%% Removing  low frequency words
word_freq = np.array(w_sparse[song_year_mask_num,:].sum(axis=0)).flatten()

#word_freq.min()
#np.median(word_freq)
#np.sum(word_freq > 500)
#word_index[word_freq <= 50]

word_freq_mask =  word_freq > 500
vocab_mask = np.logical_and(stop_mask, word_freq_mask)
vocab_mask_num = vocab_mask[vocab_mask].index

#%% Removing songs with too many or too few words
lower_lim = 50
upper_lim = 1000

song_word_counts = np.array(w_sparse[:,vocab_mask_num].sum(axis=1)).flatten()

song_word_mask = np.logical_and(song_word_counts>=100,song_word_counts<1000)
song_mask = np.logical_and(song_year_mask,song_word_mask)
song_mask_num = song_mask[song_mask].index

artists = train_meta[song_mask].artist_id.value_counts().rename_axis('artist_id').reset_index(name='num_songs')
artists['a_num'] = artists.index

song_meta = train_meta.loc[song_mask, ['track_id','artist_id','year']].merge(artists)
song_meta["t"] = np.floor((song_meta.year - year_start)/interval).astype(int)
small_sparse = w_sparse[song_mask_num,:][:,vocab_mask_num]

#%% Algorithm function

def dyn_MF(small_sparse,song_meta,artists,T,num_topics=10,seed=10,max_its=1000,tol=1):
    np.random.seed(seed)
    ### Reference numbers
    V = small_sparse.shape[1]
    C = num_topics
    A = len(artists)
    N = small_sparse.shape[0]
    
    trunc_norm_var = 2

    ### Priors
    a_beta = 10
    b_beta = .1
    a_gamma = 50
    a_theta = 1
    b_theta = .1
    a_mu = 10
    a_rho = 50

    ### Initializing arrays
    #E_z = np.empty(shape=(N,V,C), dtype=float)
    rho_artist = np.empty(shape=(A,C), dtype=float)
    
    kappa_s = a_beta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(V,C,T))
    kappa_r = a_beta*b_beta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(V,C,T))
    
    eta_s = a_gamma + truncnorm.rvs(a=0,b=trunc_norm_var, size=(V,C,T-1))
    eta_r = a_gamma*(kappa_s[:,:,:(T-1)]/kappa_r[:,:,:(T-1)])  + truncnorm.rvs(a=0,b=trunc_norm_var, size=(V,C,T-1))
    
    nu_s = a_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(A,1))
    nu_r = a_theta*b_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(A,C))
    
    tau_s = a_rho + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    tau_r = a_rho/b_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    
    omega_s = a_mu + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    omega_r = a_mu*(tau_s/tau_r) + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    
    tracking_values = pd.DataFrame(columns=["ELBO"], dtype="float")

    print(datetime.datetime.today().time())
    
    # Precomputing some updates that are fixed
    tau_s[:,:] = a_rho + a_mu
    eta_s[:,:,:] = a_gamma + a_beta
    nu_s[:,0] = a_theta + a_rho * artists.num_songs.to_numpy()
    

    ### Iterations
    old_ELBO = float('-inf')
    for e in range(max_its):
        ELBO = 0
    #    print(e)
        for t in reversed(range(T)):
            t_mask = song_meta.t==t
            t_mask_num = t_mask[t_mask].index
            artist_to_song = song_meta[t_mask].a_num
            x = small_sparse[t_mask_num,:].toarray()
            
            E_z, ELBO_contribution = update_phi(omega_s[t_mask,:], omega_r[t_mask,:], kappa_s[:,:,t], kappa_r[:,:,t], x)
            omega_s[t_mask,:], omega_r[t_mask,:] = update_mu(a_mu, E_z, tau_s[t_mask,:], tau_r[t_mask,:], kappa_s[:,:,t], kappa_r[:,:,t])
            tau_r[t_mask,:] = update_rho(a_rho, a_mu, nu_s[artist_to_song,:], nu_r[artist_to_song,:], omega_s[t_mask,:], omega_r[t_mask,:])
            
            if t == 0:
                kappa_s[:,:,t], kappa_r[:,:,t] = update_beta_first(a_beta, b_beta, a_gamma, E_z, eta_s[:,:,t], eta_r[:,:,t], omega_s[t_mask,:], omega_r[t_mask,:])
                eta_r[:,:,t] = update_gamma(a_gamma, a_beta, kappa_s[:,:,t:(t+2)], kappa_r[:,:,t:(t+2)])
            elif t == (T-1):
                kappa_s[:,:,t], kappa_r[:,:,t] = update_beta_last(a_beta, E_z, eta_s[:,:,t-1], eta_r[:,:,t-1], omega_s[t_mask,:], omega_r[t_mask,:])
            else:
                kappa_s[:,:,t], kappa_r[:,:,t] = update_beta(a_beta, a_gamma, E_z, eta_s[:,:,(t-1):(t+1)], eta_r[:,:,(t-1):(t+1)], omega_s[t_mask,:], omega_r[t_mask,:])
                eta_r[:,:,t] = update_gamma(a_gamma, a_beta, kappa_s[:,:,t:(t+2)], kappa_r[:,:,t:(t+2)])
            
            ELBO += ELBO_contribution
    
        E_rho = tau_s/tau_r
        for a in range(A):
            artist_mask = (song_meta.a_num == a)
            rho_artist[a,:] = np.sum(E_rho[artist_mask,:],axis=0)
    
        nu_r[:,:] = update_theta(a_theta, b_theta, a_rho, rho_artist[:,:])
        
        # Calculate ELBO
        if e%5==0:
            loggamma_terms = np.sum( loggamma(nu_s) ) + np.sum( loggamma(tau_s) ) + np.sum( loggamma(omega_s) ) + np.sum( loggamma(eta_s) ) + np.sum( loggamma(kappa_s) )
            slogr_terms = - np.sum( nu_s*np.log(nu_r) ) - np.sum( tau_s*np.log(tau_r) ) - np.sum( omega_s*np.log(omega_r) ) - np.sum( eta_s*np.log(eta_r) ) - np.sum( kappa_s*np.log(kappa_r) ) 
            other_theta = np.sum( (nu_r-a_theta*b_theta)*E_g(nu_s,nu_r) )
            other_rho = np.sum( (tau_r-a_rho*E_g(nu_s,nu_r)[song_meta.a_num,:])*E_g(tau_s,tau_r) )
            other_mu = np.sum( (a_mu-omega_s)*E_log_g(omega_s,omega_r) ) + np.sum( (omega_r-a_mu*E_g(tau_s,tau_r))*E_g(omega_s,omega_r) )
            other_gamma = + np.sum( (eta_r-a_gamma*E_g(kappa_s[:,:,:(T-1)],kappa_r[:,:,:(T-1)]))*E_g(eta_s,eta_r) )
            other_beta = np.sum( (kappa_r[:,:,0]-a_beta*b_beta)*E_g(kappa_s[:,:,0],kappa_r[:,:,0]) ) + np.sum( (kappa_r[:,:,1:]-a_beta*E_g(eta_s,eta_r))*E_g(kappa_s[:,:,1:],kappa_r[:,:,1:]) ) 
            combined_gamma_beta = np.sum( (a_beta-kappa_s+a_gamma)*E_log_g(kappa_s,kappa_r) ) - a_gamma*np.sum(E_log_g(kappa_s[:,:,T-1],kappa_r[:,:,T-1]))
            ELBO += loggamma_terms + slogr_terms + other_theta + other_rho + other_mu + other_gamma + other_beta + combined_gamma_beta
            tracking_values = tracking_values.append({"ELBO":ELBO},ignore_index=True)
            
            print(C,e,ELBO - old_ELBO)
            if ELBO - old_ELBO < tol: break
            old_ELBO = ELBO
    
    print(datetime.datetime.today().time())
    return tracking_values, kappa_s, kappa_r, eta_s, eta_r, nu_s, nu_r, tau_s, tau_r, omega_s, omega_r
    
#%% Running algorithm
    
#test_10 = dyn_MF(small_sparse,song_meta,artists,T,num_topics=10,seed=10,max_its=1000,tol=1)
test_10_2 = dyn_MF(small_sparse,song_meta,artists,T,num_topics=10,seed=42,max_its=1000,tol=500)
test_10_3 = dyn_MF(small_sparse,song_meta,artists,T,num_topics=10,seed=23,max_its=1000,tol=200)
test_20 = dyn_MF(small_sparse,song_meta,artists,T,num_topics=20,seed=27,max_its=150,tol=500)
#test_30 = dyn_MF(small_sparse,song_meta,artists,T,num_topics=30,seed=2457,max_its=1000,tol=500)

#%% Visualizing ELBO
tracking_values = test_10_3[0]
tracking_values2 = test_10_2[0]
sns.lineplot(x=tracking_values.index, y=tracking_values.ELBO);
sns.lineplot(x=tracking_values2.index, y=tracking_values2.ELBO);
sns.lineplot(x=tracking_values.index[10:], y=tracking_values.ELBO[10:]);
sns.lineplot(x=tracking_values2.index[10:], y=tracking_values2.ELBO[10:]);

tracking_values20 = test_20[0]
sns.lineplot(x=tracking_values20.index*5, y=tracking_values20.ELBO);
sns.lineplot(x=(tracking_values20.index*5)[1:], y=tracking_values20.ELBO[1:]);

#%% Algorithm function (test data)

def dyn_MF_test(train_output,small_sparse,song_meta,artists,T,num_topics=10,seed=10,max_its=1000,tol=1):
    np.random.seed(seed)
    ### Reference numbers
    V = small_sparse.shape[1]
    C = num_topics
    A = len(artists)
    N = small_sparse.shape[0]
    
    trunc_norm_var = 2

    ### Priors
    a_beta = 10
    b_beta = .1
    a_gamma = 50
    a_theta = 1
    b_theta = .1
    a_mu = 10
    a_rho = 50

    ### Initializing arrays
    #E_z = np.empty(shape=(N,V,C), dtype=float)
    rho_artist = np.empty(shape=(A,C), dtype=float)
    
    kappa_s = train_output[1]
    kappa_r = train_output[2]
    
    eta_s = train_output[3]
    eta_r = train_output[4]
    
    nu_s = a_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(A,1))
    nu_r = a_theta*b_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(A,C))
    
    tau_s = a_rho + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    tau_r = a_rho/b_theta + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    
    omega_s = a_mu + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    omega_r = a_mu*(tau_s/tau_r) + truncnorm.rvs(a=0,b=trunc_norm_var, size=(N,C))
    
    tracking_values = pd.DataFrame(columns=["ELBO"], dtype="float")

    print(datetime.datetime.today().time())
    
    # Precomputing some updates that are fixed
    tau_s[:,:] = a_rho + a_mu
    nu_s[:,0] = a_theta + a_rho * artists.num_songs.to_numpy()
    

    ### Iterations
    old_ELBO = float('-inf')
    for e in range(max_its):
        ELBO = 0
    #    print(e)
        for t in reversed(range(T)):
            t_mask = song_meta.t==t
            t_mask_num = t_mask[t_mask].index
            artist_to_song = song_meta[t_mask].a_num
            x = small_sparse[t_mask_num,:]
            
            E_z, ELBO_contribution = update_phi(omega_s[t_mask,:], omega_r[t_mask,:], kappa_s[:,:,t], kappa_r[:,:,t], x)
            omega_s[t_mask,:], omega_r[t_mask,:] = update_mu(a_mu, E_z, tau_s[t_mask,:], tau_r[t_mask,:], kappa_s[:,:,t], kappa_r[:,:,t])
            tau_r[t_mask,:] = update_rho(a_rho, a_mu, nu_s[artist_to_song,:], nu_r[artist_to_song,:], omega_s[t_mask,:], omega_r[t_mask,:])
                     
            ELBO += ELBO_contribution
    
        E_rho = tau_s/tau_r
        for a in range(A):
            artist_mask = (song_meta.a_num == a)
            rho_artist[a,:] = np.sum(E_rho[artist_mask,:],axis=0)
    
        nu_r[:,:] = update_theta(a_theta, b_theta, a_rho, rho_artist[:,:])
        
        # Calculate ELBO
        if e%2==0:
            loggamma_terms = np.sum( loggamma(nu_s) ) + np.sum( loggamma(tau_s) ) + np.sum( loggamma(omega_s) )
            slogr_terms = - np.sum( nu_s*np.log(nu_r) ) - np.sum( tau_s*np.log(tau_r) ) - np.sum( omega_s*np.log(omega_r) )
            other_theta = np.sum( (nu_r-a_theta*b_theta)*E_g(nu_s,nu_r) )
            other_rho = np.sum( (tau_r-a_rho*E_g(nu_s,nu_r)[song_meta.a_num,:])*E_g(tau_s,tau_r) )
            other_mu = np.sum( (a_mu-omega_s)*E_log_g(omega_s,omega_r) ) + np.sum( (omega_r-a_mu*E_g(tau_s,tau_r))*E_g(omega_s,omega_r) )
            ELBO += loggamma_terms + slogr_terms + other_theta + other_rho + other_mu
            tracking_values = tracking_values.append({"ELBO":ELBO},ignore_index=True)
            
            print(C,e,ELBO)
#            print(C,e,ELBO - old_ELBO)
            if ELBO - old_ELBO < tol: break
            old_ELBO = ELBO
    
    print(datetime.datetime.today().time())
    return tracking_values, kappa_s, kappa_r, eta_s, eta_r, nu_s, nu_r, tau_s, tau_r, omega_s, omega_r

#%% Reading in data

test_meta = pd.read_csv('./data/test-track-meta.csv',index_col=0)
test_sparse = sparse.load_npz('./data/test-counts-sparse.npz')


#%% Selecting time period
test_year_mask = np.logical_and(test_meta.year >= year_start, test_meta.year <= year_end)
test_year_mask_num = test_year_mask[test_year_mask].index

#%% Removing songs with too many or too few words

song_word_counts = np.array(test_sparse[:,vocab_mask_num].sum(axis=1)).flatten()

song_word_mask = np.logical_and(song_word_counts>=100,song_word_counts<1000)
song_mask = np.logical_and(test_year_mask,song_word_mask)
song_mask_num = song_mask[song_mask].index

artists_test = test_meta[song_mask].artist_id.value_counts().rename_axis('artist_id').reset_index(name='num_songs')
artists_test['a_num'] = artists_test.index

test_song_meta = test_meta.loc[song_mask, ['track_id','artist_id','year']].merge(artists_test)
test_song_meta["t"] = np.floor((test_song_meta.year - year_start)/interval).astype(int)
test_small_sparse = test_sparse[song_mask_num,:][:,vocab_mask_num]

#%% Dividing the test data in 2 parts
w_test_in = np.copy(test_small_sparse.toarray())
w_test_out = np.copy(test_small_sparse.toarray())
w_test_out_bool = np.zeros(shape=w_test_out.shape, dtype=bool)
w_index = np.array(range(w_test_out.shape[1]))

for i in range(w_test_in.shape[0]):
    w_in, w_out = train_test_split(w_index, stratify= w_test_in[i,:] > 0, test_size=.5, random_state = i)
    w_test_in[i,w_in] = 0
    w_test_out[i,w_out] = 0
    w_test_out_bool[i,w_in] = True

#%% Running on test data

test_output_10 = dyn_MF_test(test_10_3,w_test_in,test_song_meta,artists_test,T,num_topics=10,seed=936,max_its=1000,tol=200)
test_output_20 = dyn_MF_test(test_20,w_test_in,test_song_meta,artists_test,T,num_topics=20,seed=1074,max_its=1000,tol=200)

#%% Sections below generate charts for the report

#%% Checking convergence
chain_code = ["a","b"]

ELBO_1 = test_10_2[0].copy()
ELBO_1["chain"] = "a"
ELBO_1["iteration"] = ELBO_1.index

ELBO_2 = test_10_3[0].copy()
ELBO_2["chain"] = "b"
ELBO_2["iteration"] = ELBO_2.index

combined_ELBO = pd.DataFrame(columns=["ELBO","chain","iteration"])
combined_ELBO = combined_ELBO.append(ELBO_1,ignore_index=True)
combined_ELBO = combined_ELBO.append(ELBO_2,ignore_index=True)
      
combined_ELBO.melt(id_vars=["ELBO","chain"])
ax = sns.lineplot(x="iteration",y="ELBO",hue="chain", data=combined_ELBO.loc[combined_ELBO.iteration>0])
ax.set(ylabel="ELBO");
ax.get_figure().savefig("1_ELBO.png", format="png")

#%% Log predictive likelihood

pred_lik = np.empty(shape=(2), dtype="float")

mu_pred_20 = gamma.rvs(a=test_output_20[9], scale= 1/ test_output_20[10])
beta_pred_20 = gamma.rvs(a=test_output_20[1], scale= 1/ test_output_20[2])

dot_prod = np.empty(shape=(mu_pred_20.shape[0],beta_pred_20.shape[0]))
for i in range(mu_pred_20.shape[0]):
    t = test_song_meta.t[i]
    for j in range(beta_pred_20.shape[0]):
        dot_prod[i,j] = np.sum(mu_pred_20[i,:] * beta_pred_20[j,:,t])
    
pred_lik[1] = np.sum(poisson.logpmf(w_test_out, dot_prod))
    
    
mu_pred_10 = gamma.rvs(a=test_output_10[9], scale= 1/ test_output_10[10])
beta_pred_10 = gamma.rvs(a=test_output_10[1], scale= 1/ test_output_10[2])

dot_prod = np.empty(shape=(mu_pred_10.shape[0],beta_pred_10.shape[0]))
for i in range(mu_pred_10.shape[0]):
    t = test_song_meta.t[i]
    for j in range(beta_pred_10.shape[0]):
        dot_prod[i,j] = np.sum(mu_pred_10[i,:] * beta_pred_10[j,:,t])
    
pred_lik[0] = np.sum(poisson.logpmf(w_test_out, dot_prod))

ax = sns.barplot(x = [10,20], y = pred_lik)
ax.set(xlabel="Number of topics", ylabel="Log predictive likelihood");
ax.get_figure().savefig("2_pred-log-lik.png", format="png")
    
#%% Precision recall function
def prec_rec(test_data, test_song_meta, vocab_list, vocab_mask, w_test_out_bool, w_test_out, top_M=20):
    
    mu_val = test_data[9]/test_data[10]
    beta_val = test_data[1]/test_data[2]
    article_word_rates = np.empty(shape=(mu_val.shape[0],beta_val.shape[0]))
    for i in range(mu_val.shape[0]):
        t = test_song_meta.t[i]
        for j in range(beta_val.shape[0]):
            dot_prod[i,j] = np.sum(mu_pred_10[i,:] * beta_pred_10[j,:,t])
    
    output = pd.DataFrame(columns=["recall","precision"])
    
    for i in range(article_word_rates.shape[0]):
        vocab_words = vocab_list.loc[vocab_mask,['vocab']].copy()
        vocab_words["rates"] = article_word_rates[i,:]
        top_M_words = vocab_words[w_test_out_bool[i,:]].sort_values(by="rates",ascending=False).iloc[0:top_M,0]
        x_out_words_mask = np.logical_and(w_test_out_bool[i,:], w_test_out[i,:] > 0)
        words_in_article = vocab_list.loc[vocab_mask,['vocab']].loc[x_out_words_mask,"vocab"]
        
        overlap = len(set(top_M_words).intersection(set(words_in_article)))
        adj_precision = overlap / min(words_in_article.size,top_M)
        recall = overlap / len(words_in_article)
        output = output.append({"recall":recall, "precision":adj_precision},ignore_index=True)
    
    return output

#%% Calculating precision and recall
    
prec_rec_10 = prec_rec(test_output_10, test_song_meta, vocab_list, vocab_mask, w_test_out_bool, w_test_out, top_M=20)
prec_rec_10["topics"] = 10

prec_rec_20 = prec_rec(test_output_20, test_song_meta, vocab_list, vocab_mask, w_test_out_bool, w_test_out, top_M=20)
prec_rec_20["topics"] = 20

prec_rec_by_topics = pd.DataFrame(columns=["recall","precision","topics"])
prec_rec_by_topics = prec_rec_by_topics.append(prec_rec_10,ignore_index=True)
prec_rec_by_topics = prec_rec_by_topics.append(prec_rec_20,ignore_index=True)

prec_rec_by_topics_long = pd.melt(prec_rec_by_topics, id_vars="topics")

ax = sns.boxplot(x="topics", y="value", data=prec_rec_by_topics_long[prec_rec_by_topics_long.variable=="recall"])
ax.set(xlabel="Number of topics", ylabel="Recall at 20 recommendations");
ax.get_figure().savefig("3_recall-topics.png", format="png")

ax = sns.boxplot(x="topics", y="value", data=prec_rec_by_topics_long[prec_rec_by_topics_long.variable=="precision"])
ax.set(xlabel="Number of topics", ylabel="Adj precision at 20 recommendations");
ax.get_figure().savefig("4_precision-topics.png", format="png")

#%% Top 10 words for each topic in last slice

time_per = 25
beta = test_10_2[1]/test_10_2[2]
top_10 = pd.DataFrame()
for i in range(10):
    test = pd.DataFrame(beta[:,i,time_per], word_index[vocab_mask])
    top_10[i] = test.sort_values(by=0, ascending=False)[0:10].index

top_10.to_csv("top_10_words_by_topic.csv")

#%% Top 10 words from love topic 

topic_sel = 0
topic_over_time = pd.DataFrame()
for i in range(0,T,5):
    test = pd.DataFrame(beta[:,topic_sel,i], word_index[vocab_mask])
    topic_over_time[i] = test.sort_values(by=0, ascending=False)[0:10].index
    
topic_over_time.to_csv("top_10_words_over_time.csv")

#%% Looking at specific words over time

# Love
np.where(word_index[vocab_mask] == "love")
word_df = pd.DataFrame(columns = ["intensity","year","topic"])
for i in range(10):
    interim = pd.DataFrame(beta[0,i,:],columns = ["intensity"])
    interim["year"] = list(range(0,26))
    interim["topic"] = "topic " + str(i)
    word_df = word_df.append(interim, ignore_index=True)

ax = sns.lineplot(x="year", y="intensity", hue="topic", data=word_df)
ax.set(xlabel="Year (indexed)", ylabel="Intensity (expected beta value)");
ax.get_figure().savefig("7_love-over-time.png", format="png")

# Funk
np.where(word_index[vocab_mask] == "funk")
word_df = pd.DataFrame(columns = ["intensity","year","topic"])
for i in range(10):
    interim = pd.DataFrame(beta[1365,i,:],columns = ["intensity"])
    interim["year"] = list(range(0,26))
    interim["topic"] = "topic " + str(i)
    word_df = word_df.append(interim, ignore_index=True)

ax = sns.lineplot(x="year", y="intensity", hue="topic", data=word_df)
ax.set(xlabel="Year (indexed)", ylabel="Intensity (expected beta value)");
ax.get_figure().savefig("8_funk-over-time.png", format="png")

#%% Total topic intensities

beta_sums = np.sum(beta, axis=0)

sns.barplot(x=list(range(10)), y=beta_sums[:,10])

prec_rec_by_topics_long = pd.melt(prec_rec_by_topics, id_vars="topics")

beta_sums_df = pd.DataFrame(beta_sums)
beta_sums_df["topic"] = beta_sums_df.index
beta_sums_long = pd.melt(beta_sums_df, id_vars="topic")

ax = sns.lineplot(x="variable",y="value",hue="topic",data=beta_sums_long)
ax.set(xlabel="Year (indexed)", ylabel="Total topic intensity (sum of expected beta values)");
ax.get_figure().savefig("9_total-topic-intensity.png", format="png")

#%% Average topic intensity by artist (not used in project)

theta = test_10_2[5]/test_10_2[6]

theta_sums = np.mean(theta, axis=0)

sns.barplot(x=list(range(10)),y=theta_sums)
