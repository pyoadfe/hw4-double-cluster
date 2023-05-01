#!/usr/bin/env python3
%matplotlib inline
import numpy as np
import scipy as sp

sigma1 = 0.3
sigma2 = 0.7
tau = 0.4
mu1 = 0.6
sigma1 = 0.1
mu2 = 0.8
sigma2 = 0.6
n = 1000

x1 = np.random.normal(mu1,sigma1,size=(int(n*tau)))
x2 = np.random.normal(mu2,sigma2,size=(int(n*(1-tau))))
x = np.concatenate([x1,x2])

def sumlog(T_norm):
    return -np.sum(np.log(T_norm))

def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol = 1e-3):
    sigma12 = sigma1**2
    sigma22 = sigma2**2
    T1 = (tau/np.sqrt(2*np.pi*sigma12)*np.exp(-0.5*(x-mu1)**2/sigma12))
    T2 = ((1-tau)/np.sqrt(2*np.pi*sigma22)*np.exp(-0.5*(x-mu2)**2/sigma22))
    T_norm = T1 + T2
    
    T_new = -np.sum(np.log(T_norm))
    return T_new



x = np.array([5,3,8,7])
print(sp.optimize.minimize(lambda par: max_likelihood(x, *par), x0 = np.array([tau, mu1, mu2, sigma1, sigma2]),
                           bounds = [(0,0.99), (-400, 400), (-400, 400), (-400, 400), (-400, 400)]).x)











def normrasp(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2 / (2*sigma**2))



def func(x, tau, mu1, sigma1, mu2, sigma2):
    return tau * normrasp(x, mu1, sigma1) + (1 - tau)*normrasp(x, mu2, sigma2)




def m_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol = 1e-3):
    y = np.array([tau, mu1, sigma1, mu2, sigma2])
    while True:
        tau = np.sum(tau * normrasp(x, mu1, sigma1))/ np.sum(func(x, tau, mu1, sigma1, mu2, sigma2))
        mu1 = np.sum(x * normrasp(x, mu1, sigma1))/np.sum(normrasp(x, mu1, sigma1))
        mu2 = np.sum(x * normrasp(x, mu2, sigma2))/np.sum(normrasp(x, mu2, sigma2))
        sigma1 = np.sqrt(np.sum((x-sigma1)**2 * normrasp(x, mu1, sigma1))/np.sum(x * normrasp(x, mu1, sigma1)))
        sigma2 = np.sqrt(np.sum((x-sigma2)**2 * normrasp(x, mu2, sigma2))/np.sum(x * normrasp(x, mu2, sigma2)))
        ynov = np.array([tau, mu1, sigma1, mu2, sigma2])
        
        if np.linalg.norm(y-ynov) <= rtol:
            break
        y = ynov
        
    return y    






