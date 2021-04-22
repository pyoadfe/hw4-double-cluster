#!/usr/bin/env python3


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    sigma1 = 0.5
    sigma2 = 0.6
    tau = 0.25
    mu1 = 0.5
    sigma1 = 0.2
    mu2 = 0.9
    sigma2 = 0.7
    n = 10000
    x1 = np.random.normal(mu1,sigma1,size=(int(n*tau)))
    x2 = np.random.normal(mu2,sigma2,size=(int(n*(1-tau))))
    x = np.concatenate([x1,x2])
    
    def sumlog(T_norm):
        return -np.sum(np.log(T_norm))

    sigma12 = sigma1**2
    sigma22 = sigma2**2
    T1 = (tau/np.sqrt(2*np.pi*sigma12)*np.exp(-0.5*(x-mu1)**2/sigma12))
    T2 = ((1-tau)/np.sqrt(2*np.pi*sigma22)*np.exp(-0.5*(x-mu2)**2/sigma22))
    T_norm = T1 + T2
    
    T_nov = -np.sum(np.log(T_norm))
    return T_nov 


x = np.array([5,3,8,7])
print(sp.optimize.minimize(lambda par: max_likelihood(x, *par), x0 = np.array([tau, mu1, mu2, sigma1, sigma2]),
                           bounds = [(0,0.99), (-400, 400), (-400, 400), (-400, 400), (-400, 400)]).x)   
    
    
  


    
def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    
    
    def normrasp(x, mu, sigma):
        return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2 / (2*sigma**2))
    
    
    def func(x, tau, mu1, sigma1, mu2, sigma2):
        return tau * normrasp(x, mu1, sigma1) + (1 - tau)*normrasp(x, mu2, sigma2)

    
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

x = np.array([0.99,0.1,6,4,8,4,3,3])
m_double_gauss(x, 0.4, 0, 0.2, 0, 1)





def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    pass


if __name__ == "__main__":
    pass
