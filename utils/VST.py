import numpy as np

def Anscombe_forward_GP(z, alpha, sigma):
    
    fz = (2/alpha) * np.sqrt(alpha*z + (3/8)*alpha**2 + sigma**2)
    return fz

def Anscombe_forward(z):
    
    return 2*np.sqrt(z + 3/8)

def Anscombe_inverse_asympt_unbiased(D):
    asymtotic = (D/2)**2 - 1/8
    asymtotic[D < 2*np.sqrt(3/8)] = 0
    
    return asymtotic

def Anscombe_inverse_closed_form(D):
    
    exact_inverse = (D/2)**2 + np.sqrt(3/2)*(1/4)*(D**(-1)) - (11/8)*(D**(-2)) + (5/8)*np.sqrt(3/2)*(D**(-3)) - 1/8
    
    return exact_inverse

    
