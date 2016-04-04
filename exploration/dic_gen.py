"""
================================================================================                   
    Currency Fair Project - Part 2 - Parameter Dictionary Generation
================================================================================                   
Generates a bunch of parameter dictionaries for use in the program
 marketplace_predictions.py 

The chosen parameters to vary are:
    - RATE_TYPE : constant, sine, gbm
    - N_TYPE    : fixed, gaussian
    - N_EUR     : int
    - MU_EUR    : float
    - SD_EUR    : float
Rate / Type specific:
    - SINE_PERIOD  : float
    - RATES_STD    : float
================================================================================                   
"""
import shelve



set_1 = {'RATE_TYPE' : 'constant',
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 100,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_2 = {'RATE_TYPE' : 'sine',
         'SINE_PERIOD' : 30,
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 100,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_3 = {'RATE_TYPE' : 'gbm',
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 100,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_4 = {'RATE_TYPE' : 'constant',
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 120,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_5 = {'RATE_TYPE' : 'sine',
         'SINE_PERIOD' : 30,
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 120,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_6 = {'RATE_TYPE' : 'gbm',
         'N_TYPE'    : 'fixed',
         'N_EUR'     : 120,
         'MU_EUR'    : 9.0,
         'SD_EUR'    : 1.0 }

set_7 = {'RATE_TYPE' : 'gbm',
         'N_TYPE'    : 'gaussian',
         'N_STD'     : 20,
         'N_EUR'     : 120,
         'MU_EUR'    : 9.5,
         'SD_EUR'    : 0.5 }

def default_dic():
    
    dic = {}
    dic['N_STD'] = 10      
    # GBM params #                                                               
    dic['DT'] = 0.1   # Time step (days)                                         
    dic['MU'] = 0.00025  # drift factor                                          
    dic['SIGMA'] = 0.05 # volatility                                             
    # STD ratio of Gaussian selection of seller rates. Large vals = small dist.  
    dic['RATES_STD'] = 100.0 # STD DEV = RATES/RATES_STD                         
    dic['N_USD'] = 100  ;     dic['MU_USD'] = 9.0  ;    dic['SD_USD'] = 1.0                                                          
    dic['DAYS_CUTOFF'] = 7 ; dic['ITERS'] = 300 ;  dic['INITIAL_PRICE'] = 1.0 

    return dic

for num , ps in enumerate([set_1,set_2,set_3,set_4,set_5,set_6,set_7],1):

    dout = shelve.open('./param_dics/'+\
        str(num)+'_parameter_set.db', writeback = True)

    d = default_dic()
    
    for k, v in ps.items(): d[k] = v
    for k,v in d.items():   dout[k] = v

    dout.sync() 
    dout.close()
