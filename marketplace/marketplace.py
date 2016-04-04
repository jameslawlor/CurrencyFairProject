"""
================================================================================
            Currency Fair Project - Part 1 - Market simulation
================================================================================

This program simulates a peer-to-peer currency marketplace with a variable
exchange rate between the two currencies. Sellers are initialised with an amount
of currency and their own exchange rate which is then advertised at market.
Buyers in the opposite currency then use the marketplace and the currencies
are traded, with the most advantageous exchange rates chosen first.

The parameters which govern the marketplace and buyer/seller behaviour are
chosen in the topmost function set_params(). These are then fed into the main
function via a dictionary.

Below is a more in-depth description of:

    1. How the number of buyers and sellers is chosen and controlled
    2. How the sellers' exchange rates are chosen
    3. How the currency amounts for buyers and sellers are allocated
    4. Rules of the marketplace
    5. The characterisation of the underlying interbank FOREX rate.
    6. Outputs

The program will record the FOREX fluctuations, initial parameters and the
 individual details of successful and unsuccessful sellers. These data 
will then be used in Part 2 and 3 of this project where exploration of the models 
and a prediction of the seller activity, based on its initial parameters,
will be undertaken.

================================================================================

1. Number of buyers and sellers:
    The number of buyers and sellers can be either fixed or chosen from a
    Gaussian distribution which thus changes between market iterations (time
     steps).

    Variables:
        - N_$CURRENCY (int) : Number of sellers in $CURRENCY
        - N_TYPE      (str) : either 'fixed' or 'gaussian' 
        - N_STD       (int) : Standard dev. for Gaussian profile

2. Seller Rates:
    Sellers will choose their rate randomly from a Gaussian distribution 
    centered around the current FOREX rate (CURRENT_RATE). The standard dist.
    of this function is chosen by the ratio CURRENT_RATE / 'RATES_STD' where
    'RATES_STD' is selected in set_params().

3. Buyer and Seller Amounts:
    The amount of currency that each buyer and seller has is chosen randomly
    by sampling of a Log-normal distribution profile. Currency is chosen
    in terms of their 'home' currency. The distribution profile can be 
    chosen differently between currencies and is characterised by the following.

    Variables:
        - MU_$CURRENCY  (float) : Lognorm mean for $CURRENCY 
        - SD_$CURRENCY  (float) : Lognorm standard dev. for $CURRENCY

4. Market Iteration:
    The marketplace is iterated for a set length of time before the seller
    records are written to a database. The behaviour is governed by the 
    following parameters which can be chosen in set_params().
 
    Variables:
        - DAYS_CUTOFF     (int) : Days a sale is on the market before being
                                  cancelled
        - ITERS           (int) : Number of market iterations
        - INITIAL_PRICE (float) : Starting FOREX rate

5. Exchange Rate:
    This can be modelled as 'constant', 'sine' or 'gbm' (Geometric Brownian
    Motion), chosen in the set_params() function.
    
    Variables:
        - 'RATE_TYPE' (str) : 'constant', 'sine' or 'gbm'

    Depending on the choice of rate, additional parameters can be chosen
    specific to that rate.
        - if 'sine' - parameter 'SINE_PERIOD' determines period of oscillation
        - if 'gbm'  - parameters 'DT', 'MU' and 'SIGMA' characterise time 
                      evolution - see accompanying ipynb for details.

6. Outputs:
   The data can be output into either CSV (four files)  or SQL format (one .db
    with four tables): parameters; rates history; successful sellers; and
    unsuccessful sellers. The path, filename and filetype is chosen in
    set_params().

================================================================================
"""
# Author: James Lawlor <jalawlor@tcd.ie>

import numpy as np
import matplotlib.pylab as plt
import random
import pandas as pd
import sqlite3
import os, sys

def set_params():
    """
    set global parameters for FOREX rate, buyer/seller 
    distributions and main iteration routine, see 
    program docstring for more details.

    Returns:
        - dic (dictionary) : Dictionary of parameters
    """
    # Random Seed 
    seed = 5; np.random.seed(seed); random.seed(seed)

    dic = {}
    #specify output type and file name
    dic['file_name'] = 'mu_9_5_sd_0_5_sine_pred_var'    # file name
    dic['output_format'] = "csv" # sql or csv
    dic['path'] = 'exploration_data/'

   # if os.path.exists(dic['path']+dic['file_name']+'_params.db')\
   #      or os.path.exists(dic['path']+dic['file_name']+'.db'):
   #     print 'Path already in use! Choose another file name'
   #     print 'exiting...'
   #     sys.exit(1)

    # FOREX parameters
    dic['RATE_TYPE'] = 'sine'# FOREX model: constant, sine or gbm
    dic['N_TYPE'] = 'gaussian'  # Allocation of N buyers/sellers: fixed or gaussian
    dic['N_STD'] = 10        # STD of gaussian N_TYPE, mean = N_EUR set below

    # sine FOREX params
    dic['SINE_PERIOD'] = 30 # oscillation period in days

    # GBM params #
    dic['DT'] = 0.1   # Time step (days)
    dic['MU'] = 0.00025  # drift factor
    dic['SIGMA'] = 0.05 # volatility

    # STD ratio of Gaussian selection of seller rates. Large vals = small dist.
    dic['RATES_STD'] = 100.0 # STD DEV = RATES/RATES_STD

    # main program parameters
    dic['N_EUR'] = 100 # Number of EUR sellers
    dic['MU_EUR'] = 9.5 # lognorm mean seller amount distribution
    dic['SD_EUR'] = 0.5 # lognorm std

    dic['N_USD'] = 100  # USD values
    dic['MU_USD'] = 9.0
    dic['SD_USD'] = 1.0

    dic['DAYS_CUTOFF'] = 60# days until unsold order cancelled
    dic['ITERS'] = 300 # Number of days to run market simulation
    dic['INITIAL_PRICE'] = 1.0 # initial USD to EUR rate

    return dic 

def sine_rate(initial, t,period):
    """
    Oscillatory sine model of exchange rate

    Args:
        initial (float) : the initial rate at t = 0
        t       (int)   : current time 
        period  (int)   : oscillation period
    Returns:
        (float) updated rate at time t using sine function and above parameters
    """
    return initial - 0.1*np.sin(2*np.pi*t/period)

def gbm_rate(current, dt, mu, sigma):
    """
    return an updated FOREX price using geometric Brownian motion

    Args:
        current (float) : current FOREX rate
        dt      (float) : time step
        mu      (float) : drift factor
        sigma   (float) : volatility
    Returns:
        (float) updated rate using GBM model for FOREX fluctuations
    """
    W = np.random.standard_normal() 
    W = np.cumsum(W)[0]*np.sqrt(dt)
    x = (mu-0.5*sigma**2)*dt+(sigma*W) 
    return current*np.exp(x)

def rate_update(current, t, parameters_dic):
    """
    updates the exchange rate every market iteration using a 
    specified FOREX rate model

    Args:
        current (float)      :  current FOREX rate
        t       (int)        :  current time
        parameters_dic (dic) :  parameters dictionary
    Returns:
        current rate
    """
    # call Sinuosoidal model
    if parameters_dic['RATE_TYPE'] == 'sine':
        return sine_rate( parameters_dic['INITIAL_PRICE'],
                          t,
                          parameters_dic['SINE_PERIOD'])
    # call Geometric Brownian Motion model
    elif parameters_dic['RATE_TYPE'] == 'gbm':
        return gbm_rate( current,parameters_dic['DT'],
                         parameters_dic['MU'],
                         parameters_dic['SIGMA'])
    # returns unchanged 
    elif parameters_dic['RATE_TYPE'] == 'constant':
        return current

def whos_trading(rate,param_dic,sell_dic,currency,timestamp):
    """
    Initialises and returns buyers and sellers for a select currency

    Args:
        rate       (float) : current FOREX rate
        param_dic  (dic)   : dictionary of simulation parameters
        sell_dic   (dic)   : current dictionary of all market sellers
        currency   (str)   : selected currency ('usd' or 'eur')
        timestamp  (int)   : current time

    Returns:
        buyers(numpy arr.)  : 1D vector of buyer amounts in selected currency
        sell_dic    (dic)   : updated version of input sell_dic with new sellers
    """
    curr_str = currency.upper() #convert currency to uppercase
    n = param_dic['N_'+curr_str] # number of buyers/sellers requested
    if param_dic['N_TYPE'] == 'fixed':  # Fixed number of buyers and selllers
        n_buyers = n  
        n_sellers = n
    elif param_dic['N_TYPE'] == 'gaussian': # Random number of buyers/sellers
        n_buyers = int(np.abs(              # using Gaussian distribution
            np.random.normal(n,param_dic['N_STD'])))    
        n_sellers = int(np.abs(
            np.random.normal(n,param_dic['N_STD']))) 

    # Initialise buyers vector
    buyers = np.random.lognormal(param_dic['MU_'+curr_str],
                                 param_dic['SD_'+curr_str], n_buyers)

    # Initialise sellers
    for _ in range(n_sellers):
        # Create unique key
        if len(sell_dic) == 0:  # Safe check for first seller in simulation
            key = 0
        else:                   
            key = max(sell_dic.keys(), key=int) + 1
        # Create dictionary of attributes for this seller
        sell_dic[key] = {
                    'Currency'        : curr_str,        # Seller currency
                    'Day Advertised'  : int(timestamp),  # Day sale created
                    'Day Tracker'     : [],              # Tracks days on market
                    'Sale Tracker'    : [],              # Tracks amount
                    # Choose a rate using gaussian around current interbank
                    'Rate Offered'    : np.random.normal(
                                            rate,rate/param_dic['RATES_STD']),
                    # Chooses initial amount using lognorm distribution
                    # (see ipython notebook)
                    'Initial Amount'  : np.random.lognormal(
                                            param_dic['MU_'+curr_str],
                                            param_dic['SD_'+curr_str]),
                     }
        # Init. seller's purse in order to keep separate from tracking data
        sell_dic[key]['Current Amount'] = sell_dic[key]['Initial Amount']

    return buyers, sell_dic

def do_trades(b_to_s_rate,buyers,sellers_dic,day,sell_currency):
    """
    Completes all trades between buyers in one currency and 
    sellers in another, with exchange rate b_to_s_rate between
    buyer -> seller currencies.
    This function is called twice per iteration in __main__ for
    each combination of buyers and sellers

    Args:
        b_to_s_rate (float) : FOREX rate between buyers -> sellers currencies 
        buyers (numpy arr.) : 1D array of buyer amounts in own currency
        sellers_dic   (dic) : seller dictionary of all sellers
        day           (int) : today's timestamp
        sell_currency (str) : currency of the sellers
    Returns:
        unsold_buyers (list): list of unsold buyer amounts
        sellers_dic   (dic) : updated seller dictionary
    """
    # company will do trade at 0.5% b_to_s_rate if no suitable trade found
    # so we consider sellers that offer rates better than that only
    company_b_to_s = b_to_s_rate*(0.995)
    company_s_to_b = 1.0/company_b_to_s

    # for speed, take subset of sellers_dictionary of sellers who
    # have competetive rates better than company offer
    competetive_sellers = {key: val for key, val in sellers_dic.items()\
                             if val['Rate Offered'] < company_s_to_b\
                             and val['Currency'] == sell_currency}

    # convert this to an array for safety - we don't want to change dict
    # while iterating over it later
    seller_array = []
    for key,value in sorted( \
            competetive_sellers.items(),key=lambda x: x[1]['Rate Offered']):
        rate   =   competetive_sellers[key]['Rate Offered']
        amount =   competetive_sellers[key]['Current Amount']
        seller_array.append([key,rate,amount])

    # iterate through buyer list
    for buyer_idx in range(len(buyers)):
        # iterate through array of competetive sellers 
        for row_n in range(len(seller_array)):
            seller_id , seller_rate, seller_amount = seller_array[row_n]
            buyer_amount = buyers[buyer_idx]
            # calculate trade amount in seller currency
            trade_amount =  seller_amount - buyer_amount/seller_rate
            # if seller has enough money, remove amount from purse
            if trade_amount  >= 0.0:      
                seller_array[row_n][2] -= buyer_amount/seller_rate
                buyers[buyer_idx] = 0.0     # set buyer amount to 0
                break
            # elif seller has no money, set their purse to zero and update buyer 
            elif trade_amount < 0.0:
                seller_array[row_n][2] = 0.0
                buyers[buyer_idx] -= seller_amount*seller_rate

    # update successful seller details in sellers_dic
    for row_n in range(len(seller_array)):
        key , seller_rate, seller_amount = seller_array[row_n]
        sellers_dic[key]['Current Amount'] = seller_amount
        sellers_dic[key]['Sale Tracker'].append(seller_amount)  
        sellers_dic[key]['Day Tracker'].append(t)

    # Update uncompetetive sellers also to track their Days and Sales 
    for key in {key: val for key, val in sellers_dic.items() \
                     if val['Rate Offered'] >= company_s_to_b \
                     and val['Currency'] == sell_currency}:
        sellers_dic[key]['Sale Tracker'].append(
            sellers_dic[key]['Current Amount'])
        sellers_dic[key]['Day Tracker'].append(t)

    unsold_buyers =  [x for x in buyers if x != 0.0]

    return unsold_buyers , sellers_dic

if __name__ == "__main__":
   
    param_dic = set_params()    # Get parameters

    for run_number in range(1): # can run several markets
        # initial FOREX rate for USD to EUR
        rate = param_dic['INITIAL_PRICE']
#        # init arrays for plotting
        len_checker = [] ; rate_tracker = [] ; trng_out = [];buyers_tracker = []
        # create dictionaries for sellers and output
        sellers_dic = {}
        completed_sellers = {} ; uncompleted_sellers = {} # used for output
        usd_success_tracker = [] ; eur_success_tracker = []

        ########################################################
        ##########      Market Iteration    ####################
        ########################################################

        for t in range(0,param_dic['ITERS']):  # Time step

            # Initialise new buyers and sellers
            eur_buyers, sellers_dic = whos_trading(1.0/rate, param_dic,
                                                   sellers_dic, 'eur', t)
            usd_buyers, sellers_dic = whos_trading(rate, param_dic,
                                                   sellers_dic, 'usd', t)

#            for k , seller in sellers_dic.iteritems():
#                if seller['Currency']=='USD':
#                    usd_cnt += 1
#                else:
#                    eur_cnt += 1

            # TRADE: USD BUYING EUROS
            unsold_buyers_usd, sellers_dic = do_trades(rate, usd_buyers,
                                    sellers_dic, t, 'EUR')
            # TRADE: EUR BUYING USD 
            unsold_buyers_eur, sellers_dic = do_trades(1.0/ rate, eur_buyers,
                                     sellers_dic, t, 'USD')

            remaining_buyers_usd = unsold_buyers_usd
            remaining_buyers_eur = unsold_buyers_eur

            usd_cnt = 0 ; eur_cnt = 0                                            

            #### TIDY UP #####
            for key in sellers_dic.keys():
                # move moneyless sellers to completed dic
                if sellers_dic[key]['Current Amount'] == 0.0:
                    if sellers_dic[key]['Currency'] == 'USD':
                        usd_cnt += 1
                    else:
                        eur_cnt += 1

                    completed_sellers[key] = sellers_dic[key]
                    completed_sellers[key]['ID'] = key
                    del sellers_dic[key]
                else:
                # if seller has not completed and is beyond cutoff
                    if t - sellers_dic[key]['Day Advertised'] >= param_dic[  
                                                                'DAYS_CUTOFF']:
                        uncompleted_sellers[key] = sellers_dic[key]
                        uncompleted_sellers[key]['ID'] = key
                        del sellers_dic[key]

#            usd_cnt = 0 ; eur_cnt = 0                                            
#            for k , seller in sellers_dic.iteritems():
#                if seller['Currency']=='USD':
 #                   usd_cnt += 1
 #               else:
 #                   eur_cnt += 1

                
#            usd_cnt = 0 ; eur_cnt = 0                                            
#            for k , seller in sellers_dic.iteritems():
#                if seller['Currency']=='USD':
#                    usd_cnt += 1
#                else:
#                    eur_cnt += 1

        
            #if enough time has elapsed for market to settle, save data
            if t > param_dic['DAYS_CUTOFF']:
                trng_out.append(t)
                len_checker.append(len(sellers_dic))
                buyers_tracker.append([len(remaining_buyers_usd),
                    len(remaining_buyers_eur)])
                rate_tracker.append(rate)
                usd_success_tracker.append(usd_cnt)
                eur_success_tracker.append(eur_cnt)
#                print t, buyers_tracker[-1][0], buyers_tracker[-1][1], \
                print t, usd_cnt, eur_cnt, len_checker[-1], rate_tracker[-1]

            # update the FOREX rate ready for next time step
            rate = rate_update(rate, t, param_dic)

        ########################################################
        ##########  Market Iteration Finished  #################
        ########################################################
        # Now write the data to pandas dataframes and output

        # Dataframe of market behaviour over time
        rates_df = pd.DataFrame(
                     {'USD to EUR rate': rate_tracker,
                     'USD sales' : np.array(usd_success_tracker)/           \
                            (1.0*param_dic['N_USD']),\
                     'EUR sales' : np.array(eur_success_tracker)/           \
                            (1.0*param_dic['N_EUR']) \
                            
                    },
                    columns=['EUR sales',
                             'USD sales',
                             'USD to EUR rate'],
                     index=trng_out)
        rates_df.index.name = 'Day'

        # Dataframe of sellers who sold all their amounts
        sold_df                 = pd.DataFrame.from_dict(completed_sellers)
        sold_df.columns         = range(len(completed_sellers))
        sold_df                 = sold_df.transpose()
        sold_df['Day Tracker']  = sold_df['Day Tracker'].map(\
                                    lambda x: str(x).strip('[]'))  
        sold_df['Sale Tracker'] = sold_df['Sale Tracker'].map(\
                                    lambda x: str(x).strip('[]'))  

        # Dataframe of sellers who didn't finish selling
        unsold_df                 = pd.DataFrame.from_dict(uncompleted_sellers)
        unsold_df.columns         = range(len(uncompleted_sellers))
        unsold_df                 = unsold_df.transpose()
        unsold_df['Day Tracker']  = unsold_df['Day Tracker'].map(\
                                        lambda x: str(x).strip('[]'))  
        unsold_df['Sale Tracker'] = unsold_df['Sale Tracker'].map(\
                                        lambda x: str(x).strip('[]'))  

        # save parameters
        param_df = pd.DataFrame.from_dict(param_dic, orient='index')
        param_df.columns = ['value']

        # get path and filename requested 
        fname = param_dic['file_name'] ; pathname = param_dic['path']

        if param_dic['output_format'] == "sql":             # SAVE TO SQL
                con = sqlite3.connect(pathname+fname+'.db')
                param_df.to_sql('parameters', con)
                rates_df.to_sql('rates', con) 
                sold_df.to_sql('sold', con)
                unsold_df.to_sql('unsold', con)
        elif param_dic['output_format'] == "csv":           # SAVE TO CSV
                param_df.to_csv(pathname+fname+'_params.csv')
                rates_df.to_csv(pathname+fname+'_rates.csv')
                sold_df.to_csv(pathname+fname+'_sold.csv')
                unsold_df.to_csv(pathname+fname+'_unsold.csv')
    
        plt.plot(trng_out,np.array(usd_success_tracker)/(1.0*param_dic['DAYS_CUTOFF']*param_dic['N_USD']), 'k')
        plt.plot(trng_out,np.array(eur_success_tracker)/(1.0*param_dic['DAYS_CUTOFF']*param_dic['N_EUR']), 'r')
#        plt.plot(trng_out,np.array([x[1] for x in buyers_tracker])/float(param_dic['N_EUR']), 'r')
        plt.plot(trng_out, np.array(rate_tracker), 'b')
        plt.show()
