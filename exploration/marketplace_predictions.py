"""
================================================================================
    FOREX Project - Part 2 - Prediction Data Set Generation
================================================================================

 This program is extremely similar to marketplace.py in the top project folder,
 but has been modified to generate a large amount of data for prediction 
 purposes. 

 Please see the accompanying Prediction notebook. 

================================================================================
"""
# Author: James Lawlor <jalawlor@tcd.ie>

import numpy as np
import matplotlib.pylab as plt
import random
import pandas as pd
import sqlite3
import os, sys

def sine_rate(initial, t,period):
    """
    Oscillatory sine model of exchange rate
    """
    return initial - 0.1*np.sin(2*np.pi*t/period)

def gbm_rate(current, dt, mu, sigma):
    """
    return an updated FOREX price using geometric Brownian motion
    """
    W = np.random.standard_normal() 
    W = np.cumsum(W)[0]*np.sqrt(dt)
    x = (mu-0.5*sigma**2)*dt+(sigma*W) 
    return current*np.exp(x)

def rate_update(current, t, parameters_dic):
    """
    updates the exchange rate every market iteration using a 
    specified FOREX rate model
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
   
    #param_dic = set_params()    # Get parameters


    print 'WARNING: this will overwrite large amounts of data used for the Prediction part of the project\n'
    print 'If you wish to proceed, comment out Lines 164-166 of this program'
    stop

    import shelve
    from os import listdir
    from os.path import isfile, join
   
    dic_files =  [f for f in listdir('./param_dics/') if
         isfile(join('./param_dics', f))] 

    seed = 5; np.random.seed(seed); random.seed(seed)

    for f in dic_files:
        param_dic = shelve.open('./param_dics/'+f, writeback = True)
        param_dic['file_name']     = f.split('_')[0] + '_set'
        param_dic['output_format'] = 'csv'
        param_dic['path']          = './data/'


        for run_number in range(5): # can run several markets
            print " %%%% Running Set " + str(f.split('_')[0]) + ' run ' +\
                 str(run_number) + ' of 5.'
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
            #        print t, usd_cnt, eur_cnt, len_checker[-1], rate_tracker[-1]
    
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
            fname = param_dic['file_name']+'_run_'+str(run_number) 
            pathname = param_dic['path']
    
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
        
            #plt.plot(trng_out,np.array(usd_success_tracker)/(1.0*param_dic['DAYS_CUTOFF']*param_dic['N_USD']), 'k')
            #plt.plot(trng_out,np.array(eur_success_tracker)/(1.0*param_dic['DAYS_CUTOFF']*param_dic['N_EUR']), 'r')
    #       # plt.plot(trng_out,np.array([x[1] for x in buyers_tracker])/float(param_dic['N_EUR']), 'r')
            #plt.plot(trng_out, np.array(rate_tracker), 'b')
            #plt.show()
