================================================================================
        FOREX Marketplace Simulation, Analysis and Predictions

		      --- README / Introduction ----

			James Lawlor - 4/4/2015
================================================================================

This project simulates a peer-to-peer currency marketplace of two currencies with a variable exchange rate between them. A range of market characteristics, e.g. exchange rate variations, trade frequency and trade amounts, can be chosen and the impact on the marketplace then studied. The first and second parts of this project are devoted to doing this. A third part of the project demonstrates how Machine Learning techniques can be used to predict whether an advertised sale placed on the market will be bought or not, and if so how long it takes for the sale to occur. Finally, a summary of the project is provided.

An outline of the project is shown below, the recommended reading order is

	Part 1: marketplace.ipynb
		marketplace.py 
	Part 2: data_exploration.ipynb
		data_exploration_2.ipynb
	Part 3: deal_or_no_deal.ipynb
		time_to_sale_prediction.ipynb
	Part 4: summary.txt

--- Contents ---

	1. Marketplace Simulation:
	
		marketplace/marketplace.ipyb 		- Introduction to the mechanics and rules of the marketplace
		marketplace/marketplace.py		- Python implementation for the marketplace simulation

	2. Data Exploration & Analysis:

		exploration/data_exploration.ipynb	- Exploring the effects of different exchange rates (constant, sinuosoidal and geometric Brownian Motion)
		exploration/data_exploration_2.ipynb	- Exploring how buyer/seller attributes affects the general trends of sales

		These two notebooks are supported by the programs:

		exploration/marketplace_predictions.py	- Bulk marketplace simulation for many different initial parameters,
							  outputs to directory /prediction/data/ for use in Data Exploration and Predictions
		exploration/dic_gen.py			- Generates Python dictionaries of parameters (saved to directory /prediction/param_dics/)
							  for input into marketplace_predictions.py
		exploration/data/*.csv			- CSV files for use with the prediction notebooks in the next section

	3. Predictions:

		deal_or_no_deal.ipynb  			- Classification algorithms used to predict whether a trader will successfully sell or not
		time_to_sale_prediction.ipynb		- Regression algorithms used to predict the length of time it takes a successful trade to complete
		classification_prediction_table.csv	- Table generated by deal_or_no_deal.ipynb
		time_to_sale_prediction_table.csv	- Table generated by time_to_sale_prediction.ipynb
	
	4. Summary:

		summary.txt

--- Running the Programs and Notebooks ---

Requires Python 2.7 or higher, and IPython notebooks. Uses the libraries sklearn, numpy, scipy, matplotlib and seaborn. 
