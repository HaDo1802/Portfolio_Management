import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime as dt, timedelta
from scipy import optimize as sc
import plotly.graph_objects as go
# Load Data
def get_data(stocks, start, end):
    stk_data = web.DataReader(stocks, 'stooq', start=start, end=end)
    stk_data = stk_data[[name for name in stk_data.columns if name[0] == 'Close']]
    stk_data.columns = stk_data.columns.droplevel()
    simreturns = stk_data.pct_change().dropna()  # Drop NaN values
    mean_return = simreturns.mean()
    covar = simreturns.cov()
    return mean_return, covar

# Calculate the portfolio performance
def port_performance(weights, mean_return, covar):
    port_return = np.sum(mean_return * weights) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(covar, weights))) * np.sqrt(252)
    return port_return, port_std

# Optimization function to maximize Sharpe ratio
def negative_sharpe(weights, mean_return, covar, risk_free_rate=0):
    port_return, port_std = port_performance(weights, mean_return, covar)
    sharpe_ratio = (port_return - risk_free_rate) / port_std
    return -sharpe_ratio  # Negative because we want to maximize Sharpe
def max_sharpe_ratio(mean_return, covar, risk_free_rate=0, constraint_set=(0, 1)):
    num_assets = len(mean_return)
    # Initial guess: equal weights
    initial_weights = np.array([1.0/num_assets] * num_assets)
    args = (mean_return, covar, risk_free_rate)
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for _ in range(num_assets))
    # Minimize the negative Sharpe ratio
    result = sc.minimize(negative_sharpe, initial_weights,args=args,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints)
    return result
#Minimize the Portfolio Risk
def PortfolioVar(weights, mean_return, covar):
    portfolio_Var = port_performance(weights, mean_return, covar)[1]
    return portfolio_Var 
def min_portfolio_var(mean_return, covar, constraint_set=(0, 1)):
    num_assets = len(mean_return)
    # Initial guess: equal weights
    # Arguments for the negative_sharpe function
    args = (mean_return, covar)
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for _ in range(num_assets))
    # Minimize the negative Sharpe ratio
    result = sc.minimize(
        PortfolioVar, num_assets*[1./num_assets],
        args=args,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints)
    return result
#Calculate the minimum Variance based on the Targeted Return
def PortfolioReturn(weights, mean_return, covar):
    return port_performance(weights, mean_return, covar)[0]
def efficientFrontier(mean_return, covar, target_return, constraint_set=(0, 1)):
    '''For each target return, we try to find the lowest Variance associated '''
    num_assets = len(mean_return)
    # Initial guess: equal weights
    # Arguments for the negative_sharpe function
    args = (mean_return, covar)
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: PortfolioReturn(x, mean_return, covar) - target_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for _ in range(num_assets))
    # Minimize the negative Sharpe ratio
    result = sc.minimize(
        PortfolioVar, num_assets*[1./num_assets],
        args=args,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints)
    return result

#Function to call out all the measurements:
def calculated_Result( mean_return, covar,risk_free_rate=0, constraint_set=(0, 1)):
    #MAX Sharp Ratio Portfolio
    optimal_weights=max_sharpe_ratio(mean_return, covar)['x']
    opt_return, opt_std = port_performance(optimal_weights, mean_return, covar)
    optimal_allocation = pd.DataFrame(optimal_weights,index=mean_return.index,columns=["Allocation"])
    #MIN Volatility Portfolio
    minimal_weights = min_portfolio_var(mean_return, covar)['x']
    minimal_return, minimal_std = port_performance(minimal_weights, mean_return, covar)
    minimal_allocation=pd.DataFrame( minimal_weights,index=mean_return.index,columns=["Allocation"])
    
    #Find the Variance for all of these targeted return
    List_of_Vol=[]
    targetReturns =np.linspace(minimal_return,opt_return,20)
    for target in targetReturns:
        List_of_Vol.append(efficientFrontier(mean_return,covar,target)['fun'])
        
    minimal_return, minimal_std = round(minimal_return*100,2), round(minimal_std*100,2)
    opt_return, opt_std = round(opt_return*100,2), round(opt_std*100,2)
    return opt_return, opt_std, optimal_allocation, minimal_return, minimal_std,minimal_allocation,List_of_Vol,targetReturns

def EF_graph(mean_return, covar, risk_free_rate=0, constraint_set=(0, 1)):
    '''Return the graph for efficient frontier, min Vol, and max Sr'''
    opt_return, opt_std, optimal_allocation, minimal_return, minimal_std, minimal_allocation, ef_vols, targetReturns = calculated_Result(mean_return, covar, risk_free_rate, constraint_set)
    
    # Max Sharpe Ratio
    MaxSharpRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[opt_std],
        y=[opt_return],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )
    
    # MIN Vol
    MinVol = go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[minimal_std],
        y=[minimal_return],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )
    
    # Efficient Frontier
    EF_Curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines+markers',
        x=[round(vol*100, 2) for vol in ef_vols],
        y=[round(ret*100, 2) for ret in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    
    data = [MaxSharpRatio, MinVol, EF_Curve]
    
    layout = go.Layout(
        title='Portfolio Optimization with the Efficient Frontier',
        yaxis=dict(title='Annualized Return (%)'),
        xaxis=dict(title='Annualized Volatility (%)'),
        showlegend=True,
        legend=dict(
            x=0.75, 
            y=0, 
            traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2
        ),
        width=800,
        height=600
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig
    
END_DATE = dt.now()
START_DATE = END_DATE - timedelta(days=365)
stk_tickers_1 = ['MSFT', 'GOOGL', 'NVDA'] 
mean_return, covar = get_data(stk_tickers_1, START_DATE, END_DATE)
weights = np.array([0.3, 0.3, 0.4])
port_return, port_std = port_performance(weights, mean_return, covar)
optimal_allocation=calculated_Result(mean_return, covar)
#print(calculated_Result(mean_return, covar,risk_free_rate=0, constraint_set=(0, 1)))
fig = EF_graph(mean_return, covar)
fig.show()