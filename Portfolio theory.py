import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from scipy.stats import norm
Cash=10000
start=datetime.strptime("2015-01-01","%Y-%m-%d")
end=datetime.strptime("2022-12-12","%Y-%m-%d")
num_years=end.year-start.year
Stocks=["AAPL","TSLA","COST","AC","GOOG"]
###############
#Download and save
#df=yf.download(Stocks,start,end)["Adj Close"].pct_change()
#df_SP=yf.download("^GSPC",start,end)["Adj Close"].pct_change()
#concat_df=pd.concat([df,df_SP],axis=1).dropna()
#concat_df.to_csv("Fang portfolio"+".csv")
concat_df=pd.read_csv("Fang portfolio.csv")
df = concat_df.iloc[:, 1:len(concat_df.columns)-1]
df_SP = concat_df.iloc[:, -1]
print(f"The df is \n{df}")
print(f"SP500 data is \n{df_SP}")

##############
#To compare to SP500

Return_SP=df_SP.mean()
volat_SP=(df_SP.var())**(1/2)*np.sqrt(250)
#df=np.log(1+df["Adj Close"].pct_change())

###################################3
def portfolioreturn(weights):
    return np.dot(df.mean(),weights)
   
#weights=[0.4,0.3,0.2,0.1]
#print(f"Daily return={portfolioreturn(weights)}")

#cov matrix for correlation
Cor_matrix=df.corr()
print(f"The correlation matrix is \n{Cor_matrix})")

#the variance of our portfolio in one year (*np.sqrt(250))
def portfoliostd(weights):
    return (np.dot(np.dot(df.cov(),weights),weights))**(1/2)*np.sqrt(250)
#print(f"The annual volatility of our portfolio is {portfoliostd(weights)}")

#lets create weights for simulation
def weightcreater(Df):
    rand=np.random.random(len(Df.columns))
    rand/=rand.sum()
    return rand
##############################################################
#Optimizing the weight using simulation over 700 weights
#Give the annual risk tolerance c and min 21<c<41
c=21
np.random.seed(123)
Weights=[]
Stds=[]
S_returns=[]
for i in range(700):
    weight=weightcreater(df)
    Weights.append(weight)
    Return=portfolioreturn(weight)*250
    S_returns.append(Return)
    std=portfoliostd(weight)
    Stds.append(std)

stds_plus_minus_one_c = [x for x in Stds if abs(x - c/100) <= 1/100]
indexes_with_std_pmc = [i for i, std in enumerate(Stds) if std in stds_plus_minus_one_c]
max_return_index = max(indexes_with_std_pmc, key=lambda i: S_returns[i])
weight_least_risk_port_sim = Weights[max_return_index]
returns_with_stdc = S_returns[max_return_index]
print(f"max index={max_return_index}")
#########################################################

#The least riskiest portfolio
least_risk_port=Weights[Stds.index(min(Stds))]  
return_of_least_risk=S_returns[Stds.index(min(Stds))]  

plt.scatter(Stds,S_returns,s=10) 
plt.scatter(min(Stds),return_of_least_risk, color="r")
plt.scatter(Stds[max_return_index],returns_with_stdc, color="purple")
plt.title("Efficient Frontier")  
plt.xlabel("Portfolio annual standard deviation(Risk)")
plt.ylabel("Portfolio annual return")
plt.show() 

#############################################

#Optimizing the weight using Mean Variance Optimization MVO
#Give the annual risk tolerance c
c=20

def objective(weight):
    neg_ret=-portfolioreturn(weight)
    return neg_ret

#sum of weights is 1
def constraint1(weight):
    return portfoliostd(weight)-c/100
#The risk of portfolio is c
def constraint2(weight):
    return np.sum(weight)-1
#defining the bounds for weights
bounds=[(0,1)]*len(df.columns)

constraints=[
    {"type":"eq","fun":constraint1},
    {"type":"eq","fun":constraint2}
]

#Set an initial value for weights
initial_weight=np.ones(len(df.columns))/len(df.columns)

#performing portfolio optimization
result=minimize(objective,initial_weight,method="SLSQP",constraints=constraints,bounds=bounds)
optimal_weights=result.x

# Normalize the weights
normalized_opt_weights = optimal_weights / np.sum(optimal_weights)
print(f"Normalized optimal weights when the risk tolerance is {c} percent using MVO:{normalized_opt_weights}")
for i, column in enumerate(df.columns):
    print(f"The normalized weight for {column}: {normalized_opt_weights[i]}")

# Calculate the expected return and risk of the optimized portfolio
expected_return = portfolioreturn(normalized_opt_weights)*250
portfolio_risk = portfoliostd(normalized_opt_weights)
 

#result
data={"Risk amount":[min(Stds),Stds[max_return_index],portfolio_risk,volat_SP],
      "Annual return":[return_of_least_risk,returns_with_stdc,expected_return,Return_SP*250],
      f"Cum return {num_years} ys":[return_of_least_risk*num_years,returns_with_stdc*num_years,expected_return*num_years,Return_SP*250*num_years],
      f"{Stocks[0]}":[least_risk_port[0],weight_least_risk_port_sim[0],normalized_opt_weights[0],0],
      f"{Stocks[1]}":[least_risk_port[1],weight_least_risk_port_sim[1],normalized_opt_weights[1],0],
      f"{Stocks[2]}":[least_risk_port[2],weight_least_risk_port_sim[2],normalized_opt_weights[2],0],
      f"{Stocks[3]}":[least_risk_port[3],weight_least_risk_port_sim[3],normalized_opt_weights[3],0],
      f"{Stocks[4]}":[least_risk_port[4],weight_least_risk_port_sim[4],normalized_opt_weights[4],0]
}
Result=pd.DataFrame(data,index=["Least risky portfolio","Best portfolio using simulation","Best portfolio using MVO","SP500"])
print(Result)

######################################
#Calculating the beta of portfolios
var_SP=np.var(df_SP)
Cov_Appl_SP=np.cov(df["AAPL"],df_SP)[0,1]
Cov_TSLA_SP=np.cov(df["TSLA"],df_SP)[0,1]
Cov_COST_SP=np.cov(df["COST"],df_SP)[0,1]
Cov_AC_SP=np.cov(df["AC"],df_SP)[0,1]
Cov_GOOG_SP=np.cov(df["GOOG"],df_SP)[0,1]
Beta_AAPL=Cov_Appl_SP/var_SP
Beta_TSLA=Cov_TSLA_SP/var_SP
Beta_COST=Cov_COST_SP/var_SP
Beta_AC=Cov_AC_SP/var_SP
Beta_GOOG=Cov_GOOG_SP/var_SP
print(f"betas=:{Beta_AAPL},{Beta_TSLA},{Beta_AC}")
Beta_vector=[Beta_AAPL,Beta_TSLA,Beta_COST,Beta_AC,Beta_GOOG]
pbeta_min_risk=np.dot(Beta_vector,least_risk_port)
pbeta_sim=np.dot(Beta_vector,weight_least_risk_port_sim)
pbeta_optimized=np.dot(Beta_vector,normalized_opt_weights)

######################################################
#Evaluating VaR
Confidence=0.95
z_alpha=-norm.ppf(1-Confidence)
VaR_min_risk_port=z_alpha*return_of_least_risk
VaR_sim_port=z_alpha*returns_with_stdc
VaR_opt_port=z_alpha*expected_return
data_risk_analysis={"Beta":[pbeta_min_risk,pbeta_sim,pbeta_optimized],
      "VaR":[VaR_min_risk_port,VaR_sim_port,VaR_opt_port],
      f"VaR for {Cash}":[VaR_min_risk_port*Cash,VaR_sim_port*Cash,VaR_opt_port*Cash]
      
}
Result_beta_var=pd.DataFrame(data_risk_analysis,index=["Least risky portfolio","Best portfolio using simulation","Best portfolio using MVO"])

print(f"Beta and VaR analysis for each portfolio:\n {Result_beta_var}")




    


    




