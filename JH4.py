import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

nInst=100
currentPos = np.zeros(nInst)
portfolioValue = 100000

def getMyPosition(prcSoFar: np.array):
    global currentPos
    (nins, nt) = prcSoFar.shape
    rpos = GetNewPosition(prcSoFar.T)
    rpos = ComparePositions(currentPos, rpos)
    currentPos = rpos
    return currentPos

def ComparePositions(currentPos:np.array, rpos:np.array) -> np.array:
    """
    Meant to look at previous weights and see if the change is significant enough to justify the commission cost
    """
    print("currentPos",currentPos)
    return rpos

def GetNewPosition(market_data: np.array) -> np.array:
    global currentPos
    global portfolioValue
    market_data = pd.DataFrame(market_data, columns=range(100))
    weights = GetNewWeights(market_data)
    print(np.argmax(weights))
    print(np.argmin(weights))
    portfolioValue = GetNewPortfolioValue(portfolioValue, market_data, weights)
    currentPrices = np.array(market_data.iloc[-1,:])
    position = np.rint((portfolioValue * weights) / currentPrices)

    # print(f"\ncurrentPrices:\n", currentPrices)
    # print(f"\nweights:\n", weights)
    # print(f"\nposition:\n", position)
    
    return position

def GetNewPortfolioValue(current_value, market_data: np.array, weights:np.array) -> int:
    pct_return = market_data.iloc[-1,:] / market_data.iloc[-2,:]
    result = np.sum(pct_return * weights * current_value)
    return result

def GetNewWeights(market_data: pd.DataFrame) -> np.array:
    mu = expected_returns.mean_historical_return(market_data)
    S = risk_models.sample_cov(market_data)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
    # ef.add_constraint(lambda w: w[:] <= 0.10)
    ef.add_constraint(lambda w: w[22] + w[41] <= 0.08)
    weights = np.array([x for x in ef.max_sharpe().values()])
    assert weights.shape == (nInst,), f"weights.shape == {weights.shape} but should be {(nInst,)}"
    return weights

getMyPosition(market_data.T.values)