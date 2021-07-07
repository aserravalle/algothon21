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
    currentPos += rpos
    return currentPos

def GetNewPosition(market_data: np.array) -> np.array:
    global currentPos
    global portfolioValue
    market_data = pd.DataFrame(market_data, columns=range(100))
    weights = GetNewWeights(market_data)
    currentPrices = np.array(market_data.iloc[-1,:])
    position = np.rint((portfolioValue * weights) / currentPrices)
    return position

def GetNewWeights(market_data: pd.DataFrame) -> np.array:
    mu = expected_returns.mean_historical_return(market_data)
    S = risk_models.sample_cov(market_data)
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w[22] + w[41] <= 0.08)
    weights = np.array([x for x in ef.max_sharpe().values()])
    assert weights.shape == (nInst,), f"weights.shape == {weights.shape} but should be {(nInst,)}"
    return weights

weights_opt = np.array([
    0.0124729086575895,0.0429874059998886,0.0784947873994312,0.0,0.0,0.0087502385801264,0.0,0.0,0.1270702410709571,0.0,0.0,0.0,
    0.0509175747314244,0.0,0.0,0.0292337669593769,0.0,0.011967166431767,0.0325507684007594,0.0148428045147731,0.0048846154105992,
    0.0, 0.031686319401326, 0.0, 0.0053440833856292, 0.002781367811903, 0.2408703568687767, 0.0009925113896684, 0.0086212654436397,
    0.0, 0.0155123642136002, 0.0, 0.0108841711209465, 0.0336985663877109, 0.0090905113860708, 0.0, 0.0, 0.0, 0.0461520731535632,
    0.0074511352520559, 0.0124323625418458, 0.048313680598674, 0.0, 0.0342041590588955, 0.0, 0.0322012649808268, 0.0, 0.0,
    0.019420964221114, 0.0, 0.0002685627430153, 0.0, 0.0, 0.0, 0.0004579405799816, 0.0, 0.0, 0.0, 0.0024078913134016, 0.0, 0.0,
    0.0, 0.0, 0.0006005287313613, 0.0003657721361912, 0.0006181340384045, 0.0, 0.0, 0.000611168937363, 0.0001748948049471, 0.0,
    0.0009588829920902, 0.0, 0.0, 0.0, 0.0017272149405503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003339734918945, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0047089571093863, 0.0, 0.003212469730607, 0.0, 0.0031874618561432, 0.0, 0.0022184028192443, 0.0, 0.0, 0.0,0.0010436330108392, 0.0002689139645895, 0.0
])
