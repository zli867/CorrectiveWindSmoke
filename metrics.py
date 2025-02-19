import numpy as np
import scipy.stats


# reference: https://www.tandfonline.com/doi/full/10.1080/10962247.2016.1265027
def remove_nan_values(prediction, observation):
    valid_idx = (~np.isnan(prediction)) & (~np.isnan(observation))
    valid_p = prediction[valid_idx]
    valid_o = observation[valid_idx]
    return valid_p, valid_o


def MB(prediction, observation):
    return np.sum(prediction - observation) / len(prediction)


def ME(prediction, observation):
    return np.sum(np.abs(prediction - observation)) / len(prediction)


def RMSE(prediction, observation):
    return np.sqrt(np.sum((prediction - observation) ** 2) / len(prediction))


def CRMSE(prediction, observation):
    p_mean, o_mean = np.mean(prediction), np.mean(observation)
    return np.sqrt((1 / len(prediction) * np.sum(((prediction - p_mean) - (observation - o_mean)) ** 2)))


def NMB(prediction, observation):
    return (np.sum(prediction - observation) / np.sum(observation)) * 100


def NME(prediction, observation):
    return ((np.sum(np.abs(prediction - observation))) / np.sum(observation)) * 100


def MNB(prediction, observation):
    return (1 / len(prediction)) * np.sum((prediction - observation) / observation) * 100


def MNE(prediction, observation):
    return (1 / len(prediction)) * np.sum(np.abs(prediction - observation) / observation) * 100


def FB(prediction, observation):
    return (2 / len(prediction)) * np.sum((prediction - observation) / (prediction + observation)) * 100


def FE(prediction, observation):
    return (2 / len(prediction)) * np.sum(np.abs(prediction - observation) / (prediction + observation)) * 100


def IOA(prediction, observation):
    numerator = np.sum((prediction - observation) ** 2)
    prediction_shift = np.abs((prediction - np.mean(observation)))
    observation_shift = np.abs((observation - np.mean(observation)))
    denominator = np.sum((prediction_shift + observation_shift) ** 2)
    return 1 - numerator / denominator


def pearson_r(prediction, observation):
    r = scipy.stats.pearsonr(prediction, observation)
    return r[0]
