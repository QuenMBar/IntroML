import math
import numpy as np
import pandas as pa
data1 = [8.0, 3.0, 2.0]


def entropy(data: list):
    """Calculates the entropy for a dataset

    Args:
        data (list): List of the amount of each item in the nodes

    Returns:
        float: entropy for the dataset
    """
    probs = []
    for d in data:
        probs.append(d/sum(data))
    result = 0
    for p in probs:
        result += p*math.log2(p)
    return -result


def infoGain(parent: list, children: list):
    result = 0.0
    result = entropy(parent)
    temp = 0.0
    for child in children:
        temp += ((sum(child) / sum(parent)) * entropy(child))
    return result - temp


data = pa.read_csv("ml-bugs.csv")
dataBrown = data[data['Color'] == 'Brown']
dataNotBrown = data[data['Color'] != 'Brown']
dataBlue = data[data['Color'] == 'Blue']
dataGreen = data[data['Color'] == 'Green']
dataNotBlue = data[data['Color'] != 'Blue']
dataNotGreen = data[data['Color'] != 'Green']
data17 = data[data['Length (mm)'] < 17.0]
data20 = data[data['Length (mm)'] < 20.0]
dataNot17 = data[data['Length (mm)'] >= 17.0]
dataNot20 = data[data['Length (mm)'] >= 20.0]

# print(data['Species'].value_counts().transpose().to_numpy())
brownInfo = infoGain(data['Species'].value_counts().transpose().to_numpy(), [
    dataBrown['Species'].value_counts().transpose().to_numpy(),
    dataNotBrown['Species'].value_counts().transpose().to_numpy()])

blueInfo = infoGain(data['Species'].value_counts().transpose().to_numpy(), [
    dataBlue['Species'].value_counts().transpose().to_numpy(),
    dataNotBlue['Species'].value_counts().transpose().to_numpy()])

greenInfo = infoGain(data['Species'].value_counts().transpose().to_numpy(), [
    dataGreen['Species'].value_counts().transpose().to_numpy(),
    dataNotGreen['Species'].value_counts().transpose().to_numpy()])

seventenInfo = infoGain(data['Species'].value_counts().transpose().to_numpy(), [
    data17['Species'].value_counts().transpose().to_numpy(),
    dataNot17['Species'].value_counts().transpose().to_numpy()])

twentyInfo = infoGain(data['Species'].value_counts().transpose().to_numpy(), [
    data20['Species'].value_counts().transpose().to_numpy(),
    dataNot20['Species'].value_counts().transpose().to_numpy()])

print('Brown: {} \n Blue: {} \n Green: {} \n 17: {} \n 20: {}'.format(
    brownInfo, blueInfo, greenInfo, seventenInfo, twentyInfo))
