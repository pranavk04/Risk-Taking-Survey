import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import chisquare
import statistics

data = []

ages = []
risk = []


ages_grouped = [20,30,40,50,60,70]
risk_grouped = [[], [], [], [], [], []]

risk_median = []

with open('data-primed.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar='|')
    for row in reader:
        data.append(row)

data.sort(key=lambda x: int(x[0]))

for row in data:
    ages.append(row[0])
    r = 0
    rp = 0
    for x in range(9):
        if row[x+1] == 'Yes':
            r = r + x + 1

    for x in range(9):
        if row[x+11] == 'Yes':
            r = r + 0.5*(x + 1)

    risk.append(r)

for i in range(0, len(ages)):
    ages[i] = int(ages[i])

for i in range(0, len(risk)):
    risk[i] = int(risk[i])

for i in range(0, len(ages)):
    if (ages[i] <= 20):
        risk_grouped[0].append(risk[i])
    elif ( 20 < ages[i] <= 30 ):
        risk_grouped[1].append(risk[i])
    elif ( 30 < ages[i] <= 40 ):
        risk_grouped[2].append(risk[i])
    elif ( 40 < ages[i] <= 50 ):
        risk_grouped[3].append(risk[i])
    elif ( 50 < ages[i] <= 60 ):
        risk_grouped[4].append(risk[i])
    elif ( 60 < ages[i] <= 70 ):
        risk_grouped[5].append(risk[i])


for i in range(0, len(risk_grouped)):
    risk_median.append(statistics.median(risk_grouped[i]))

print(risk_median)

median_data_primed = {'ages': ages_grouped, 'risk': risk_median}

median_risk_data = pd.DataFrame(data=median_data_primed)
median_x = median_risk_data.ages
median_y = np.log(median_risk_data.risk)

median_model = np.polyfit(median_x, median_y, 1)
print(median_model)
median_predict = np.poly1d(median_model)

median_x_lr = [20, 30, 40, 50, 60, 70]
median_y_lr = median_predict(median_x_lr)
median_y_exp = np.exp(-0.00344564) * np.exp(3.39473154*median_x_lr)

plt.scatter(median_x, median_y)
plt.plot(median_x_lr, median_y_lr, c='g')
plt.plot(median_x_lr, median_y_exp, c='r')
#plt.plot(ages_grouped, risk_median, 'bo')
plt.xlabel('Age groups (<20, <30, etc.)')
plt.ylabel('Median risk factor')
plt.title('Median risk factor based on age groups')
plt.show()

# data_primed = {'ages': ages, 'risk': risk}

# risk_data = pd.DataFrame(data=data_primed)

# x = risk_data.ages
# y = risk_data.risk

# model = np.polyfit(x, y, 1)
# predict = np.poly1d(model)

# x_lr = range(13, 65)
# y_lr = predict(x_lr)

# plt.scatter(x,y)
# plt.plot(x_lr, y_lr, c='r')

# plt.ylabel('Risk Factor based on "Would you have you done this activity?"')
# plt.xlabel('Age')

# print('R2 Value: ')
# print(r2_score(y, predict(x)))

# plt.title('Risk Factor vs. Age')
# plt.show()
