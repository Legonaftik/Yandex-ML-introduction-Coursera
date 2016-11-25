import pandas as pd
import math
import sklearn.metrics

data = pd.read_csv("data-logistic.csv", header=None)
y = data.ix[:, 0]
x = [data.ix[:, 1], data.ix[:, 2]]
l = len(y)
k = 0.1
C = 10
evcl = 10
cnt = 0
w1 = 0
w2 = 0

# while evcl > 0.00001 and cnt < 10000:
#     s1 = 0
#     s2 = 0
#     v1 = w1
#     v2 = w2
#     for i in range(l):
#         a = w1*x[0][i] + w2*x[1][i]
#         a = 1 + math.exp(-1*y[i]*a)
#         a = 1 - 1/a
#         s1 += y[i]*x[0][i]*a
#         s2 += y[i]*x[1][i]*a
#     w1 += k/l*s1
#     w2 += k/l*s2
#     evcl = math.sqrt((w1-v1)**2 + (w2-v2)**2)
#     cnt += 1
#     print(cnt, w1, w2)  # 0.287811620472 0.0919833021593
#
# scores = []
# for i in range(len(y)):
#     scores.append(1 / (1 + math.exp(-w1*x[0][i] - w2*x[1][i])))
# print(sklearn.metrics.roc_auc_score(y, scores))


while evcl > 0.00001 and cnt < 10000:
    s1 = 0
    s2 = 0
    v1 = w1
    v2 = w2
    for i in range(l):
        a = w1*x[0][i] + w2*x[1][i]
        a = 1 + math.exp(-1*y[i]*a)
        a = 1 - 1/a
        s1 += y[i]*x[0][i]*a
        s2 += y[i]*x[1][i]*a
    w1 = w1 + k/l*s1 - k*C*w1
    w2 = w2 + k/l*s2 - k*C*w2
    evcl = math.sqrt((w1-v1)**2 + (w2-v2)**2)
    cnt += 1
    print(cnt, w1, w2)  # 0.0285587545462 0.0247801372497

scores = []
for i in range(len(y)):
    scores.append(1 / (1 + math.exp(-w1*x[0][i] - w2*x[1][i])))
print(sklearn.metrics.roc_auc_score(y, scores))  # 0.936285714286
