import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y = []

with open('rendered.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    ct = 0
    for row in plots:
        r = []
        x.append(ct)
        ct += 1
        for i in row:
            r.append(int(i))
        y.append(np.max(r))

print(x)
plt.plot(x,y)
plt.xlabel('segment')
plt.ylabel('average rendered quality')
plt.title('Average quality')
# plt.legend()
# plt.show()
plt.savefig('max_baseline.png')