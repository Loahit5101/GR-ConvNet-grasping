import matplotlib.pyplot as plt
import csv


time = []
theta1 = []
theta1ref= []
theta2 = []
theta2ref = []
d = []
d_ref = []

with open('/home/loahit/GRconvnet/utils/visualisation/Control_plots3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for column in plots:
        time.append(column[0])
        theta1.append(column[1])
        theta1ref.append(column[2])
        theta2.append(column[3])
        theta2ref.append(column[4])
        d.append(column[5])
        d_ref.append(column[6])

        




theta1=[float(x) for x in theta1]
time=[float(x) for x in time]
theta1ref=[float(x) for x in theta1ref]
theta2=[float(x) for x in theta2]
theta2ref=[float(x) for x in theta2ref]
d=[float(x) for x in d]
d_ref=[float(x) for x in d_ref]



plt.plot(time, theta1,'b')
plt.plot(time, theta1ref,'r')
plt.xlabel('time')
plt.ylabel('theta1')
plt.title('Theta1 vs Time')
plt.legend()
plt.show()


plt.plot(time, theta2,'b')
plt.plot(time, theta2ref,'r')
plt.xlabel('time')
plt.ylabel('theta1')
plt.title('Theta2 vs Time')
plt.legend()
plt.show()


plt.plot(time, d,'b')
plt.plot(time, d_ref,'r')
plt.xlabel('time')
plt.ylabel('theta1')
plt.title('d vs Time')
plt.legend()
plt.show()

