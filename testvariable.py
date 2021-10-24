from variable import Variable
'''
x_1 = Variable(name = 'x_1')
x_2 = Variable(name = 'x_2')
x_3 = Variable(name = 'x_3')

z = x_1 + x_2 + x_3
for i in range(100):
    for j in range(100):
        for k in range(100):
            assert(i+j+k == z(x_1 = i, x_2 = j, x_3 = k))

print(z.gradient(x_1 = 1, x_2 = 3, x3 = 5))


z = Variable.exp(x_1 + x_2**2) + 3 * Variable.log(27 - x_1 * x_2 * x_3)
print(z(x_1 = 3, x_2 = 1, x_3 = 7))
print(z.gradient(x_1 = 3, x_2 = 1, x_3 = 7))
'''
'''
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# PROVES THAT THE SAME INSTANCE OF VARIABLE CAN BE USED MULTIPLE TIMES
x_1 = Variable(name = 'x_1')
y_1 = Variable(name = 'y_1')
cost = x_1 * y_1 + x_1/y_1

print(cost(x_1 = 121, y_1 = 34))
print(cost.gradient(x_1 = 121, y_1 = 34))
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''

x_1 = Variable(name = 'x_1')
y_1 = Variable(name = 'y_1')
cost_1 = x_1/y_1
cost_final = Variable.exp(cost_1) + cost_1

print(cost_final(x_1 = 143, y_1 = 150))

print(cost_final.gradient(x_1 = 143, y_1 = 150))
