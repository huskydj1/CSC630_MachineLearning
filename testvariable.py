'''from variable import Variable

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
print(z.gradient(x_1 = 3, x_2 = 1, x_3 = 7))'''


class temporaryclass:
    def __init__(self, value):
        self.value = value
    def __add__(self, other):
        #print(type(self), type(other))
        return 0
    def __radd__(self, other):
        return 0

temporarylist = [temporaryclass(i) for i in range(0, 10)]

print(sum(temporarylist))