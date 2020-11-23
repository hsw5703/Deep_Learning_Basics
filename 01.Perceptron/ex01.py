# and gate : MCP Neuron

def AND(x1, x2) :
    w1, w2, theta = 0.5, 0.5, 0.7
    a = w1 * x1 + w2 * x2

    if a < theta :
        return 0
    else :
        return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))