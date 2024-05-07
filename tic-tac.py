from neural import NeuralNet

# print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# xor_data = [
#     ([0, 0], [0]),
#     ([1, 0], [1]),
#     ([0, 1], [1]),
#     ([0, 0], [0])
# ]

# xor_nn = NeuralNet(2, 2, 1)
# xor_nn.train(xor_data, iters=1000, print_interval=100)

# print(xor_nn.test_with_expected(xor_data))

print("<<<<<<<<<<<<<< Tic Tac Toe Data >>>>>>>>>>>>>>\n")

tictactoe_data = [
    ([.9, .6, .8, .3, .1], [1.0]),
    ([.8, .8, .4, .6, .4], [1.0]),
    ([.7, .2, .4, .6, .3], [1.0]),
    ([.5, .5, .8, .4, .8], [0.0]),
    ([.3, .1, .6, .8, .8], [0.0]),
    ([.6, .3, .4, .3, .6], [0.0])
]

tic_nn = NeuralNet(5, 10, 1)

tic_nn.train(tictactoe_data)

print(tic_nn.test_with_expected(tictactoe_data))

tic_test = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]

print(tic_nn.test(tic_test))