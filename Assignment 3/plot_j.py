import matplotlib.pyplot as plt


def plot(all_J):
    plt.plot(all_J[0:300])
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.title("Cost function using Gradient Descent")
    plt.show()