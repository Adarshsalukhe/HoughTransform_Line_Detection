import numpy as np
import matplotlib.pyplot as plt

def plot_hough_space():
    points = [(10, 10), (20, 20), (30, 30)]
    thetas = np.linspace(0, np.pi, 180)
    plt.figure()

    for x, y in points:
        rhos = x * np.cos(thetas) + y * np.sin(thetas)
        plt.plot(thetas, rhos, label=f"Point ({x}, {y})")

    plt.xlabel("Theta (radians)")
    plt.ylabel("Rho")
    plt.legend()
    plt.title("Hough Space Representation")
    plt.grid()
    plt.show()

plot_hough_space()
