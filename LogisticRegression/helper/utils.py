import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x, derivative=False):
    prob = 1 / ( 1 + np.exp(-x))
    
    if derivative:
        d_sigmoid = prob * (1 - prob)
        return d_sigmoid
    return prob

# Hypothesis function
def h_theta(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def plot_h_theta_1d(b, w):
    x_min = -6
    x_max = 6
    x = np.linspace(x_min, x_max, 101)
    N = len(x)
    x = x.reshape((N,1))
    w = np.array(w).reshape((1,1))
    plt.figure(figsize=(10,2))
    plt.plot(x, h_theta(x, w, b), linewidth=2)
    plt.plot(x, sigmoid((x @ w + b), derivative=True))

    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)

    plt.xlim(x_min, x_max)
    plt.ylim(-0.5, 1.5)
    plt.show()


def plot_linear_hyperplane(b, w_magnitude, w_angle):
    w1 = w_magnitude * np.cos(w_angle)
    w2 = w_magnitude * np.sin(w_angle)

    x_min = -6
    x_max = 6

    N1 = 101
    N = N1**2

    x1 = np.linspace(x_min, x_max, N1)
    x2 = np.linspace(x_min, x_max, N1)
    X1, X2 = np.meshgrid(x1,x2)

    X1_flat = X1.reshape((N,1))
    X2_flat = X2.reshape((N,1))
    X = np.hstack((X1_flat, X2_flat))

    w = np.array((w1, w2)).reshape((2,1))

    Y = ((X@w)+b).reshape((N1, N1))

    fig = plt.figure(figsize=(12,4))

    ax = plt.subplot(1,2,1, projection='3d')
    ax.plot_surface(X1, X2, Y, alpha=0.3, rstride=2, cstride=2, linewidth=3)
    ax.plot_wireframe(X1, X2, Y, rstride=10, cstride=10)

    ax.quiver(0, 0, 0, w1, w2, 0, arrow_length_ratio=0.1, linewidth=2, color='r', label='w=[{:0.2f}, {:0.2f}]'.format(w1, w2))
    ax.legend()

    ax.set_zlim(x_min, x_max)

    # Setting up perspective of viewing the surface plot
    view_elev = 10
    view_azim = -60
    ax.view_init(view_elev, view_azim)


    ax.set_xlabel('x1', fontsize = 16)
    ax.set_ylabel('x2', fontsize = 16)
    ax.set_zlabel('y', fontsize = 16)


    # Ploting contour
    ax = plt.subplot(1, 2, 2)
    contour_plot = ax.contour(X1, X2, Y, extent = (0,0,1,1))
    ax.clabel(contour_plot, inline=1, fontsize=12)
    ax.set_title('contour plot', fontsize = 16)
    ax.set_xlabel('x1', fontsize = 16)
    ax.set_ylabel('x2', fontsize = 16)
    ax.set_aspect('equal')

    ax.quiver(0, 0, w1, w2, angles="xy", scale_units="xy", scale=1, linewidth=2, color='r', label='w=[{:0.2f}, {:0.2f}]'.format(w1, w2))
    ax.legend()

    plt.show()




# Updated plot function
def plot_linear_logistic_2d(b, w_magnitude, w_angle):
    # Calculate w1 and w2 from magnitude and angle
    w1 = w_magnitude * np.cos(w_angle)
    w2 = w_magnitude * np.sin(w_angle)

    # Create meshgrid for feature space
    x_min = -6
    x_max = 6
    N1 = 101
    N = N1**2

    x1 = np.linspace(x_min, x_max, N1)
    x2 = np.linspace(x_min, x_max, N1)
    X1, X2 = np.meshgrid(x1, x2)

    X1_flat = X1.reshape((N, 1))
    X2_flat = X2.reshape((N, 1))
    X = np.hstack((X1_flat, X2_flat))

    # Define weights vector
    w = np.array([w1, w2]).reshape((2, 1))

    # Get the predicted probabilities (Y) using the logistic hypothesis
    Y = h_theta(X, w, b).reshape((N1, N1))

    fig = plt.figure(figsize=(12, 6))

    # 3D surface plot of the logistic regression surface
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X1, X2, Y, alpha=0.3, rstride=2, cstride=2, linewidth=3)
    ax.plot_wireframe(X1, X2, Y, rstride=10, cstride=10)

    # Show the weight vector w=[w1, w2]
    ax.quiver(0, 0, 0, w1, w2, 0, arrow_length_ratio=0.1, linewidth=2, color='r', label='w=[{:0.2f}, {:0.2f}]'.format(w1, w2))
    ax.legend()

    # View settings
    ax.view_init(elev=10, azim=-60)
    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_zlabel('y', fontsize=16)

    # 2D contour plot showing decision boundary and contours
    ax = plt.subplot(1, 2, 2)

    # Decision boundary: Where the logistic function equals 0.5
    decision_boundary = ax.contour(X1, X2, Y, levels=[0.5], colors='red')
    ax.clabel(decision_boundary, inline=1, fontsize=12)

    # Contour plot for different probability levels
    contour_plot = ax.contourf(X1, X2, Y, alpha=0.7, cmap=plt.cm.coolwarm)
    plt.colorbar(contour_plot, ax=ax)

    # Add weight vector to contour plot
    ax.quiver(0, 0, w1, w2, angles="xy", scale_units="xy", scale=1, linewidth=2, color='r', label='w=[{:0.2f}, {:0.2f}]'.format(w1, w2))
    ax.legend()

    # Set axis labels and title
    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_aspect('equal')
    ax.set_title('Decision Boundary with Contours', fontsize=16)

    plt.show()