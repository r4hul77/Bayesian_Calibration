import seaborn as sns
import matplotlib.pyplot as plt
import torch
import signal
import roma
import numpy as np

def plot_torch_tensors(pos_x, pos_y, x_label='pos_x', y_label='pos_y', label="Data", scale=True):
    # Convert tensors to numpy arrays


    # Set the same limits for x and y axes
    limits = torch.stack([pos_x, pos_y]).flatten().min(), torch.stack([pos_x, pos_y]).flatten().max()

    pos_x = pos_x.to("cpu").numpy()
    pos_y = pos_y.to("cpu").numpy()

    # Create a scatter plot using Seaborn
    sns.scatterplot(x=pos_x.flatten(), y=pos_y.flatten(),  label=label, alpha=0.5)

    # Set the limits for the x and y axes
    if scale:
        plt.gca().set_aspect('equal', adjustable='box')

    # Add axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Show the plot

def plot_multiple_tensors(list_of_dicts, xlabel='pos_x', ylabel='pos_y', scale=True):
    for list in list_of_dicts:
        plot_torch_tensors(**list, scale=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_acc_data(acc_data, sample_rate, og_data=None, og_sample_rate=None):
    """
    Plot the accelerometer data in the x, y, and z axes.
    
    Parameters:
        - acc_data (torch.Tensor): Tensor of shape (N, 3) containing the accelerometer data.
        - sample_rate (float): The sampling rate of the accelerometer data.
    """
    if og_sample_rate is not None:
        time_og = torch.arange(len(og_data)).to("cpu") / og_sample_rate
    time = torch.arange(len(acc_data)).to("cpu") / sample_rate
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(time.to("cpu"), acc_data[:, 0].to("cpu"), label='IMU Data')
    if og_sample_rate is not None:
        ax[0].plot(time_og.to("cpu"), og_data[:, 0].to("cpu"), label='Original Data')
    ax[0].set_ylabel('X Acceleration (m/s^2)')
    ax[1].plot(time.to("cpu"), acc_data[:, 1].to("cpu"), label='IMU Data')
    if og_sample_rate is not None:
        ax[1].plot(time_og.to("cpu"), og_data[:, 1].to("cpu"), label='Original Data')
    ax[1].set_ylabel('Y Acceleration (m/s^2)')
    ax[2].plot(time.to("cpu"), acc_data[:, 2].to("cpu"), label='IMU Data')
    if og_sample_rate is not None:
        ax[2].plot(time_og.to("cpu"), og_data[:, 2].to("cpu"), label='Original Data')
    ax[2].set_ylabel('Z Acceleration (m/s^2)')
    ax[2].set_xlabel('Time (s)')
    plt.show()

def plot_angular_velocity_data(ang_vel_data, sample_rate, og_data=None, og_sample_rate=None):
    """
    Plot the angular velocity data in the x, y, and z axes.
    
    Parameters:
        - ang_vel_data (torch.Tensor): Tensor of shape (N, 3) containing the angular velocity data.
        - sample_rate (float): The sampling rate of the angular velocity data.
        - og_data (torch.Tensor, optional): Tensor of shape (M, 3) containing the original angular velocity data
          (if available) for comparison. Default is None.
        - og_sample_rate (float, optional): The sampling rate of the original angular velocity data. Default is None.
    """
    if og_sample_rate is not None:
        time_og = torch.arange(len(og_data)) / og_sample_rate
    time = torch.arange(len(ang_vel_data)) / sample_rate
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(time, ang_vel_data[:, 0], label='IMU Data')
    if og_sample_rate is not None:
        ax[0].plot(time_og, og_data[:, 0], label='Original Data')
    ax[0].set_ylabel('X Angular Velocity (rad/s)')
    ax[1].plot(time, ang_vel_data[:, 1], label='IMU Data')
    if og_sample_rate is not None:
        ax[1].plot(time_og, og_data[:, 1], label='Original Data')
    ax[1].set_ylabel('Y Angular Velocity (rad/s)')
    ax[2].plot(time, ang_vel_data[:, 2], label='IMU Data')
    if og_sample_rate is not None:
        ax[2].plot(time_og, og_data[:, 2], label='Original Data')
    ax[2].set_ylabel('Z Angular Velocity (rad/s)')
    ax[2].set_xlabel('Time (s)')
    plt.show()

def plot_acc_histogram(acc_data, bins=50, og_data=None):
    """
    Plot a histogram of the accelerometer data in the x, y, and z axes.
    
    Parameters:
        - acc_data (torch.Tensor): Tensor of shape (N, 3) containing the accelerometer data.
        - bins (int): The number of bins to use in the histogram.
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax[0].hist(acc_data[:, 0], bins=bins, label='IMU Data')
    if og_data is not None:
        ax[0].hist(og_data[:, 0], bins=bins, alpha=0.5, label='Original Data')
    ax[0].set_ylabel('Frequency')
    ax[1].hist(acc_data[:, 1], bins=bins, label="IMU Data")
    if og_data is not None:
        ax[1].hist(og_data[:, 1], bins=bins, alpha=0.5, label='Original Data')
    ax[1].set_ylabel('Frequency')
    ax[2].hist(acc_data[:, 2], bins=bins)
    if og_data is not None:
        ax[2].hist(og_data[:, 2], bins=bins, alpha=0.5, label='Original Data')
    ax[2].set_ylabel('Frequency')
    ax[2].set_xlabel('Acceleration (m/s^2)')
    plt.show()

def plot_acc_psd(acc_data, sample_rate):
    """
    Plot the power spectral density of the accelerometer data in the x, y, and z axes.
    
    Parameters:
        - acc_data (torch.Tensor): Tensor of shape (N, 3) containing the accelerometer data.
        - sample_rate (float): The sampling rate of the accelerometer data.
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i in range(3):
        f, Pxx = signal.welch(acc_data[:, i], fs=sample_rate, nperseg=256, scaling='density')
        ax[i].semilogy(f, Pxx)
        ax[i].set_ylabel('Power Spectral Density')
    ax[2].set_xlabel('Frequency (Hz)')
    plt.show()

def plot_angvel_histogram(angvel_data, bins=50, og_data=None):
    """
    Plot a histogram of the angular velocity data in the x, y, and z axes.
    
    Parameters:
        - angvel_data (torch.Tensor): Tensor of shape (N, 3) containing the angular velocity data.
        - bins (int): The number of bins to use in the histogram.
        - og_data (torch.Tensor): Tensor of shape (M, 3) containing the original angular velocity data.
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax[0].hist(angvel_data[:, 0], bins=bins, label='IMU Data')
    if og_data is not None:
        ax[0].hist(og_data[:, 0], bins=bins, alpha=0.5, label='Original Data')
    ax[0].set_ylabel('Frequency')
    ax[1].hist(angvel_data[:, 1], bins=bins, label="IMU Data")
    if og_data is not None:
        ax[1].hist(og_data[:, 1], bins=bins, alpha=0.5, label='Original Data')
    ax[1].set_ylabel('Frequency')
    ax[2].hist(angvel_data[:, 2], bins=bins)
    if og_data is not None:
        ax[2].hist(og_data[:, 2], bins=bins, alpha=0.5, label='Original Data')
    ax[2].set_ylabel('Frequency')
    ax[2].set_xlabel('Angular Velocity (rad/s)')
    plt.show()
def plot_coordinate_axes(quaternion):
    """
    Plot the coordinate axes based on a rotation quaternion.
    
    Parameters:
        - quaternion (torch.Tensor): Tensor of shape (4,) representing the rotation quaternion.
    """
    # Normalize quaternion
    R = roma.unitquat_to_rotmat(quaternion.to("cpu"))
    
    # Define axes
    axes = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    
    # Rotate axes by rotation matrix
    rotated_axes = R @ axes.T
    
    x_axis = rotated_axes[:, 0]
    y_axis = rotated_axes[:, 1]
    z_axis = rotated_axes[:, 2]
    
    # Plot axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], 'r', label='X')
    ax.text(x_axis[0], x_axis[1], x_axis[2], 'X')
    ax.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], 'g', label='Y')
    ax.text(y_axis[0], y_axis[1], y_axis[2], 'Y')
    ax.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], 'b', label='Z')
    ax.text(z_axis[0], z_axis[1], z_axis[2], 'Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_axes(q_real, q_calc):
    
    def get_quat_axes(quaternion):    
        R = roma.unitquat_to_rotmat(quaternion.to("cpu"))
        
        # Define axes
        axes = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        
        # Rotate axes by rotation matrix
        rotated_axes = R @ axes.T
        
        x_axis = rotated_axes[:, 0]
        y_axis = rotated_axes[:, 1]
        z_axis = rotated_axes[:, 2]
        return x_axis, y_axis, z_axis    
    # Plot axes
    x_axis, y_axis, z_axis = get_quat_axes(q_real)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], 'r', label='X-R')
    ax.text(x_axis[0], x_axis[1], x_axis[2], 'X-R')
    ax.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], 'g', label='Y-R')
    ax.text(y_axis[0], y_axis[1], y_axis[2], 'Y-R')
    ax.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], 'b', label='Z-R')
    ax.text(z_axis[0], z_axis[1], z_axis[2], 'Z-R')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x_axis, y_axis, z_axis = get_quat_axes(q_calc)
    ax.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], 'r', label='X-C')
    ax.text(x_axis[0], x_axis[1], x_axis[2], 'X-C')
    ax.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], 'g', label='Y-C')
    ax.text(y_axis[0], y_axis[1], y_axis[2], 'Y-C')
    ax.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], 'b', label='Z-C')
    ax.text(z_axis[0], z_axis[1], z_axis[2], 'Z-C')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
