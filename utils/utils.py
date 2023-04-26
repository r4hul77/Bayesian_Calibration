import torch
import roma

def add_gauss_noise(data, sigma):
    noise = torch.randn_like(data) * sigma
    return data + noise


def generate_gps_data(pos, sigma, drop):
    ret = pos[::drop, :]
    return add_gauss_noise(ret, sigma)

def euler_to_rot(euler_angles): #Tested
    # Extract the individual Euler angles
    alpha, beta, gamma = euler_angles[0], euler_angles[1], euler_angles[2]

    # Compute the sin and cosine of each angle
    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    cg = torch.cos(gamma)
    sa = torch.sin(alpha)
    sb = torch.sin(beta)
    sg = torch.sin(gamma)

    # Construct the rotation matrix
    R = torch.zeros((3, 3), dtype=torch.float32)
    R[0, 0] = cb * cg
    R[0, 1] = -cb * sg
    R[0, 2] = sb
    R[1, 0] = ca * sg + sa * sb * cg
    R[1, 1] = ca * cg - sa * sb * sg
    R[1, 2] = -sa * cb
    R[2, 0] = sa * sg - ca * sb * cg
    R[2, 1] = sa * cg + ca * sb * sg
    R[2, 2] = ca * cb

    return R

def quat_to_euler(quat): #Tested
    # Extract the individual quaternion components
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]

    # Compute the Euler angles
    alpha = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    beta = torch.asin(2 * (q0 * q2 - q3 * q1))
    gamma = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    return torch.stack([alpha, beta, gamma])


    
    

def euler_to_quat(euler_angles): #Tested
    # Extract the individual Euler angles
    alpha, beta, gamma = euler_angles[0], euler_angles[1], euler_angles[2]

    # Compute the sin and cosine of each angle
    ca = torch.cos(alpha / 2)
    cb = torch.cos(beta / 2)
    cg = torch.cos(gamma / 2)
    sa = torch.sin(alpha / 2)
    sb = torch.sin(beta / 2)
    sg = torch.sin(gamma / 2)

    # Construct the quaternion
    q = torch.zeros((4,), dtype=torch.float32, device=euler_angles.device)
    q[0] = ca * cb * cg + sa * sb * sg
    q[1] = sa * cb * cg - ca * sb * sg
    q[2] = ca * sb * cg + sa * cb * sg
    q[3] = ca * cb * sg - sa * sb * cg

    return q

def generate_imu_data(accel, sigma, drop, rot, ang_vel, sigma_ang_vel):
    accel_new = accel[::drop, :]
    ang_vel_new = ang_vel[::drop, :]
    quats = rot.repeat(accel_new.shape[0], 1)
    accel_new = roma.quat_action(quats, accel_new, is_normalized=True)
    ang_vel_new = roma.quat_action(quats, ang_vel_new, is_normalized=True)
    return add_gauss_noise(accel_new, sigma), add_gauss_noise(ang_vel_new, sigma_ang_vel)
