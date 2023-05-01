import torch
import roma
import numpy as np
import logging
import utils
from torch.utils.tensorboard import SummaryWriter

# Wire up the logger with a file name a format such that it shows up the function name where it was called for better debuging experience
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

class BaseCalibrator:
    
    def __init__(self, accel_noise, pos_noise, sample_size, horizon_size, init_pos, imu_dt, gps_dt, vel_noise=torch.zeros((1,3), dtype=torch.float32), init_vel = torch.zeros((1,3)), logging_dir=None, real_q=None, init_quats=None) -> None:
        self.accel_noise = accel_noise
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.init_state = torch.concat([init_pos, init_vel])
        self.state_noise = torch.concat([self.pos_noise, self.vel_noise])
        self.sample_size = sample_size
        self.horizon_size = horizon_size
        self.gps_recived = False
        self.imu_recived = False
        #Use States ?
        logging.debug(f"Sizes of Tensors:\n self.states = {torch.randn(self.sample_size, 6, dtype=torch.float32).shape}"
                      f"\nself.state noise = {self.state_noise.shape}\ninit_state = {self.init_state.shape}")
        self.states = torch.randn(self.sample_size, 6, dtype=torch.float32) * self.state_noise + self.init_state
        if(init_quats is not None):
            self.quaternions = init_quats
            assert self.quaternions.shape == (self.sample_size, 4), f"Quaternions must be of shape ({sample_size}, 4) but recived {self.quaternions.shape}" 
        else:
            self.quaternions = generate_random_quaternions(self.sample_size)
        self.rms_errors = torch.empty(self.sample_size, self.horizon_size, dtype=torch.float32)
        self.grav_vec = torch.tensor([[0, 0, 9.81]])
        self.imu_dt = imu_dt
        self.gps_dt = gps_dt
        temp_tensor = torch.hstack([torch.zeros(3, 3, dtype=torch.float32), torch.eye(3, dtype=torch.float32)])
        logging.debug(f"Temp Tensor Shape {temp_tensor.shape}")
        self.A = torch.eye(6, dtype=torch.float32)  + torch.vstack([temp_tensor * self.imu_dt, torch.zeros(3, 6)])
        self.B = torch.tensor([[0.5*self.imu_dt*self.imu_dt, 0, 0],
                               [0, 0.5*self.imu_dt*self.imu_dt, 0],
                               [0, 0, 0.5*self.imu_dt*self.imu_dt],
                               [self.imu_dt, 0, 0],
                               [0, self.imu_dt, 0],
                               [0, 0, self.imu_dt]])
        logging.debug(f"Initial Gravity Vector {self.grav_vec.mean(dim=0)} with shape {self.grav_vec.shape}")
        self.gps_pos = torch.zeros(3, dtype=torch.float32)
        self.imu_cnt = 0
        self.gps_cnt = 0
        self.pivot = 0
        self.real_q = real_q
        print("Initialized Base Calibrator")
        self.writer = None 
        if(logging_dir is not None):
            self.writer = SummaryWriter(logging_dir)
            self.summarize_states()
            self.summarize_quaternions()
            
            
    
    def summarize_rms(self):
        self.writer.add_histogram("RMS Errors", self.rms_errors, self.gps_cnt)
    
    def summarize_states(self):
        axis = ["X", "Y", "Z"]
        for i in range(3):
            self.writer.add_histogram(f"States/Positions/{axis[i]}", self.states[:, i], self.imu_cnt)
            self.writer.add_histogram(f"States/Velocities/{axis[i]}", self.states[:, i], self.imu_cnt)
            self.writer.add_scalar(f"States/Positions/Mean/{axis[i]}", self.states[:, i].mean(), self.imu_cnt)
            self.writer.add_scalar(f"States/Velocities/Mean/{axis[i]}", self.states[:, i].mean(), self.imu_cnt)
            self.writer.add_scalar(f"States/Positions/Std/{axis[i]}", self.states[:, i].std(), self.imu_cnt)
            self.writer.add_scalar(f"States/Velocities/Std/{axis[i]}", self.states[:, i].std(), self.imu_cnt)
            self.writer.add_scalar(f"States/Positions/Max/{axis[i]}", self.states[:, i].max(), self.imu_cnt)
            self.writer.add_scalar(f"States/Velocities/Max/{axis[i]}", self.states[:, i].max(), self.imu_cnt)
            self.writer.add_scalar(f"States/Positions/Min/{axis[i]}", self.states[:, i].min(), self.imu_cnt)
            self.writer.add_scalar(f"States/Velocities/Min/{axis[i]}", self.states[:, i].min(), self.imu_cnt)
            self.writer.add_scalar(f"States/Positions/Median/{axis[i]}", self.states[:, i].median(), self.imu_cnt)
            self.writer.add_scalar(f"States/Velocities/Median/{axis[i]}", self.states[:, i].median(), self.imu_cnt)
    
    
    def summarize_quaternions(self):
        axis = ["X", "Y", "Z", "W"]
        for i, ax in enumerate(axis):
            self.writer.add_histogram(f"Quaternions/{ax}", self.quaternions[:, i], self.gps_cnt)
            self.writer.add_scalar(f"Quaternions/Mean/{ax}", self.quaternions[:, i].mean(), self.gps_cnt)
            self.writer.add_scalar(f"Quaternions/Std/{ax}", self.quaternions[:, i].std(), self.gps_cnt)
            self.writer.add_scalar(f"Quaternions/Max/{ax}", self.quaternions[:, i].max(), self.gps_cnt)
            self.writer.add_scalar(f"Quaternions/Min/{ax}", self.quaternions[:, i].min(), self.gps_cnt)
        if self.real_q is not None:
            self.writer.add_histogram("Quaternions/Error", utils.utils.delta_q(q1=self.quaternions, real_q=self.real_q), self.gps_cnt)
        
        
    def sample(self, prior, crieterion):
        posterior = prior
        return posterior, crieterion
    
    def update_imu(self, imu_data):
        logging.debug(f"Updating IMU Recived Data {imu_data} with shape {imu_data.shape}")
        self.imu_recived = True
        inv_quats = roma.quat_inverse(self.quaternions)
        logging.debug(f"Inversed Quaternions {inv_quats} with shape {inv_quats.shape}")
        imu_data_tx = torch.randn(self.sample_size, 3, dtype=torch.float32)*self.accel_noise + torch.unsqueeze(imu_data, dim=0)
        logging.debug(f"IMU Data Transformed to shape {imu_data_tx.shape}")
        accel = (roma.quat_action(inv_quats, imu_data_tx, is_normalized=True) + self.grav_vec)
        logging.debug(f"Accel {accel.mean(dim=0)} with shape {accel.shape}")
        #self.imu_positions += 0.5*accel * self.imu_dt**2
        self.states = (torch.matmul(self.A, self.states.transpose(1, 0)) + torch.matmul(self.B, accel.transpose(1,0))).transpose(1, 0)
        self.summarize_states()
        self.imu_cnt += 1
        return self.states
    
    def update_gps(self, gps_data):
        self.gps_recived = True
        self.rms_errors[:, self.pivot] = torch.norm(self.states[:, :3] - gps_data, dim=1)
        self.gps_pos = gps_data
        self.gps_cnt += 1
        self.pivot += 1
        self.pivot%=self.horizon_size
    
    def horizon_size_reached(self):
        return self.gps_cnt >= self.horizon_size
    
    def update(self, imu_data, gps_data):
        postier = self.quaternions
        out_crieterion = (self.states, self.rms_errors)
        if(imu_data is not None):
            self.update_imu(imu_data)
        if(gps_data is not None):
            self.update_gps(gps_data)
        if(self.gps_recived and self.imu_recived):
            if(self.horizon_size_reached()):
                postier, out_crieterion = self.calculate_postier()
                self.summarize_quaternions()
                self.summarize_rms()
        return postier, out_crieterion

    def calculate_postier(self):
        logging.debug(f"Calculating Postier")
        logging.debug(f"Shgape of RMS Errors {self.rms_errors.shape}")
        probs = torch.softmax(-self.rms_errors, dim=0)
        self.writer.add_histogram("Postier/Probabilities", probs, self.gps_cnt)
        logging.debug(f"Probs shape {probs.shape}")
        resampled = torch.multinomial(torch.squeeze(probs), self.sample_size, replacement=True)
        logging.debug(f"Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        logging.debug(f"Means of positions {self.states.mean(dim=0)} with shape {self.states.shape}")
        logging.debug(f"Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        quats = self.quaternions[resampled]
        positions = self.states[resampled]
        self.states = positions
        rms = self.rms_errors[resampled]
        self.quaternions = quats
        self.rms_errors = torch.zeros_like(self.rms_errors)
        logging.debug(f"ReSampled Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        logging.debug(f"ReSampled Means of positions {self.states.mean(dim=0)} with shape {self.states.shape}")
        logging.debug(f"ReSampled Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        return quats, (self.states, rms)
            
            


def generate_random_quaternions(num_quats):
    """
    Generates `num_quats` random quaternions using a uniform distribution in PyTorch.

    Args:
        - num_quats (int): The number of random quaternions to generate.

    Returns:
        - quats (torch.Tensor): A tensor of shape (num_quats, 4) containing the generated quaternions.
    """
    return roma.random_unitquat(num_quats)
