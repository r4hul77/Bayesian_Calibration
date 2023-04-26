import torch
import roma
import numpy as np
import logging

# Wire up the logger with a file name a format such that it shows up the function name where it was called for better debuging experience
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)


class BaseCalibrator:
    
    def __init__(self, accel_noise, pos_noise, sample_size, horizon_size, init_pos, imu_dt, gps_dt) -> None:
        self.accel_noise = accel_noise
        self.pos_noise = pos_noise
        self.sample_size = sample_size
        self.horizon_size = horizon_size
        self.gps_recived = False
        self.imu_recived = False
        self.imu_positions = torch.randn(self.sample_size, 3, dtype=torch.float32) * self.pos_noise + init_pos
        self.quaternions = generate_random_quaternions(self.sample_size)
        self.rms_errors = torch.empty(self.sample_size, self.horizon_size, dtype=torch.float32)
        self.grav_vec = torch.ones_like(self.imu_positions) * torch.tensor([0, 0, 9.81])
        logging.debug(f"Initial Gravity Vector {self.grav_vec.mean(dim=0)} with shape {self.grav_vec.shape}")
        self.imu_dt = imu_dt
        self.gps_dt = gps_dt
        self.gps_pos = torch.zeros(3, dtype=torch.float32)
        self.imu_cnt = 0
        self.gps_cnt = 0
        self.pivot = 0 
        
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
        logging.debug(f"IMU Positions Mean {self.imu_positions.mean(dim=0)} with shape {self.imu_positions.shape}")
        self.imu_positions += (roma.quat_action(inv_quats, imu_data_tx, is_normalized=True) + self.grav_vec) * self.imu_dt* self.imu_dt
        logging.debug(f"IMU Positions Mean {self.imu_positions.mean(dim=0)} with shape {self.imu_positions.shape}")
        self.imu_cnt += 1
        return self.imu_positions
    
    def update_gps(self, gps_data):
        self.gps_recived = True
        self.rms_errors[:, self.pivot] = torch.norm(self.imu_positions - gps_data, dim=1)
        self.gps_pos = gps_data
        self.gps_cnt += 1
        self.pivot += 1
        self.pivot%=self.horizon_size
    
    def horizon_size_reached(self):
        return self.gps_cnt >= self.horizon_size
    
    def update(self, imu_data, gps_data):
        postier = self.quaternions
        out_crieterion = (self.imu_positions, self.rms_errors)
        if(imu_data is not None):
            self.update_imu(imu_data)
        if(gps_data is not None):
            self.update_gps(gps_data)
        if(self.gps_recived and self.imu_recived):
            if(self.horizon_size_reached()):
                postier, out_crieterion = self.calculate_postier()
        return postier, out_crieterion

    def calculate_postier(self):
        logging.debug(f"Calculating Postier")
        logging.debug(f"Shgape of RMS Errors {self.rms_errors.shape}")
        probs = torch.softmax(-self.rms_errors, dim=0)
        logging.debug(f"Probs shape {probs.shape}")
        resampled = torch.multinomial(torch.squeeze(probs), self.sample_size, replacement=True)
        logging.debug(f"Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        logging.debug(f"Means of positions {self.imu_positions.mean(dim=0)} with shape {self.imu_positions.shape}")
        logging.debug(f"Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        quats = self.quaternions[resampled]
        positions = self.imu_positions[resampled]
        rms = self.rms_errors[resampled]
        self.quaternions = quats
        self.imu_positions = torch.rand_like(self.imu_positions) * self.pos_noise + self.gps_pos
        logging.debug(f"Positions Shape {self.imu_positions.shape}")
        self.rms_errors = torch.zeros_like(self.rms_errors)
        logging.debug(f"ReSampled Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        logging.debug(f"ReSampled Means of positions {self.imu_positions.mean(dim=0)} with shape {self.imu_positions.shape}")
        logging.debug(f"ReSampled Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        print(f"Mean of Quats {quats.mean(dim=0)}")       
        return quats, (positions, rms)
            
            


def generate_random_quaternions(num_quats):
    """
    Generates `num_quats` random quaternions using a uniform distribution in PyTorch.

    Args:
        - num_quats (int): The number of random quaternions to generate.

    Returns:
        - quats (torch.Tensor): A tensor of shape (num_quats, 4) containing the generated quaternions.
    """
    quats = torch.empty(num_quats, 4, dtype=torch.float32)
    quats[:, 0].uniform_(-1, 1)
    quats[:, 1].uniform_(-1, 1)
    quats[:, 2].uniform_(-1, 1)
    quats[:, 3].uniform_(-1, 1)
    quats /= torch.norm(quats, dim=1, keepdim=True)
    return quats