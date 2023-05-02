import torch
import roma
import numpy as np
import logging
import utils
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Wire up the logger with a file name a format such that it shows up the function name where it was called for better debuging experience
logging.basicConfig(filename='/home/r4hul/Projects/Bayesian_Calibration/debug.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

class TorchQueue:
    
    def __init__(self, max_size, tensor_size, dtype=torch.float32) -> None:
        self.max_size = max_size
        self.tensor_size = tensor_size
        self.dtype = dtype
        self.pivot = -1
        self.tensor = torch.zeros((self.max_size, *self.tensor_size), dtype=self.dtype)
    
    def push(self, tensor):
        self.pivot = (self.pivot+1)%self.max_size
        logging.debug(f"Pushing tensor of shape {tensor} to the queue at positon {self.pivot}")
        self.tensor[self.pivot] = tensor
    
    def modify(self, tensor):
        logging.debug(f"Modifying tensor of shape {tensor} at position {self.pivot}")
        self.tensor[self.pivot] = tensor

    def get(self):
        logging.debug(f"Getting tensor of shape {self.tensor} at position {self.pivot}")
        return self.tensor[self.pivot]


class BaseCalibrator:
    
    def __init__(self, accel_noise, pos_noise, sample_size, horizon_size, init_pos, imu_dt, gps_dt, vel_noise=torch.zeros((1,3), dtype=torch.float32), init_vel = torch.zeros((1,3)), logging_dir=None, real_q=None, init_quats=None) -> None:
        self.accel_noise = accel_noise
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.init_state = torch.concat([init_pos, init_vel])
        self.state_noise = torch.concat([self.pos_noise, self.vel_noise])
        self.sample_size = sample_size
        self.horizon_size = horizon_size
        self.gps_queue = TorchQueue(self.horizon_size, (self.sample_size,3), dtype=torch.float32)
        self.states = TorchQueue(self.horizon_size, (self.sample_size,6), dtype=torch.float32)
        self.gps_recived = False
        #self.gps_queue.push(torch.randn(self.sample_size, 3, dtype=torch.float32) * self.pos_noise + init_pos)
        self.imu_recived = False
        #Use States ?
        logging.debug(f"Sizes of Tensors:\n self.states = {torch.randn(self.sample_size, 6, dtype=torch.float32).shape}"
                      f"\nself.state noise = {self.state_noise.shape}\ninit_state = {self.init_state.shape}")
        self.states.push(torch.randn(self.sample_size, 6, dtype=torch.float32) * self.state_noise + self.init_state)
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
        self.A = (torch.eye(6, dtype=torch.float32)  + torch.vstack([temp_tensor * self.imu_dt, torch.zeros(3, 6)])).repeat(self.sample_size, 1, 1)
        self.B = torch.tensor([[0.5*self.imu_dt*self.imu_dt, 0, 0],
                               [0, 0.5*self.imu_dt*self.imu_dt, 0],
                               [0, 0, 0.5*self.imu_dt*self.imu_dt],
                               [self.imu_dt, 0, 0],
                               [0, self.imu_dt, 0],
                               [0, 0, self.imu_dt]]).repeat(self.sample_size, 1, 1)
        logging.debug(f"Initial Gravity Vector {self.grav_vec.mean(dim=0)} with shape {self.grav_vec.shape}")
        self.gps_pos = torch.zeros(3, dtype=torch.float32)
        self.imu_cnt = 0
        self.gps_cnt = 0
        self.real_q = real_q
        print("Initialized Base Calibrator")
        self.writer = None 
        if(logging_dir is not None):
            self.writer = SummaryWriter(logging_dir)
            self.states_summary_cnt = 0
            self.quaternions_summary_cnt = 0
            self.rms_summary_cnt = 0
            self.summarize_states()
            self.summarize_quaternions()
            self.state_error_summary = 0
            self.imu_summary_cnt = 0
            
            
    
    def summarize_rms(self):
        errors = self.get_error()
        self.writer.add_histogram("RMS Errors", errors, self.rms_summary_cnt)
        self.writer.add_scalar("RMS Errors/Mean", errors.mean(), self.rms_summary_cnt)
        self.writer.add_scalar("RMS Errors/Std", errors.std(), self.rms_summary_cnt)
        self.writer.add_scalar("RMS Errors/Max", errors.max(), self.rms_summary_cnt)
        self.writer.add_scalar("RMS Errors/Min", errors.min(), self.rms_summary_cnt)
        self.rms_summary_cnt += 1
        
    
    def summarize_states(self, real_state=None):
        axis = ["X", "Y", "Z"]
        state = self.states.get()
        for i in range(3):
            self.writer.add_histogram(f"States/Positions/{axis[i]}", state[ :, i], self.states_summary_cnt)
            self.writer.add_histogram(f"States/Velocities/{axis[i]}", state[ :, i], self.states_summary_cnt)
            self.writer.add_scalar(f"States/Positions/Mean/{axis[i]}", state[ :, i].mean(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Velocities/Mean/{axis[i]}", state[ :, i].mean(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Positions/Std/{axis[i]}", state[ :, i].std(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Velocities/Std/{axis[i]}", state[ :, i].std(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Positions/Max/{axis[i]}", state[ :, i].max(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Velocities/Max/{axis[i]}", state[ :, i].max(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Positions/Min/{axis[i]}", state[ :, i].min(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Velocities/Min/{axis[i]}", state[ :, i].min(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Positions/Median/{axis[i]}", state[ :, i].median(), self.states_summary_cnt)
            self.writer.add_scalar(f"States/Velocities/Median/{axis[i]}", state[ :, i].median(), self.states_summary_cnt)
        self.states_summary_cnt += 1
        if real_state is not None:
            for i in range(3):
                self.writer.add_histogram(f"States/Positions/Error/{axis[i]}", state[ :, i] - real_state[ i], self.states_summary_cnt)
                self.writer.add_scalar(f"States/Positions/Error/Mean/{axis[i]}", (state[ :, i] - real_state[i]).mean(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Positions/Error/Std/{axis[i]}", (state[ :, i] - real_state[i]).std(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Positions/Error/Max/{axis[i]}", (state[ :, i] - real_state[i]).max(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Positions/Error/Min/{axis[i]}", (state[ :, i] - real_state[i]).min(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Positions/Real/{axis[i]}", real_state[i], self.states_summary_cnt)
            for i in range(3, 6):
                self.writer.add_histogram(f"States/Velocities/Error/{axis[i-3]}", state[ :, i] - real_state[ i], self.states_summary_cnt)
                self.writer.add_scalar(f"States/Velocities/Error/Mean/{axis[i-3]}", (state[ :, i] - real_state[i]).mean(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Velocities/Error/Std/{axis[i-3]}", (state[ :, i] - real_state[i]).std(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Velocities/Error/Max/{axis[i-3]}", (state[ :, i] - real_state[i]).max(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Velocities/Error/Min/{axis[i-3]}", (state[ :, i] - real_state[i]).min(), self.states_summary_cnt)
                self.writer.add_scalar(f"States/Velocities/Real/{axis[i-3]}", real_state[i], self.states_summary_cnt)
            self.state_error_summary += 1
    
    def summarize_quaternions(self):
        axis = ["X", "Y", "Z", "W"]
        logging.debug(f"Quaternions shape {self.quaternions.shape}")
        for i, ax in enumerate(axis):
            self.writer.add_histogram(f"Quaternions/{ax}", self.quaternions[:, i], self.quaternions_summary_cnt)
            self.writer.add_scalar(f"Quaternions/Mean/{ax}", self.quaternions[:, i].mean(), self.quaternions_summary_cnt)
            self.writer.add_scalar(f"Quaternions/Std/{ax}", self.quaternions[:, i].std(), self.quaternions_summary_cnt)
            self.writer.add_scalar(f"Quaternions/Max/{ax}", self.quaternions[:, i].max(), self.quaternions_summary_cnt)
            self.writer.add_scalar(f"Quaternions/Min/{ax}", self.quaternions[:, i].min(), self.quaternions_summary_cnt)
        if self.real_q is not None:
            self.writer.add_histogram("Quaternions/Error", utils.utils.delta_q(q1=self.quaternions, real_q=self.real_q), self.quaternions_summary_cnt)
        self.quaternions_summary_cnt += 1
        
    def sample(self, prior, crieterion):
        posterior = prior
        return posterior, crieterion
    
    def update_imu(self, imu_data, real_accel=None):
        logging.debug(f"Updating IMU Recived Data {imu_data} with shape {imu_data.shape}")
        self.imu_recived = True
        inv_quats = roma.quat_inverse(self.quaternions)
        logging.debug(f"Inversed Quaternions {inv_quats} with shape {inv_quats.shape}")
        imu_data_tx = torch.randn(self.sample_size, 3, dtype=torch.float32)*self.accel_noise + torch.unsqueeze(imu_data, dim=0)
        logging.debug(f"IMU Data Transformed to shape {imu_data_tx.shape}")
        accel = (roma.quat_action(inv_quats, imu_data_tx, is_normalized=True) + self.grav_vec)
        logging.debug(f"Accel {accel.mean(dim=0)} with shape {accel.shape}")
        #self.imu_positions += 0.5*accel * self.imu_dt**2
        state = self.update_state(accel, self.states.get())
        self.summarize_accels(accel, real_accel)
        
        self.imu_cnt += 1
        return state

    def update_state(self, accel, state):
        logging.debug(f"Updating State with accel {accel.shape} and state {state.shape}")
        logging.debug(f"A {self.A.shape} B {self.B.shape}")
        logging.debug(f"State {state} Accel {accel}")
        state =  (torch.bmm(self.A, state.unsqueeze(2)) + torch.bmm(self.B, accel.unsqueeze(2))).squeeze(2)
        logging.debug(f"Updated State {state}")
        return state

    def summarize_accels(self, accel, real_accel):
        axis = ["X", "Y", "Z"]
        for i in range(3):
            self.writer.add_scalar(f"Accel/Mean/{axis[i]}", accel[:, i].mean(), self.imu_summary_cnt)

        if real_accel is not None:
            for i in range(3):
                self.writer.add_histogram(f"Accel/Error/{axis[i]}", accel[:, i] - real_accel[i], self.imu_summary_cnt)
                self.writer.add_scalar(f"Accel/Error/Mean/{axis[i]}", (accel[:, i] - real_accel[i]).mean(), self.imu_summary_cnt)
                self.writer.add_scalar(f"Accel/Error/Std/{axis[i]}", (accel[:, i] - real_accel[i]).std(), self.imu_summary_cnt)
                self.writer.add_scalar(f"Accel/Real/{axis[i]}", real_accel[i], self.imu_summary_cnt)                
        self.imu_summary_cnt += 1
            
    
    def update_gps(self, gps_data):
        self.gps_recived = True
        self.gps_queue.push(gps_data)
        self.gps_cnt += 1

    
    def horizon_size_reached(self):
        return self.gps_cnt >= self.horizon_size
    
    def update(self, imu_data, gps_data, real_state=None, real_accel=None):
        postier = self.quaternions
        out_crieterion = (self.states, self.rms_errors)
        if(imu_data is not None):
            state = self.update_imu(imu_data, real_accel)
            self.states.modify(state)
        if(gps_data is not None):
            logging.debug(f"Updating GPS Queue with data {self.gps_queue.tensor} and gps data {gps_data} with shape {gps_data.shape}")
            self.update_gps(gps_data)
            logging.debug(f"GPS Queue {self.gps_queue.tensor} with shape {self.gps_queue.tensor.shape}")
            logging.debug(f"States {self.states.tensor} with shape {self.states.tensor.shape} is being pushed  with state {self.states.get()} with shape {self.states.get().shape}")
            
            logging.debug(f"States {self.states.tensor} with shape {self.states.tensor.shape}")
            if(self.gps_recived and self.imu_recived):
                if(self.horizon_size_reached()):
                    postier, out_crieterion = self.calculate_postier()
                    self.summarize_quaternions()
                    self.summarize_rms()
                    self.summarize_states(real_state)
            self.states.push(self.states.get())
        self.summarize_states()
        return postier, out_crieterion

    def get_error(self):
        logging.debug(f"Self States Shape {self.states.tensor.shape} and GPS Queue Shape {self.gps_queue.tensor.shape}")
        error = torch.sum(torch.norm(self.gps_queue.tensor - self.states.tensor[:, :, :3], dim=2), dim=0)
        logging.debug(f"Error {error} with shape {error.shape}")
        logging.debug(f"GPS Queue {self.gps_queue.tensor} with shape {self.gps_queue.tensor.shape}")
        logging.debug(f"States {self.states.tensor} with shape {self.states.tensor.shape}")
        logging.debug(f"Error Shape {error.shape}")
        self.rms_errors = error
        return error

    def calculate_postier(self):
        #logging.debug(f"Calculating Postier")
        #logging.debug(f"Shgape of RMS Errors {self.rms_errors.shape}")
        probs = torch.softmax(-self.get_error(), dim=0)
        self.writer.add_histogram("Postier/Probabilities", probs, self.gps_cnt)
        logging.debug(f"Probs shape {probs.shape}")
        resampled = torch.multinomial(torch.squeeze(probs), self.sample_size, replacement=True)
        #logging.debug(f"Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        #logging.debug(f"Means of positions {self.states.tensor.mean(dim=0)} with shape {self.states.tensor.shape}")
        #logging.debug(f"Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        logging.debug(f"Quaternions {self.quaternions} with shape {self.quaternions.shape}")
        quats = self.quaternions[resampled]
        logging.debug(f"Resampled Quats {quats} with shape {quats.shape}")
        positions = self.states.tensor[:, resampled, :]
        self.states.tensor = positions
        rms = self.rms_errors[resampled]
        self.quaternions = quats
        self.rms_errors = torch.zeros_like(self.rms_errors)
        #logging.debug(f"ReSampled Means of quaternion {self.quaternions.mean(dim=0)} with shape {self.quaternions.shape}")
        #logging.debug(f"ReSampled Means of positions {self.states.tensor.mean(dim=0)} with shape {self.states.tensor.shape}")
        #logging.debug(f"ReSampled Means of rms errors {self.rms_errors.mean(dim=0)} with shape {self.rms_errors.shape}")
        logging.debug("Done Resampling")
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
