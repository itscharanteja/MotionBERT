import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

class CyclingBiomechanicsAnalyzer:
    def __init__(self, pose_data_3d):
        """
        Initialize the analyzer with 3D pose data from MotionBERT.
        
        Args:
            pose_data_3d: numpy array with shape (frames, joints, 3)
                         where joints follow the MotionBERT format
        """
        self.pose_data = pose_data_3d
        self.n_frames = pose_data_3d.shape[0]
        self.fps = 30  # Assuming 30 fps, adjust based on your video
        
        # Define joint indices according to MotionBERT format
        # These may need to be adjusted based on your exact implementation
        self.joint_indices = {
            'hip_r': 1,
            'knee_r': 2,
            'ankle_r': 3,
            'hip_l': 4,
            'knee_l': 5,
            'ankle_l': 6,
            'pelvis': 0,
            'spine': 7
        }
        
        # Results storage
        self.joint_angles = {}
        self.angular_velocities = {}
        self.range_of_motion = {}
    
    def _compute_vector(self, joint1, joint2, frame_idx):
        """Compute vector from joint1 to joint2 at frame_idx"""
        vec = self.pose_data[frame_idx, self.joint_indices[joint2]] - self.pose_data[frame_idx, self.joint_indices[joint1]]
        return vec / np.linalg.norm(vec)  # Normalize to unit vector
    
    def _compute_angle(self, joint1, joint2, joint3, frame_idx):
        """Compute angle between three joints at frame_idx (in degrees)"""
        vec1 = self._compute_vector(joint2, joint1, frame_idx)
        vec2 = self._compute_vector(joint2, joint3, frame_idx)
        
        # Compute angle using dot product
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)
    
    def calculate_joint_angles(self):
        """Calculate hip, knee, and ankle angles for both legs over all frames"""
        # Initialize storage for joint angles
        hip_angle_r = np.zeros(self.n_frames)
        knee_angle_r = np.zeros(self.n_frames)
        hip_angle_l = np.zeros(self.n_frames)
        knee_angle_l = np.zeros(self.n_frames)
        
        for frame in range(self.n_frames):
            # Right leg
            hip_angle_r[frame] = self._compute_angle('spine', 'hip_r', 'knee_r', frame)
            knee_angle_r[frame] = self._compute_angle('hip_r', 'knee_r', 'ankle_r', frame)
            
            # Left leg
            hip_angle_l[frame] = self._compute_angle('spine', 'hip_l', 'knee_l', frame)
            knee_angle_l[frame] = self._compute_angle('hip_l', 'knee_l', 'ankle_l', frame)
        
        # Apply Savitzky-Golay filter to smooth the data
        hip_angle_r = savgol_filter(hip_angle_r, window_length=11, polyorder=3)
        knee_angle_r = savgol_filter(knee_angle_r, window_length=11, polyorder=3)
        hip_angle_l = savgol_filter(hip_angle_l, window_length=11, polyorder=3)
        knee_angle_l = savgol_filter(knee_angle_l, window_length=11, polyorder=3)
        
        # Store results
        self.joint_angles = {
            'hip_right': hip_angle_r,
            'knee_right': knee_angle_r,
            'hip_left': hip_angle_l,
            'knee_left': knee_angle_l
        }
        
        return self.joint_angles
    
    def calculate_angular_velocity(self):
        """Calculate angular velocity for all joints"""
        if not self.joint_angles:
            self.calculate_joint_angles()
        
        # Initialize angular velocity storage
        angular_velocities = {}
        
        # Calculate angular velocities (degrees per second)
        for joint, angles in self.joint_angles.items():
            # Compute difference and convert to deg/s based on fps
            angular_vel = np.diff(angles) * self.fps
            # Add a zero at the end to maintain array size
            angular_vel = np.append(angular_vel, 0)
            # Apply smoothing
            angular_vel = savgol_filter(angular_vel, window_length=11, polyorder=3)
            angular_velocities[joint] = angular_vel
        
        self.angular_velocities = angular_velocities
        return angular_velocities
    
    def calculate_range_of_motion(self):
        """Calculate range of motion (ROM) for each joint"""
        if not self.joint_angles:
            self.calculate_joint_angles()
        
        # Initialize ROM storage
        rom = {}
        
        # Calculate min, max, and range for each joint
        for joint, angles in self.joint_angles.items():
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            range_angle = max_angle - min_angle
            
            rom[joint] = {
                'min': min_angle,
                'max': max_angle,
                'range': range_angle
            }
        
        self.range_of_motion = rom
        return rom
    
    def analyze_pedaling_cycle(self):
        """Detect pedaling cycles and analyze consistency"""
        # This implementation detects cycles based on knee angle pattern
        # A more sophisticated approach might use additional signals or frequency analysis
        
        # Get knee angle from the calculated joint angles
        knee_angle = self.joint_angles['knee_right']
        
        # Find peaks (knee extension)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(knee_angle, distance=15)  # Minimum distance between peaks
        
        # Assuming each peak-to-peak represents one pedaling cycle
        cycle_durations = np.diff(peaks) / self.fps  # in seconds
        cadence = 60 / np.mean(cycle_durations)  # in RPM
        
        return {
            'cycle_points': peaks,
            'cycle_durations': cycle_durations,
            'cadence_rpm': cadence,
            'cycle_consistency': np.std(cycle_durations)  # Lower std = more consistent
        }
    
    def generate_summary_report(self):
        """Generate a comprehensive summary of the biomechanical analysis"""
        # Calculate all metrics if not already done
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()
        
        cycle_data = self.analyze_pedaling_cycle()
        
        # Create summary dictionary
        summary = {
            'duration_seconds': self.n_frames / self.fps,
            'cadence_rpm': cycle_data['cadence_rpm'],
            'cycle_consistency': cycle_data['cycle_consistency'],
            'range_of_motion': self.range_of_motion,
            'peak_angular_velocity': {
                joint: np.max(np.abs(vel)) for joint, vel in self.angular_velocities.items()
            }
        }
        
        return summary
    
    def plot_joint_angles(self, save_path=None):
        """Plot joint angles over time"""
        if not self.joint_angles:
            self.calculate_joint_angles()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        time = np.arange(self.n_frames) / self.fps
        
        # Plot right leg
        axes[0].plot(time, self.joint_angles['hip_right'], label='Hip')
        axes[0].plot(time, self.joint_angles['knee_right'], label='Knee')
        axes[0].set_title('Right Leg Joint Angles')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot left leg
        axes[1].plot(time, self.joint_angles['hip_left'], label='Hip')
        axes[1].plot(time, self.joint_angles['knee_left'], label='Knee')
        axes[1].set_title('Left Leg Joint Angles')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Angle (degrees)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def save_data_to_csv(self, filepath):
        """Save all calculated metrics to CSV for further analysis"""
        # Ensure all calculations are done
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()
        
        # Create a DataFrame
        data = {
            'time': np.arange(self.n_frames) / self.fps
        }
        
        # Add joint angles
        for joint, angles in self.joint_angles.items():
            data[f'{joint}_angle'] = angles
        
        # Add angular velocities
        for joint, velocities in self.angular_velocities.items():
            data[f'{joint}_velocity'] = velocities
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        # Save range of motion as a separate file
        rom_df = pd.DataFrame(self.range_of_motion).T
        rom_df.to_csv(filepath.replace('.csv', '_rom.csv'))


# Example usage
def main():
    # Load 3D pose data (example with random data)
    # In practice, this would be the output from MotionBERT
    # Shape: (frames, joints, 3D coordinates)
    
    # For demonstration, create random pose data
    # Adjust joint count based on your MotionBERT implementation
    n_frames = 300
    n_joints = 17  # Common for most pose estimation models
    
    # For a real implementation, replace this with your actual data loading
    pose_data = np.load('X3D.npy')
    # pose_data = np.random.rand(n_frames, n_joints, 3)
    
    # Create analyzer and run calculations
    analyzer = CyclingBiomechanicsAnalyzer(pose_data)
    joint_angles = analyzer.calculate_joint_angles()
    angular_velocities = analyzer.calculate_angular_velocity()
    range_of_motion = analyzer.calculate_range_of_motion()
    
    # Generate and print summary report
    summary = analyzer.generate_summary_report()
    print("Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Plot joint angles
    analyzer.plot_joint_angles(save_path="joint_angles.png")
    
    # Save data to CSV for further analysis
    analyzer.save_data_to_csv("cycling_biomechanics_data.csv")


if __name__ == "__main__":
    main()