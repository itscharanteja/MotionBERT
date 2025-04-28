import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import ttest_ind
import datetime
import json


class CyclingBiomechanicsAnalyzer:

    def __init__(self, pose_data_3d, fps=30, output_dir=None):
        """
        Initialize the analyzer with 3D pose data from MotionBERT.
        Args:
            pose_data_3d: numpy array with shape (frames, joints, 3)
                         where joints follow the MotionBERT format
            fps: frames per second of the input video (default: 30)
            output_dir: directory to save outputs
        """
        self.pose_data = pose_data_3d
        self.n_frames = pose_data_3d.shape[0]
        self.fps = fps
        # Create unique output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"cycling_biomech_{timestamp}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Define joint indices (adjust based on your skeleton)
        self.joint_indices = {
            'pelvis': 0,
            'hip_r': 1,
            'knee_r': 2,
            'ankle_r': 3,
            'foot_r': 4,
            'hip_l': 6,
            'knee_l': 7,
            'ankle_l': 8,
            'foot_l': 9,
            'spine': 12
        }
        self.joint_angles = {}
        self.angular_velocities = {}
        self.range_of_motion = {}

    def _compute_vector(self, joint1, joint2, frame_idx):
        """Compute normalized vector from joint1 to joint2 at frame_idx."""
        vec = self.pose_data[frame_idx, self.joint_indices[joint2]] - \
            self.pose_data[frame_idx, self.joint_indices[joint1]]
        return vec / np.linalg.norm(vec)

    def _compute_angle(self, joint1, joint2, joint3, frame_idx):
        """Compute angle between three joints at frame_idx (in degrees)."""
        vec1 = self._compute_vector(joint2, joint1, frame_idx)
        vec2 = self._compute_vector(joint2, joint3, frame_idx)
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def calculate_joint_angles(self):
        """Calculate hip, knee, and ankle angles for both legs over all frames."""
        # Initialize storage
        hip_angle_r = np.zeros(self.n_frames)
        knee_angle_r = np.zeros(self.n_frames)
        ankle_angle_r = np.zeros(self.n_frames)
        hip_angle_l = np.zeros(self.n_frames)
        knee_angle_l = np.zeros(self.n_frames)
        ankle_angle_l = np.zeros(self.n_frames)

        for frame in range(self.n_frames):
            # Right leg
            hip_angle_r[frame] = self._compute_angle(
                'spine', 'hip_r', 'knee_r', frame)
            knee_angle_r[frame] = self._compute_angle(
                'hip_r', 'knee_r', 'ankle_r', frame)
            ankle_angle_r[frame] = self._compute_angle(
                'knee_r', 'ankle_r', 'foot_r', frame)

            # Left leg
            hip_angle_l[frame] = self._compute_angle(
                'spine', 'hip_l', 'knee_l', frame)
            knee_angle_l[frame] = self._compute_angle(
                'hip_l', 'knee_l', 'ankle_l', frame)
            ankle_angle_l[frame] = self._compute_angle(
                'knee_l', 'ankle_l', 'foot_l', frame)

        # Smooth angles
        window_length = min(11, self.n_frames // 2 *
                            2 + 1)  # Ensure odd window
        hip_angle_r = savgol_filter(
            hip_angle_r, window_length=window_length, polyorder=3)
        knee_angle_r = savgol_filter(
            knee_angle_r, window_length=window_length, polyorder=3)
        ankle_angle_r = savgol_filter(
            ankle_angle_r, window_length=window_length, polyorder=3)
        hip_angle_l = savgol_filter(
            hip_angle_l, window_length=window_length, polyorder=3)
        knee_angle_l = savgol_filter(
            knee_angle_l, window_length=window_length, polyorder=3)
        ankle_angle_l = savgol_filter(
            ankle_angle_l, window_length=window_length, polyorder=3)

        # Store results
        self.joint_angles = {
            'hip_right': hip_angle_r,
            'knee_right': knee_angle_r,
            'ankle_right': ankle_angle_r,
            'hip_left': hip_angle_l,
            'knee_left': knee_angle_l,
            'ankle_left': ankle_angle_l
        }

        # Save joint angles to CSV
        angles_df = pd.DataFrame(self.joint_angles)
        angles_df['frame'] = np.arange(self.n_frames)
        angles_df['time'] = angles_df['frame'] / self.fps
        angles_df.to_csv(os.path.join(
            self.output_dir, 'joint_angles.csv'), index=False)

        return self.joint_angles

    def calculate_angular_velocity(self):
        """Calculate angular velocity for all joints (degrees per second)."""
        if not self.joint_angles:
            self.calculate_joint_angles()

        angular_velocities = {}
        for joint, angles in self.joint_angles.items():
            angular_vel = np.diff(angles) * self.fps
            angular_vel = np.append(angular_vel, 0)  # Maintain array size
            window_length = min(11, self.n_frames // 2 * 2 + 1)
            angular_vel = savgol_filter(
                angular_vel, window_length=window_length, polyorder=3)
            angular_velocities[joint] = angular_vel
        self.angular_velocities = angular_velocities

        # Save angular velocities to CSV
        velocities_df = pd.DataFrame(self.angular_velocities)
        velocities_df['frame'] = np.arange(self.n_frames)
        velocities_df['time'] = velocities_df['frame'] / self.fps
        velocities_df.to_csv(os.path.join(
            self.output_dir, 'angular_velocities.csv'), index=False)

        return angular_velocities

    def calculate_range_of_motion(self):
        """Calculate range of motion (ROM) for each joint."""
        if not self.joint_angles:
            self.calculate_joint_angles()

        rom = {}
        for joint, angles in self.joint_angles.items():
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            range_angle = max_angle - min_angle
            rom[joint] = {'min': min_angle,
                          'max': max_angle, 'range': range_angle}
        self.range_of_motion = rom
        # Save ROM to CSV
        pd.DataFrame(rom).T.to_csv(os.path.join(
            self.output_dir, 'range_of_motion.csv'))
        return rom

    def analyze_pedaling_cycle(self):
        """Detect pedaling cycles and analyze consistency."""
        if not self.joint_angles:
            self.calculate_joint_angles()

        knee_angle = self.joint_angles['knee_right']
        peaks, _ = find_peaks(knee_angle, distance=int(self.fps / 4))
        cycle_durations = np.diff(peaks) / self.fps
        cadence = 60 / \
            np.mean(cycle_durations) if len(cycle_durations) > 0 else 0
        cycle_consistency = np.std(cycle_durations) if len(
            cycle_durations) > 1 else 0

        cycle_data = {
            'cycle_points': peaks.tolist(),
            'cycle_durations': cycle_durations.tolist(),
            'cadence_rpm': float(cadence),
            'cycle_consistency': float(cycle_consistency)
        }

        # Save cycle data to JSON
        with open(os.path.join(self.output_dir, 'pedaling_cycle_data.json'), 'w') as f:
            json.dump(cycle_data, f, indent=4)

        # Plot pedaling cycles
        fig, ax = plt.subplots(figsize=(10, 6))
        time = np.arange(self.n_frames) / self.fps
        ax.plot(time, knee_angle, label='Right Knee Angle')
        ax.plot(peaks / self.fps,
                knee_angle[peaks], 'ro', label='Detected Cycles')
        ax.set_title('Pedaling Cycle Detection')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Knee Angle (degrees)')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'pedaling_cycles.png'))
        plt.close()

        return cycle_data

    def evaluate_trajectory_smoothness(self):
        """Calculate smoothness of 3D joint trajectories (mean acceleration magnitude)."""
        smoothness = {}
        for joint_name, joint_idx in self.joint_indices.items():
            trajectories = self.pose_data[:, joint_idx, :]
            accel = np.diff(trajectories, n=2, axis=0)
            smoothness[joint_name] = np.mean(np.linalg.norm(accel, axis=1))
        # Save smoothness to CSV
        pd.DataFrame.from_dict(smoothness, orient='index', columns=['mean_accel']) \
            .to_csv(os.path.join(self.output_dir, 'trajectory_smoothness.csv'))
        return smoothness

    def compare_scenarios(self, analyzer2, scenario1_name="Video1", scenario2_name="Video2"):

        # Ensure all metrics are computed
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()
        analyzer2.calculate_joint_angles()
        analyzer2.calculate_angular_velocity()
        analyzer2.calculate_range_of_motion()

        comparison = {'ROM': {}}
        for joint in self.range_of_motion:
            rom1 = self.range_of_motion[joint]['range']
            rom2 = analyzer2.range_of_motion[joint]['range']
            comparison['ROM'][joint] = {
                scenario1_name: rom1,
                scenario2_name: rom2,
                'difference': rom2 - rom1
            }

        comparison['peak_angular_velocity'] = {}
        for joint in self.angular_velocities:
            peak1 = np.max(np.abs(self.angular_velocities[joint]))
            peak2 = np.max(np.abs(analyzer2.angular_velocities[joint]))
            comparison['peak_angular_velocity'][joint] = {
                scenario1_name: peak1,
                scenario2_name: peak2,
                'difference': peak2 - peak1
            }

        stats = {}
        for joint in self.joint_angles:
            angles1 = self.joint_angles[joint]
            angles2 = analyzer2.joint_angles[joint]
            t_stat, p_val = ttest_ind(angles1, angles2, equal_var=False)
            stats[joint] = {'t_stat': t_stat, 'p_value': p_val}

        # Plot knee angle comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        time1 = np.arange(self.n_frames) / self.fps
        time2 = np.arange(analyzer2.n_frames) / analyzer2.fps
        ax.plot(time1, self.joint_angles['knee_right'],
                label=f'{scenario1_name} Knee')
        ax.plot(
            time2, analyzer2.joint_angles['knee_right'], label=f'{scenario2_name} Knee')
        ax.set_title('Right Knee Angle Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'knee_angle_comparison.png'))
        plt.close()

        # Save comparison results
        comparison_data = {
            'ROM_comparison': comparison['ROM'],
            'peak_angular_velocity_comparison': comparison['peak_angular_velocity'],
            'statistics': {joint: {'t_stat': float(stats[joint]['t_stat']),
                                   'p_value': float(stats[joint]['p_value'])}
                           for joint in stats}
        }

        # Save as JSON
        with open(os.path.join(self.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comparison_data, f, indent=4, default=float)

        # Fix the DataFrame creation issue:
        # Create ROM comparison DataFrame
        rom_data = []
        for joint in comparison['ROM']:
            joint_data = comparison['ROM'][joint]
            rom_data.append({
                'joint': joint,
                scenario1_name: joint_data[scenario1_name],
                scenario2_name: joint_data[scenario2_name],
                'difference': joint_data['difference']
            })
        rom_comparison_df = pd.DataFrame(rom_data)
        rom_comparison_df.to_csv(os.path.join(
            self.output_dir, 'rom_comparison.csv'), index=False)

        # Create velocity comparison DataFrame
        vel_data = []
        for joint in comparison['peak_angular_velocity']:
            joint_data = comparison['peak_angular_velocity'][joint]
            vel_data.append({
                'joint': joint,
                scenario1_name: joint_data[scenario1_name],
                scenario2_name: joint_data[scenario2_name],
                'difference': joint_data['difference']
            })
        vel_comparison_df = pd.DataFrame(vel_data)
        vel_comparison_df.to_csv(os.path.join(
            self.output_dir, 'velocity_comparison.csv'), index=False)

        # Create statistics DataFrame
        stats_data = []
        for joint in stats:
            stats_data.append({
                'joint': joint,
                't_stat': stats[joint]['t_stat'],
                'p_value': stats[joint]['p_value']
            })
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.output_dir,
                        'statistical_tests.csv'), index=False)

        return {'comparison': comparison, 'statistics': stats}
        """
        Compare kinematic metrics between two scenarios.
        """
        # Ensure all metrics are computed
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()
        analyzer2.calculate_joint_angles()
        analyzer2.calculate_angular_velocity()
        analyzer2.calculate_range_of_motion()

        comparison = {'ROM': {}}
        for joint in self.range_of_motion:
            rom1 = self.range_of_motion[joint]['range']
            rom2 = analyzer2.range_of_motion[joint]['range']
            comparison['ROM'][joint] = {
                scenario1_name: rom1,
                scenario2_name: rom2,
                'difference': rom2 - rom1
            }

        comparison['peak_angular_velocity'] = {}
        for joint in self.angular_velocities:
            peak1 = np.max(np.abs(self.angular_velocities[joint]))
            peak2 = np.max(np.abs(analyzer2.angular_velocities[joint]))
            comparison['peak_angular_velocity'][joint] = {
                scenario1_name: peak1,
                scenario2_name: peak2,
                'difference': peak2 - peak1
            }

        stats = {}
        for joint in self.joint_angles:
            angles1 = self.joint_angles[joint]
            angles2 = analyzer2.joint_angles[joint]
            t_stat, p_val = ttest_ind(angles1, angles2, equal_var=False)
            stats[joint] = {'t_stat': t_stat, 'p_value': p_val}

        # Plot knee angle comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        time1 = np.arange(self.n_frames) / self.fps
        time2 = np.arange(analyzer2.n_frames) / analyzer2.fps
        ax.plot(time1, self.joint_angles['knee_right'],
                label=f'{scenario1_name} Knee')
        ax.plot(
            time2, analyzer2.joint_angles['knee_right'], label=f'{scenario2_name} Knee')
        ax.set_title('Right Knee Angle Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'knee_angle_comparison.png'))
        plt.close()

        # Save comparison results
        comparison_data = {
            'ROM_comparison': comparison['ROM'],
            'peak_angular_velocity_comparison': comparison['peak_angular_velocity'],
            'statistics': {joint: {'t_stat': float(stats[joint]['t_stat']),
                                   'p_value': float(stats[joint]['p_value'])}
                           for joint in stats}
        }

        # Save as JSON
        with open(os.path.join(self.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comparison_data, f, indent=4, default=float)

        # Save as CSV for easier viewing
        rom_comparison_df = pd.DataFrame({(i, j): comparison['ROM'][i][j]
                                          for i in comparison['ROM']
                                          for j in comparison['ROM'][i]})
        rom_comparison_df.to_csv(os.path.join(
            self.output_dir, 'rom_comparison.csv'))

        vel_comparison_df = pd.DataFrame({(i, j): comparison['peak_angular_velocity'][i][j]
                                          for i in comparison['peak_angular_velocity']
                                          for j in comparison['peak_angular_velocity'][i]})
        vel_comparison_df.to_csv(os.path.join(
            self.output_dir, 'velocity_comparison.csv'))

        stats_df = pd.DataFrame({joint: {'t_stat': stats[joint]['t_stat'], 'p_value': stats[joint]['p_value']}
                                 for joint in stats}).T
        stats_df.to_csv(os.path.join(self.output_dir, 'statistical_tests.csv'))

        return {'comparison': comparison, 'statistics': stats}

    def generate_summary_report(self):
        """Generate a comprehensive summary of the biomechanical analysis."""
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()

        cycle_data = self.analyze_pedaling_cycle()
        smoothness = self.evaluate_trajectory_smoothness()

        summary = {
            'duration_seconds': self.n_frames / self.fps,
            'cadence_rpm': cycle_data['cadence_rpm'],
            'cycle_consistency': cycle_data['cycle_consistency'],
            'range_of_motion': self.range_of_motion,
            'peak_angular_velocity': {
                joint: np.max(np.abs(vel)) for joint, vel in self.angular_velocities.items()
            },
            'trajectory_smoothness': smoothness
        }

        # Save summary as CSV
        summary_dict = {
            'duration_seconds': summary['duration_seconds'],
            'cadence_rpm': summary['cadence_rpm'],
            'cycle_consistency': summary['cycle_consistency'],
        }

        for joint in summary['range_of_motion']:
            summary_dict[f'rom_{joint}_min'] = summary['range_of_motion'][joint]['min']
            summary_dict[f'rom_{joint}_max'] = summary['range_of_motion'][joint]['max']
            summary_dict[f'rom_{joint}_range'] = summary['range_of_motion'][joint]['range']

        for joint in summary['peak_angular_velocity']:
            summary_dict[f'peak_velocity_{joint}'] = summary['peak_angular_velocity'][joint]

        for joint in summary['trajectory_smoothness']:
            summary_dict[f'smoothness_{joint}'] = summary['trajectory_smoothness'][joint]

        df = pd.DataFrame([summary_dict])
        df.to_csv(os.path.join(self.output_dir,
                  'summary_biomechanics.csv'), index=False)

        # Also save as JSON for more structured data
        with open(os.path.join(self.output_dir, 'summary_biomechanics.json'), 'w') as f:
            json.dump(summary, f, indent=4, default=float)

        return summary

    def plot_joint_angles(self):
        """Plot joint angles over time and save the plot."""
        if not self.joint_angles:
            self.calculate_joint_angles()

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        time = np.arange(self.n_frames) / self.fps

        # Right leg
        axes[0, 0].plot(time, self.joint_angles['hip_right'], label='Hip')
        axes[0, 0].set_title('Right Hip Angle')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Angle (degrees)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[1, 0].plot(time, self.joint_angles['knee_right'], label='Knee')
        axes[1, 0].set_title('Right Knee Angle')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[2, 0].plot(time, self.joint_angles['ankle_right'], label='Ankle')
        axes[2, 0].set_title('Right Ankle Angle')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Angle (degrees)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Left leg
        axes[0, 1].plot(time, self.joint_angles['hip_left'], label='Hip')
        axes[0, 1].set_title('Left Hip Angle')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 1].plot(time, self.joint_angles['knee_left'], label='Knee')
        axes[1, 1].set_title('Left Knee Angle')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[2, 1].plot(time, self.joint_angles['ankle_left'], label='Ankle')
        axes[2, 1].set_title('Left Ankle Angle')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Angle (degrees)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'joint_angles_plot.png'))
        plt.close()

    def plot_angular_velocities(self):
        """Plot angular velocities over time and save the plot."""
        if not self.angular_velocities:
            self.calculate_angular_velocity()

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        time = np.arange(self.n_frames) / self.fps

        # Right leg
        axes[0, 0].plot(
            time, self.angular_velocities['hip_right'], label='Hip')
        axes[0, 0].set_title('Right Hip Angular Velocity')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[1, 0].plot(
            time, self.angular_velocities['knee_right'], label='Knee')
        axes[1, 0].set_title('Right Knee Angular Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[2, 0].plot(
            time, self.angular_velocities['ankle_right'], label='Ankle')
        axes[2, 0].set_title('Right Ankle Angular Velocity')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Left leg
        axes[0, 1].plot(time, self.angular_velocities['hip_left'], label='Hip')
        axes[0, 1].set_title('Left Hip Angular Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 1].plot(
            time, self.angular_velocities['knee_left'], label='Knee')
        axes[1, 1].set_title('Left Knee Angular Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[2, 1].plot(
            time, self.angular_velocities['ankle_left'], label='Ankle')
        axes[2, 1].set_title('Left Ankle Angular Velocity')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    'angular_velocities_plot.png'))
        plt.close()

    def save_data_to_csv(self, filepath):
        """Save all calculated metrics to CSV for further analysis."""
        if not self.joint_angles:
            self.calculate_joint_angles()
        if not self.angular_velocities:
            self.calculate_angular_velocity()
        if not self.range_of_motion:
            self.calculate_range_of_motion()

        data = {'time': np.arange(self.n_frames) / self.fps}
        for joint, angles in self.joint_angles.items():
            data[f'{joint}_angle'] = angles
        for joint, velocities in self.angular_velocities.items():
            data[f'{joint}_velocity'] = velocities
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        rom_df = pd.DataFrame(self.range_of_motion).T
        rom_df.to_csv(filepath.replace('.csv', '_rom.csv'))


# --- main function ---
def main():
    # Generate a unique output folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"biomech_outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Replace with your actual data files or generate dummy data
    try:
        pose_data1 = np.load(
            input("Enter the .npy file of 1st video"))  # Video 1
        pose_data2 = np.load(
            input("Enter the .npy file of 2nd video"))  # Video 2
    except FileNotFoundError:
        print("Pose data files not found. Generating dummy data.")
        n_frames = 300
        n_joints = 17
        pose_data1 = np.random.rand(n_frames, n_joints, 3)
        pose_data2 = np.random.rand(n_frames, n_joints, 3)

    # Analyze video 1
    output_dir_video1 = os.path.join(output_dir, "video1")
    os.makedirs(output_dir_video1, exist_ok=True)
    analyzer1 = CyclingBiomechanicsAnalyzer(
        pose_data1, fps=30, output_dir=output_dir_video1)
    analyzer1.calculate_joint_angles()
    analyzer1.calculate_angular_velocity()
    analyzer1.calculate_range_of_motion()
    analyzer1.plot_joint_angles()
    analyzer1.plot_angular_velocities()
    summary1 = analyzer1.generate_summary_report()
    analyzer1.save_data_to_csv(os.path.join(
        output_dir, "video1_biomech_data.csv"))
    print("Video 1 Summary:")
    for key, value in summary1.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")

    # Analyze video 2
    output_dir_video2 = os.path.join(output_dir, "video2")
    os.makedirs(output_dir_video2, exist_ok=True)
    analyzer2 = CyclingBiomechanicsAnalyzer(
        pose_data2, fps=30, output_dir=output_dir_video2)
    analyzer2.calculate_joint_angles()
    analyzer2.calculate_angular_velocity()
    analyzer2.calculate_range_of_motion()
    analyzer2.plot_joint_angles()
    analyzer2.plot_angular_velocities()
    summary2 = analyzer2.generate_summary_report()
    analyzer2.save_data_to_csv(os.path.join(
        output_dir, "video2_biomech_data.csv"))
    print("\nVideo 2 Summary:")
    for key, value in summary2.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")

    # Compare scenarios
    comparison_output_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_output_dir, exist_ok=True)

    # Use the comparison output dir for saving comparison results
    orig_output_dir = analyzer1.output_dir
    analyzer1.output_dir = comparison_output_dir
    comparison_results = analyzer1.compare_scenarios(
        analyzer2, "Video1", "Video2")
    analyzer1.output_dir = orig_output_dir

    # Save comprehensive report
    with open(os.path.join(output_dir, "comprehensive_report.json"), 'w') as f:
        report = {
            "video1": {
                "summary": summary1,
                "file_paths": {
                    "joint_angles": os.path.join(output_dir_video1, "joint_angles.csv"),
                    "range_of_motion": os.path.join(output_dir_video1, "range_of_motion.csv"),
                    "angular_velocities": os.path.join(output_dir_video1, "angular_velocities.csv"),
                    "trajectory_smoothness": os.path.join(output_dir_video1, "trajectory_smoothness.csv"),
                    "pedaling_cycle": os.path.join(output_dir_video1, "pedaling_cycle_data.json"),
                    "summary": os.path.join(output_dir_video1, "summary_biomechanics.csv"),
                    "plots": {
                        "joint_angles": os.path.join(output_dir_video1, "joint_angles_plot.png"),
                        "angular_velocities": os.path.join(output_dir_video1, "angular_velocities_plot.png"),
                        "pedaling_cycles": os.path.join(output_dir_video1, "pedaling_cycles.png")
                    }
                }
            },
            "video2": {
                "summary": summary2,
                "file_paths": {
                    "joint_angles": os.path.join(output_dir_video2, "joint_angles.csv"),
                    "range_of_motion": os.path.join(output_dir_video2, "range_of_motion.csv"),
                    "angular_velocities": os.path.join(output_dir_video2, "angular_velocities.csv"),
                    "trajectory_smoothness": os.path.join(output_dir_video2, "trajectory_smoothness.csv"),
                    "pedaling_cycle": os.path.join(output_dir_video2, "pedaling_cycle_data.json"),
                    "summary": os.path.join(output_dir_video2, "summary_biomechanics.csv"),
                    "plots": {
                        "joint_angles": os.path.join(output_dir_video2, "joint_angles_plot.png"),
                        "angular_velocities": os.path.join(output_dir_video2, "angular_velocities_plot.png"),
                        "pedaling_cycles": os.path.join(output_dir_video2, "pedaling_cycles.png")
                    }
                }
            },
            "comparison": {
                "rom_comparison": os.path.join(comparison_output_dir, "rom_comparison.csv"),
                "velocity_comparison": os.path.join(comparison_output_dir, "velocity_comparison.csv"),
                "statistical_tests": os.path.join(comparison_output_dir, "statistical_tests.csv"),
                "knee_angle_comparison": os.path.join(comparison_output_dir, "knee_angle_comparison.png"),
                "comparison_results": os.path.join(comparison_output_dir, "comparison_results.json")
            }
        }

        def numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(report, f, indent=4, default=numpy_to_python)

    print("\nAnalysis complete. All results saved to:", output_dir)
    print("Comprehensive report saved to:", os.path.join(
        output_dir, "comprehensive_report.json"))


if __name__ == "__main__":
    main()
