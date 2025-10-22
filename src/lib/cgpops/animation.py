import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import sys
from matplotlib.widgets import Button, Slider
import re
import os


class SpacecraftAnimator:
    def __init__(self, matlab_file_sufix='BC', Ix=1.0, Iy=1.0, Iz=1.0, use_realtime=True):
        # Initialize data containers
        self.euler_angles = [[], [], []]  # φ, θ, ψ
        self.quaternions = [[], [], [], []]  # w, x, y, z
        self.angular_rates = [[], [], []]  # ωx, ωy, ωz
        self.control_inputs = [[], [], []]  # ux, uy, uz
        self.time_data = []
        self.total_time = 0
        self.use_realtime = use_realtime
        
        # Spacecraft properties
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        
        # Animation properties
        self.current_step = 0
        self.is_playing = False
        self.animation = None
        self.is_paused = False
        self.pause_button = None
        self.frame_slider = None
        self.manual_control = False

        # Load data
        matlab_file = f'../output/cgpopsIPOPTSolution{matlab_file_sufix}.m'
        self.load_data(matlab_file)
        
        # Calculate frame intervals from time data
        self.calculate_frame_intervals()
        
        # Create figure and setup
        self.setup_plot()

    def load_data(self, matlab_file):
        """Load trajectory data from MATLAB file"""
        try:
            self.parse_matlab_file(matlab_file)
            print(f"✓ Loaded trajectory data: {len(self.time_data)} points")
        except FileNotFoundError:
            print(f"⚠ MATLAB file '{matlab_file}' not found.")
            raise
        except Exception as e:
            print(f"⚠ Error loading MATLAB file: {e}.")
            raise

    def parse_matlab_file(self, filename):
        """Parse the MATLAB .m file directly"""

        # Extract suffix from filename
        name_without_ext = os.path.splitext(filename)[0]  # Remove .m
        suffix = name_without_ext[-2:]  # Get last 2 characters
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract total time
        tf_match = re.search(rf'system{suffix}\.phase\(1\)\.tf\s*=\s*([\d.e+-]+);', content)
        self.total_time = float(tf_match.group(1)) if tf_match else 0.0
        
        # Extract states x(1-7) with points
        states = []
        for i in range(1, 8):
            pattern = rf'system{suffix}\.phase\(1\)\.x\({i}\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);'
            matches = re.findall(pattern, content)
            values = [float(val) for idx, val in sorted(matches, key=lambda x: int(x[0]))]
            states.append(values)
        
        # Extract controls u(1-3) with points
        controls = []
        for i in range(1, 4):
            pattern = rf'system{suffix}\.phase\(1\)\.u\({i}\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);'
            matches = re.findall(pattern, content)
            values = [float(val) for idx, val in sorted(matches, key=lambda x: int(x[0]))]
            controls.append(values)
        
        # Extract time points - THIS IS THE KEY PART
        pattern = rf'system{suffix}\.phase\(1\)\.t\.point\((\d+)\)\s*=\s*([-\d.e+-]+);'
        matches = re.findall(pattern, content)
        self.time_data = [float(val) for idx, val in sorted(matches, key=lambda x: int(x[0]))]
        
        # States from .m file: x(1-4) = quaternions [w,x,y,z], x(5-7) = rates [ωx,ωy,ωz]
        quat_w, quat_x, quat_y, quat_z = states[0], states[1], states[2], states[3]
        omega_x, omega_y, omega_z = states[4], states[5], states[6]
        
        # Convert quaternions to Euler angles (degrees)
        n_points = len(quat_w)
        euler_angles = np.zeros((3, n_points))
        
        for i in range(n_points):
            quat = [quat_x[i], quat_y[i], quat_z[i], quat_w[i]]  # scipy order: [x,y,z,w]
            rot = Rotation.from_quat(quat)
            euler = rot.as_euler('XYZ', degrees=True)  # [roll, pitch, yaw]
            euler_angles[:, i] = euler
        
        # Assign to class variables
        self.euler_angles[0] = euler_angles[0, :].tolist()  # roll (φ)
        self.euler_angles[1] = euler_angles[1, :].tolist()  # pitch (θ)
        self.euler_angles[2] = euler_angles[2, :].tolist()  # yaw (ψ)
        
        self.quaternions[0] = quat_w    # w
        self.quaternions[1] = quat_x    # x
        self.quaternions[2] = quat_y    # y
        self.quaternions[3] = quat_z    # z
        
        self.angular_rates[0] = omega_x  # ωx (rad/s)
        self.angular_rates[1] = omega_y  # ωy (rad/s)
        self.angular_rates[2] = omega_z  # ωz (rad/s)
        
        self.control_inputs[0] = controls[0]  # ux
        self.control_inputs[1] = controls[1]  # uy
        self.control_inputs[2] = controls[2]  # uz

    def calculate_frame_intervals(self):
        """Calculate time intervals between frames for real-time playback"""
        if len(self.time_data) < 2:
            self.frame_intervals = [100]  # Default 100ms
            return
        
        # Calculate dt between consecutive frames
        self.frame_intervals = []
        for i in range(len(self.time_data) - 1):
            dt = (self.time_data[i+1] - self.time_data[i]) * 1000  # Convert to ms
            self.frame_intervals.append(max(dt, 1))  # Minimum 1ms
        
        # Add last interval (same as previous)
        self.frame_intervals.append(self.frame_intervals[-1] if self.frame_intervals else 100)
        
        avg_interval = np.mean(self.frame_intervals)
        min_interval = np.min(self.frame_intervals)
        max_interval = np.max(self.frame_intervals)
        
        print(f"✓ Frame intervals calculated:")
        print(f"  - Average: {avg_interval:.2f}ms")
        print(f"  - Min: {min_interval:.2f}ms")
        print(f"  - Max: {max_interval:.2f}ms")
        print(f"  - Total frames: {len(self.time_data)}")
        print(f"  - Total real time: {sum(self.frame_intervals)/1000:.2f}s")

    def create_spacecraft_geometry(self):
        """Create spacecraft geometry based on inertia moments"""
        # Scale factors based on inertia ratios
        max_I = max(self.Ix, self.Iy, self.Iz)
        scale_factor = 2.0
        
        # Dimensions proportional to inertia
        dim_x = scale_factor * np.sqrt(self.Ix / max_I)
        dim_y = scale_factor * np.sqrt(self.Iy / max_I)
        dim_z = scale_factor * np.sqrt(self.Iz / max_I)
        
        # Create box vertices
        vertices = np.array([
            [-dim_x/2, -dim_y/2, -dim_z/2],
            [+dim_x/2, -dim_y/2, -dim_z/2],
            [+dim_x/2, +dim_y/2, -dim_z/2],
            [-dim_x/2, +dim_y/2, -dim_z/2],
            [-dim_x/2, -dim_y/2, +dim_z/2],
            [+dim_x/2, -dim_y/2, +dim_z/2],
            [+dim_x/2, +dim_y/2, +dim_z/2],
            [-dim_x/2, +dim_y/2, +dim_z/2]
        ])
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
        ]
        
        return faces, vertices, [dim_x, dim_y, dim_z]    
    
    def create_orientation_arrows(self, dimensions):
        """Create orientation arrows for X, Y, Z axes"""
        dim_x, dim_y, dim_z = dimensions
        arrow_length = max(dimensions) * 2.0
        
        # Arrow origins (from spacecraft center to surface)
        origins = np.array([
            [-dim_x/2, -dim_y/2, -dim_z/2],      # X-axis (red)
            [-dim_x/2, -dim_y/2, -dim_z/2],      # Y-axis (green)
            [-dim_x/2, -dim_y/2, -dim_z/2]       # Z-axis (blue)
        ])
        
        # Arrow directions
        directions = np.array([
            [arrow_length, 0, 0],
            [0, arrow_length, 0],
            [0, 0, arrow_length]
        ])
        
        return origins, directions
    
    def setup_plot(self):
        """Setup the matplotlib 3D plot"""
        self.fig = plt.figure(figsize=(18, 12))
        
        # Create subplots with 2x2 grid for the data plots
        gs = self.fig.add_gridspec(2, 4, height_ratios=[3, 1], width_ratios=[2, 2, 2, 2])
        
        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(gs[0, :], projection='3d')
        
        # Data plots
        self.ax_euler = self.fig.add_subplot(gs[1, 0])
        self.ax_quat = self.fig.add_subplot(gs[1, 1])
        self.ax_rates = self.fig.add_subplot(gs[1, 2])
        self.ax_control = self.fig.add_subplot(gs[1, 3])
        
        # Setup 3D plot
        self.ax_3d.set_xlim([-4, 4])
        self.ax_3d.set_ylim([-4, 4])
        self.ax_3d.set_zlim([-4, 4])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('Spacecraft Attitude Animation')
        
        # Create spacecraft geometry
        self.faces, self.vertices, self.dimensions = self.create_spacecraft_geometry()
        
        # Initialize 3D objects
        self.spacecraft_plot = Poly3DCollection(self.faces, alpha=0.7, facecolor='gray', edgecolor='black')
        self.ax_3d.add_collection3d(self.spacecraft_plot)
        
        # Orientation arrows
        self.arrow_plots = []
        colors = ['red', 'green', 'blue']
        for i in range(3):
            arrow = self.ax_3d.quiver(0, 0, 0, 1, 0, 0, color=colors[i], arrow_length_ratio=0.1)
            self.arrow_plots.append(arrow)
                
        # Reference frame
        self.ax_3d.quiver(0, 0, 0, 2, 0, 0, color='red', alpha=0.3, arrow_length_ratio=0.05)
        self.ax_3d.quiver(0, 0, 0, 0, 2, 0, color='green', alpha=0.3, arrow_length_ratio=0.05)
        self.ax_3d.quiver(0, 0, 0, 0, 0, 2, color='blue', alpha=0.3, arrow_length_ratio=0.05)
        
        # Setup data plots
        self.setup_data_plots()
        
        # Add text displays
        self.text_info = self.ax_3d.text2D(-0.50, 0.98, '', transform=self.ax_3d.transAxes, 
                                        verticalalignment='top', fontfamily='monospace', fontsize=12)
        
        self.setup_pause_button()
        self.setup_frame_slider()
    
    def setup_data_plots(self):
        """Setup the data visualization plots"""
        
        # Euler angles plot
        self.ax_euler.plot(self.time_data, self.euler_angles[0], 'r-', label='Roll (φ)')
        self.ax_euler.plot(self.time_data, self.euler_angles[1], 'g-', label='Pitch (θ)')
        self.ax_euler.plot(self.time_data, self.euler_angles[2], 'b-', label='Yaw (ψ)')
        self.euler_marker, = self.ax_euler.plot([], [], 'ko', markersize=8)
        self.ax_euler.set_xlabel('Time (s)')
        self.ax_euler.set_ylabel('Angle (deg)')
        self.ax_euler.set_title('Euler Angles')
        self.ax_euler.legend(fontsize=12)
        self.ax_euler.grid(True)
        
        # Quaternion plot
        self.ax_quat.plot(self.time_data, self.quaternions[0], 'k-', label='w')
        self.ax_quat.plot(self.time_data, self.quaternions[1], 'r-', label='x')
        self.ax_quat.plot(self.time_data, self.quaternions[2], 'g-', label='y')
        self.ax_quat.plot(self.time_data, self.quaternions[3], 'b-', label='z')
        self.quat_marker, = self.ax_quat.plot([], [], 'ko', markersize=8)
        self.ax_quat.set_xlabel('Time (s)')
        self.ax_quat.set_ylabel('Quaternion')
        self.ax_quat.set_title('Quaternions')
        self.ax_quat.legend(fontsize=12)
        self.ax_quat.grid(True)
        
        # Angular rates plot
        self.ax_rates.plot(self.time_data, np.rad2deg(self.angular_rates[0]), 'r-', label='ωx')
        self.ax_rates.plot(self.time_data, np.rad2deg(self.angular_rates[1]), 'g-', label='ωy')
        self.ax_rates.plot(self.time_data, np.rad2deg(self.angular_rates[2]), 'b-', label='ωz')
        self.rates_marker, = self.ax_rates.plot([], [], 'ko', markersize=8)
        self.ax_rates.set_xlabel('Time (s)')
        self.ax_rates.set_ylabel('Rate (deg/s)')
        self.ax_rates.set_title('Angular Rates')
        self.ax_rates.legend(fontsize=12)
        self.ax_rates.grid(True)
        
        # Control inputs plot
        self.ax_control.plot(self.time_data, self.control_inputs[0], 'r-', label='ux')
        self.ax_control.plot(self.time_data, self.control_inputs[1], 'g-', label='uy')
        self.ax_control.plot(self.time_data, self.control_inputs[2], 'b-', label='uz')
        self.control_marker, = self.ax_control.plot([], [], 'ko', markersize=8)
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Control')
        self.ax_control.set_title('Control Inputs')
        self.ax_control.legend(fontsize=12)
        self.ax_control.grid(True)
    
    def setup_pause_button(self):
        """Setup pause/resume button"""
        button_ax = plt.axes([0.1, 0.4, 0.1, 0.04])
        self.pause_button = Button(button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)

    def toggle_pause(self, event):
        """Toggle animation pause/resume"""
        if self.is_paused:
            self.animation.resume()
            self.pause_button.label.set_text('Pause')
            self.is_paused = False
        else:
            self.animation.pause()
            self.pause_button.label.set_text('Resume')
            self.is_paused = True

    def setup_frame_slider(self):
        """Setup frame control slider"""
        slider_ax = plt.axes([0.05, 0.35, 0.2, 0.03])
        self.frame_slider = Slider(
            slider_ax, 'Step', 0, len(self.time_data)-1, 
            valinit=0, valfmt='%d', valstep=1
        )
        self.frame_slider.on_changed(self.update_from_slider)

    def update_from_slider(self, val):
        """Update animation frame from slider"""
        frame = int(self.frame_slider.val)
        self.current_step = frame
        self.manual_control = True
        self.update_spacecraft(frame)
        self.update_data_markers(frame)
        self.update_info_text(frame)
        self.fig.canvas.draw_idle()

    def update_spacecraft(self, step):
        """Update spacecraft orientation and position"""
        if step >= len(self.time_data):
            return
        
        # Get quaternion at current step
        quat = [
            self.quaternions[1][step],  # x
            self.quaternions[2][step],  # y
            self.quaternions[3][step],  # z
            self.quaternions[0][step]   # w
        ]
        
        # Create rotation matrix
        rotation = Rotation.from_quat(quat)
        rotation_matrix = rotation.as_matrix()
        
        # Rotate spacecraft vertices
        rotated_faces = []
        for face in self.faces:
            rotated_face = []
            for vertex in face:
                rotated_vertex = rotation_matrix @ vertex
                rotated_face.append(rotated_vertex)
            rotated_faces.append(rotated_face)
        
        # Update spacecraft plot
        self.spacecraft_plot.set_verts(rotated_faces)
        
        # Update orientation arrows
        origins, directions = self.create_orientation_arrows(self.dimensions)
        colors = ['red', 'green', 'blue']
        
        # Clear existing arrows
        for arrow in self.arrow_plots:
            arrow.remove()
        self.arrow_plots.clear()
        
        # Add new arrows
        for i, (origin, direction) in enumerate(zip(origins, directions)):
            rotated_origin = rotation_matrix @ origin
            rotated_direction = rotation_matrix @ direction
            
            arrow = self.ax_3d.quiver(
                rotated_origin[0], rotated_origin[1], rotated_origin[2],
                rotated_direction[0], rotated_direction[1], rotated_direction[2],
                color=colors[i], arrow_length_ratio=0.1, linewidth=3
            )
            self.arrow_plots.append(arrow)
    
    def update_data_markers(self, step):
        """Update markers on data plots"""
        if step >= len(self.time_data):
            return
        
        current_time = self.time_data[step]
        
        # Update Euler angles marker
        euler_values = [
            self.euler_angles[0][step],
            self.euler_angles[1][step],
            self.euler_angles[2][step]
        ]
        self.euler_marker.set_data([current_time, current_time, current_time], euler_values)
        
        # Update quaternion marker
        quat_values = [
            self.quaternions[0][step],
            self.quaternions[1][step],
            self.quaternions[2][step],
            self.quaternions[3][step]
        ]
        self.quat_marker.set_data([current_time, current_time, current_time, current_time], quat_values)
        
        # Update angular rates marker
        rate_values = [
            np.rad2deg(self.angular_rates[0][step]),
            np.rad2deg(self.angular_rates[1][step]),
            np.rad2deg(self.angular_rates[2][step])
        ]
        self.rates_marker.set_data([current_time, current_time, current_time], rate_values)
        
        # Update control inputs marker
        control_values = [
            self.control_inputs[0][step],
            self.control_inputs[1][step],
            self.control_inputs[2][step]
        ]
        self.control_marker.set_data([current_time, current_time, current_time], control_values)
    
    def update_info_text(self, step):
        """Update information text display"""
        if step >= len(self.time_data):
            return
        
        current_time = self.time_data[step]
        
        info_text = f"""Step: {step}/{len(self.time_data)-1}
Time: {current_time:.3f}s / {self.total_time:.3f}s

Euler Angles (deg):
Roll:  {self.euler_angles[0][step]:8.2f}°
Pitch: {self.euler_angles[1][step]:8.2f}°
Yaw:   {self.euler_angles[2][step]:8.2f}°

Quaternion:
w: {self.quaternions[0][step]:8.4f}
x: {self.quaternions[1][step]:8.4f}
y: {self.quaternions[2][step]:8.4f}
z: {self.quaternions[3][step]:8.4f}

Angular Rates (deg/s):
ωx: {np.rad2deg(self.angular_rates[0][step]):8.2f}
ωy: {np.rad2deg(self.angular_rates[1][step]):8.2f}
ωz: {np.rad2deg(self.angular_rates[2][step]):8.2f}

Control Inputs:
ux: {self.control_inputs[0][step]:8.3f}
uy: {self.control_inputs[1][step]:8.3f}
uz: {self.control_inputs[2][step]:8.3f}

Inertia (kg⋅m²):
Ix: {self.Ix:.3f}  Iy: {self.Iy:.3f}  Iz: {self.Iz:.3f}"""
        
        self.text_info.set_text(info_text)
    
    def animate_frame(self, frame):
        """Animation frame update function"""
        if not self.manual_control:
            self.current_step = frame
            self.frame_slider.set_val(frame)
            self.update_spacecraft(frame)
            self.update_data_markers(frame)
            self.update_info_text(frame)
            
            # Set the interval for the NEXT frame
            if self.use_realtime and frame < len(self.frame_intervals) - 1:
                next_interval = self.frame_intervals[frame]
                self.animation.event_source.interval = next_interval
            
        self.manual_control = False
        return []

    def start_animation(self, repeat=True):
        """Start the animation with real-time intervals"""
        n_frames = len(self.time_data)
        
        # Use first interval as starting interval
        if self.use_realtime and hasattr(self, 'frame_intervals'):
            initial_interval = self.frame_intervals[0]
            print(f"✓ Animation starting with real-time intervals")
            print(f"  First interval: {initial_interval:.2f}ms")
            print(f"  Expected total duration: {sum(self.frame_intervals)/1000:.2f}s")
        else:
            initial_interval = 100
            print(f"✓ Animation starting with constant {initial_interval}ms interval")
        
        self.animation = FuncAnimation(
            self.fig, self.animate_frame, frames=n_frames,
            interval=initial_interval, blit=False, repeat=repeat
        )
        return self.animation
    
    def save_animation(self, filename='spacecraft_animation.gif', fps=10):
        """Save animation as GIF"""
        if self.animation is None:
            self.start_animation(repeat=False)
        
        print(f"Saving animation to {filename}...")
        self.animation.save(filename, writer='pillow', fps=fps)
        print("Animation saved!")
    
    def show(self):
        """Display the animation"""
        self.start_animation()
        plt.show()

def main(csv_file='BC', Ix=1.0, Iy=1.0, Iz=1.0, use_realtime=True):
    """Main function to run the spacecraft animator"""
    
    # Create animator instance
    animator = SpacecraftAnimator(csv_file, Ix, Iy, Iz, use_realtime)
    
    # Show animation
    animator.show()

if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif len(sys.argv) == 2:
        print("⚠ Missing inertia moment arguments.")
        print("Usage: python animation.py derivative_supplier Ix Iy Iz")
        print("Using default values: Ix=1.0, Iy=1.0, Iz=1.0")
        main(sys.argv[1])
    else:
        print("⚠ Missing arguments.")
        print("Usage: python animation.py derivative_supplier Ix Iy Iz")
        print("Using default values: derivative_supplier=BC, Ix=1.0, Iy=1.0, Iz=1.0")
        main()
