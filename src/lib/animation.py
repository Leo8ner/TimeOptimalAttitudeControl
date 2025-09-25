import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import sys
from matplotlib.widgets import Button, Slider

class SpacecraftAnimator:
    def __init__(self, csv_file='../../output/trajectory.csv', Ix=1.0, Iy=1.0, Iz=1.0):
        # Initialize data containers
        self.euler_angles = [[], [], []]  # φ, θ, ψ
        self.quaternions = [[], [], [], []]  # w, x, y, z
        self.angular_rates = [[], [], []]  # ωx, ωy, ωz
        self.control_inputs = [[], [], []]  # ux, uy, uz
        self.time_data = []
        self.total_time = 0
        
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
        self.load_data(csv_file)
        
        # Create figure and setup
        self.setup_plot()
        
    def load_data(self, csv_file):
        """Load trajectory data from CSV and inertia from header file"""
        try:
            self.parse_csv_file(csv_file)
            # print(f"✓ Loaded trajectory data: {len(self.time_data)} points")
        except FileNotFoundError:
            print(f"⚠ CSV file '{csv_file}' not found. Using sample data.")
            self.generate_sample_data()
        except Exception as e:
            print(f"⚠ Error loading CSV: {e}. Using sample data.")
            self.generate_sample_data()
    
    def parse_csv_file(self, filename):
        """Parse the CSV file with the specified format"""
        with open(filename, 'r') as f:
            content = f.read().strip()
        
        lines = content.split('\n')
        current_section = ''
        
        # Initialize data containers
        state_rows = []
        control_rows = []
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line == 'X':
                current_section = 'states'
                continue
            elif line == 'U':
                current_section = 'control'
                continue
            elif line == 'T':
                current_section = 'time'
                continue
            elif line == 'dt':
                current_section = 'dt'
                continue
            
            # Parse data lines (check if line starts with a number)
            if line and (line[0].isdigit() or line[0] == '-' or line[0] == '.'):
                try:
                    values = [float(v.strip()) for v in line.split(',')]
                    
                    if current_section == 'states':
                        state_rows.append(values)
                    elif current_section == 'control':
                        control_rows.append(values)
                    elif current_section == 'time':
                        self.total_time = values[0]
                    elif current_section == 'dt':
                        # Generate time data from dt values
                        time = 0
                        self.time_data = []
                        self.time_data.append(time)
                        for dt in values:
                            time += dt
                            self.time_data.append(time)
                            self.dt = dt
                except ValueError:
                    continue  # Skip lines that can't be parsed as numbers
        
        # Now assign the parsed rows to the correct arrays
        if len(state_rows) >= 10:
            # Euler angles (rows 0-2)
            self.euler_angles[0] = state_rows[0]  # φ (roll)
            self.euler_angles[1] = state_rows[1]  # θ (pitch)  
            self.euler_angles[2] = state_rows[2]  # ψ (yaw)
            
            # Quaternions (rows 3-6)
            self.quaternions[0] = state_rows[3]  # w
            self.quaternions[1] = state_rows[4]  # x
            self.quaternions[2] = state_rows[5]  # y
            self.quaternions[3] = state_rows[6]  # z
            
            # Angular rates (rows 7-9)
            self.angular_rates[0] = state_rows[7]  # ωx
            self.angular_rates[1] = state_rows[8]  # ωy
            self.angular_rates[2] = state_rows[9]  # ωz
        
        # Control inputs
        if len(control_rows) >= 3:
            self.control_inputs[0] = np.insert(control_rows[0], 0, 0.0)  # ux
            self.control_inputs[1] = np.insert(control_rows[1], 0, 0.0)  # uy
            self.control_inputs[2] = np.insert(control_rows[2], 0, 0.0)  # uz
        
        # Debug output
        # print(f"Debug: Parsed {len(state_rows)} state rows")
        # print(f"Debug: Parsed {len(control_rows)} control rows")
        # print(f"Debug: Euler angles: {len(self.euler_angles[0])} points")
        # print(f"Debug: Quaternions: {len(self.quaternions[0])} points") 
        # print(f"Debug: Angular rates: {len(self.angular_rates[0])} points")
        # print(f"Debug: Control inputs: {len(self.control_inputs[0])} points") 
        # print(f"Debug: Time points: {len(self.time_data)}")
        # print(f"Debug: Total time: {self.total_time}")
        
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

        # plt.tight_layout()
    
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
        # Create button axes (position: left, bottom, width, height)
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
        # Create slider axes (position: left, bottom, width, height)
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
        if not self.manual_control:  # Only update if not manually controlled
            self.current_step = frame
            self.frame_slider.set_val(frame)  # Update slider position
            self.update_spacecraft(frame)
            self.update_data_markers(frame)
            self.update_info_text(frame)
        self.manual_control = False  # Reset manual control flag
        return []
    
    def start_animation(self, interval=100, repeat=True):
        """Start the animation"""
        n_frames = len(self.time_data)
        self.animation = FuncAnimation(
            self.fig, self.animate_frame, frames=n_frames,
            interval=interval, blit=False, repeat=repeat
        )
        return self.animation
    
    def save_animation(self, filename='spacecraft_animation.gif', fps=10):
        """Save animation as GIF"""
        if self.animation is None:
            self.start_animation(interval=100, repeat=False)
        
        print(f"Saving animation to {filename}...")
        self.animation.save(filename, writer='pillow', fps=fps)
        print("Animation saved!")
    
    def show(self):
        """Display the animation"""
        self.start_animation()
        plt.show()

def main(csv_file, Ix, Iy, Iz):
    """Main function to run the spacecraft animator"""
    
    # Create animator instance
    animator = SpacecraftAnimator(csv_file, Ix, Iy, Iz)
    
    # Show animation
    animator.show()

if __name__ == "__main__":
    if len(sys.argv) == 5:
        main("../output/" + sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif len(sys.argv) == 2:
        print("⚠ Missing inertia moment arguments.")
        print("Usage: python animation.py trajectory.csv Ix Iy Iz")
        print("Using default values: Ix=1.0, Iy=1.0, Iz=1.0")
        main("../output/" + sys.argv[1], 1.0, 1.0, 1.0)
    else:
        print("⚠ Missing arguments.")
        print("Usage: python animation.py trajectory.csv Ix Iy Iz")
        print("Using default values: Ix=1.0, Iy=1.0, Iz=1.0")
        main("../../output/trajectory.csv", 1.0, 1.0, 1.0)
