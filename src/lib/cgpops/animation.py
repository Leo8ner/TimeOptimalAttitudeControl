"""Spacecraft attitude trajectory visualization and animation tool.

This module provides real-time 3D visualization of spacecraft attitude
maneuvers from MATLAB-exported trajectory optimization results. It displays
the spacecraft orientation, angular rates, quaternions, and control inputs
synchronized in real-time.

Typical usage example:
    animator = SpacecraftAnimator('BC', Ix=1.5, Iy=2.0, Iz=1.0)
    animator.show()
"""

import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


class SpacecraftAnimator:
    """Animates spacecraft attitude trajectories from MATLAB optimization output.
    
    This class loads trajectory data from MATLAB .m files, creates a 3D
    visualization of the spacecraft, and animates its attitude maneuver
    with synchronized data plots showing Euler angles, quaternions,
    angular rates, and control inputs.
    
    Attributes:
        euler_angles: List of three lists containing roll, pitch, yaw angles (degrees).
        quaternions: List of four lists containing w, x, y, z quaternion components.
        angular_rates: List of three lists containing ωx, ωy, ωz rates (rad/s).
        control_inputs: List of three lists containing ux, uy, uz control values.
        time_data: List of time points corresponding to trajectory samples.
        total_time: Total trajectory duration in seconds.
        Ix, Iy, Iz: Spacecraft moments of inertia (kg⋅m²).
        use_realtime: If True, animation plays at real-time speed.
    """
    
    # Class-level constants
    DEFAULT_INTERVAL_MS = 100  # Default animation frame interval
    MIN_INTERVAL_MS = 1  # Minimum frame interval
    GEOMETRY_SCALE_FACTOR = 2.0  # Spacecraft geometry scaling
    ARROW_LENGTH_MULTIPLIER = 2.0  # Orientation arrow length multiplier
    ARROW_HEAD_RATIO = 0.1  # Arrow head size ratio
    REFERENCE_ARROW_LENGTH = 2.0  # Reference frame arrow length
    REFERENCE_ARROW_ALPHA = 0.3  # Reference frame transparency
    PLOT_FIGURE_SIZE = (18, 12)  # Figure dimensions in inches
    AXIS_LIMITS = (-4, 4)  # 3D plot axis limits
    SPACECRAFT_ALPHA = 0.7  # Spacecraft body transparency
    MARKER_SIZE = 8  # Data plot marker size
    ARROW_LINE_WIDTH = 3  # Orientation arrow line width
    
    # Grid layout ratios
    GRID_HEIGHT_RATIOS = [3, 1]
    GRID_WIDTH_RATIOS = [2, 2, 2, 2]
    
    # Button and slider positions [left, bottom, width, height]
    BUTTON_POSITION = [0.1, 0.4, 0.1, 0.04]
    SLIDER_POSITION = [0.05, 0.35, 0.2, 0.03]
    INFO_TEXT_POSITION = (-0.50, 0.98)

    def __init__(
        self,
        matlab_file_suffix: str = 'BC',
        Ix: float = 1.0,
        Iy: float = 1.0,
        Iz: float = 1.0,
        use_realtime: bool = True
    ):
        """Initializes the SpacecraftAnimator with trajectory data.
        
        Args:
            matlab_file_suffix: Suffix of the MATLAB output file to load.
                File should be named 'cgpopsIPOPTSolution{suffix}.m'
            Ix: Spacecraft moment of inertia about X-axis (kg⋅m²).
            Iy: Spacecraft moment of inertia about Y-axis (kg⋅m²).
            Iz: Spacecraft moment of inertia about Z-axis (kg⋅m²).
            use_realtime: If True, animation uses real-time frame intervals
                based on trajectory timestamps. If False, uses constant intervals.
                
        Raises:
            FileNotFoundError: If the specified MATLAB file doesn't exist.
            ValueError: If the MATLAB file cannot be parsed correctly.
        """
        # Initialize data storage containers
        self.euler_angles: List[List[float]] = [[], [], []]  # φ, θ, ψ
        self.quaternions: List[List[float]] = [[], [], [], []]  # w, x, y, z
        self.angular_rates: List[List[float]] = [[], [], []]  # ωx, ωy, ωz
        self.control_inputs: List[List[float]] = [[], [], []]  # ux, uy, uz
        self.time_data: List[float] = []
        self.total_time: float = 0.0
        self.frame_intervals: List[float] = []
        
        # Spacecraft physical properties
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        
        # Animation state
        self.use_realtime = use_realtime
        self.current_step: int = 0
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.manual_control: bool = False
        
        # Visualization objects (initialized during setup)
        self.animation: Optional[FuncAnimation] = None
        self.pause_button: Optional[Button] = None
        self.frame_slider: Optional[Slider] = None
        self.spacecraft_plot: Optional[Poly3DCollection] = None
        self.arrow_plots: List = []
        
        # Load trajectory data and setup visualization
        matlab_file = f'../output/cgpopsIPOPTSolution{matlab_file_suffix}.m'
        self._load_data(matlab_file)
        self._calculate_frame_intervals()
        self._setup_plot()

    # =========================================================================
    # Data Loading Methods
    # =========================================================================

    def _load_data(self, matlab_file: str) -> None:
        """Loads and validates trajectory data from MATLAB file.
        
        Args:
            matlab_file: Path to the MATLAB .m file containing trajectory data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid or data is incomplete.
        """
        try:
            self._parse_matlab_file(matlab_file)
            print(f"✓ Loaded trajectory data: {len(self.time_data)} points")
        except FileNotFoundError:
            print(f"⚠ MATLAB file '{matlab_file}' not found.")
            raise
        except Exception as e:
            print(f"⚠ Error loading MATLAB file: {e}")
            raise ValueError(f"Failed to parse MATLAB file: {e}") from e

    def _parse_matlab_file(self, filename: str) -> None:
        """Parses MATLAB .m file and extracts trajectory data.
        
        The MATLAB file format contains:
        - system{suffix}.phase(1).tf: Total trajectory time
        - system{suffix}.phase(1).x(1-7).point(i): State variables
          (quaternions w,x,y,z and angular rates ωx,ωy,ωz)
        - system{suffix}.phase(1).u(1-3).point(i): Control inputs
        - system{suffix}.phase(1).t.point(i): Time points
        
        Args:
            filename: Path to the MATLAB .m file.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If required data patterns are not found.
        """
        # Extract file suffix from filename (last 2 characters before extension)
        name_without_ext = os.path.splitext(filename)[0]
        suffix = name_without_ext[-2:]
        
        # Read entire file content
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract total trajectory time
        tf_pattern = rf'system{suffix}\.phase\(1\)\.tf\s*=\s*([\d.e+-]+);'
        tf_match = re.search(tf_pattern, content)
        if not tf_match:
            raise ValueError("Could not find total time (tf) in MATLAB file")
        self.total_time = float(tf_match.group(1))
        
        # Extract state variables x(1-7): [qw, qx, qy, qz, ωx, ωy, ωz]
        states = self._extract_numbered_data(content, suffix, 'x', 7)
        
        # Extract control inputs u(1-3): [ux, uy, uz]
        controls = self._extract_numbered_data(content, suffix, 'u', 3)
        
        # Extract time points
        time_pattern = rf'system{suffix}\.phase\(1\)\.t\.point\((\d+)\)\s*=\s*([-\d.e+-]+);'
        time_matches = re.findall(time_pattern, content)
        if not time_matches:
            raise ValueError("Could not find time data in MATLAB file")
        self.time_data = [float(val) for idx, val in sorted(time_matches, key=lambda x: int(x[0]))]
        
        # Assign states: quaternions (w, x, y, z) and angular rates (ωx, ωy, ωz)
        quat_w, quat_x, quat_y, quat_z = states[0], states[1], states[2], states[3]
        omega_x, omega_y, omega_z = states[4], states[5], states[6]
        
        # Convert quaternions to Euler angles for visualization
        self._convert_quaternions_to_euler(quat_w, quat_x, quat_y, quat_z)
        
        # Store quaternion data
        self.quaternions = [quat_w, quat_x, quat_y, quat_z]
        
        # Store angular rates (rad/s)
        self.angular_rates = [omega_x, omega_y, omega_z]
        
        # Store control inputs
        self.control_inputs = controls

    def _extract_numbered_data(
        self,
        content: str,
        suffix: str,
        variable: str,
        count: int
    ) -> List[List[float]]:
        """Extracts numbered data points from MATLAB file content.
        
        Args:
            content: Full MATLAB file content as string.
            suffix: File suffix used in variable names.
            variable: Variable name ('x' for states, 'u' for controls).
            count: Number of indexed variables to extract (e.g., 7 for x(1-7)).
            
        Returns:
            List of lists, where each inner list contains the time series
            for one variable index.
            
        Raises:
            ValueError: If expected data is not found.
        """
        data = []
        for i in range(1, count + 1):
            pattern = rf'system{suffix}\.phase\(1\)\.{variable}\({i}\)\.point\((\d+)\)\s*=\s*([-\d.e+-]+);'
            matches = re.findall(pattern, content)
            if not matches:
                raise ValueError(f"Could not find data for {variable}({i})")
            values = [float(val) for idx, val in sorted(matches, key=lambda x: int(x[0]))]
            data.append(values)
        return data

    def _convert_quaternions_to_euler(
        self,
        quat_w: List[float],
        quat_x: List[float],
        quat_y: List[float],
        quat_z: List[float]
    ) -> None:
        """Converts quaternion time series to Euler angles (roll, pitch, yaw).
        
        Uses XYZ intrinsic rotation sequence. Results are stored in degrees
        in the euler_angles attribute.
        
        Args:
            quat_w: Quaternion scalar components (w).
            quat_x: Quaternion vector x components.
            quat_y: Quaternion vector y components.
            quat_z: Quaternion vector z components.
        """
        n_points = len(quat_w)
        euler_angles = np.zeros((3, n_points))
        
        for i in range(n_points):
            # scipy expects quaternion as [x, y, z, w]
            quat = [quat_x[i], quat_y[i], quat_z[i], quat_w[i]]
            rot = Rotation.from_quat(quat)
            # Convert to Euler angles using XYZ sequence
            euler = rot.as_euler('XYZ', degrees=True)  # [roll, pitch, yaw]
            euler_angles[:, i] = euler
        
        # Store as separate lists for each angle
        self.euler_angles[0] = euler_angles[0, :].tolist()  # roll (φ)
        self.euler_angles[1] = euler_angles[1, :].tolist()  # pitch (θ)
        self.euler_angles[2] = euler_angles[2, :].tolist()  # yaw (ψ)

    def _calculate_frame_intervals(self) -> None:
        """Calculates time intervals between frames for real-time playback.
        
        Computes the time difference between consecutive trajectory points
        and converts to milliseconds for matplotlib animation intervals.
        Ensures minimum interval to prevent animation issues.
        """
        if len(self.time_data) < 2:
            self.frame_intervals = [self.DEFAULT_INTERVAL_MS]
            print("⚠ Insufficient time data, using default interval")
            return
        
        # Calculate dt between consecutive frames and convert to milliseconds
        self.frame_intervals = []
        for i in range(len(self.time_data) - 1):
            dt_ms = (self.time_data[i + 1] - self.time_data[i]) * 1000
            # Enforce minimum interval to prevent rendering issues
            self.frame_intervals.append(max(dt_ms, self.MIN_INTERVAL_MS))
        
        # Duplicate last interval for final frame
        if self.frame_intervals:
            self.frame_intervals.append(self.frame_intervals[-1])
        else:
            self.frame_intervals.append(self.DEFAULT_INTERVAL_MS)
        
        # Print interval statistics
        avg_interval = np.mean(self.frame_intervals)
        min_interval = np.min(self.frame_intervals)
        max_interval = np.max(self.frame_intervals)
        total_real_time = sum(self.frame_intervals) / 1000
        
        print(f"✓ Frame intervals calculated:")
        print(f"  - Average: {avg_interval:.2f}ms")
        print(f"  - Min: {min_interval:.2f}ms")
        print(f"  - Max: {max_interval:.2f}ms")
        print(f"  - Total frames: {len(self.time_data)}")
        print(f"  - Total real time: {total_real_time:.2f}s")

    # =========================================================================
    # Geometry Creation Methods
    # =========================================================================

    def _create_spacecraft_geometry(self) -> Tuple[List, np.ndarray, List[float]]:
        """Creates 3D geometry for spacecraft visualization.
        
        Generates a rectangular box with dimensions proportional to the
        spacecraft's moments of inertia. Larger inertia corresponds to
        larger dimension along that axis.
        
        Returns:
            Tuple containing:
                - faces: List of face vertices for Poly3DCollection
                - vertices: ndarray of 8 corner vertices
                - dimensions: List of [dim_x, dim_y, dim_z] in arbitrary units
        """
        # Calculate dimensions proportional to inertia moments
        max_inertia = max(self.Ix, self.Iy, self.Iz)
        
        dim_x = self.GEOMETRY_SCALE_FACTOR * np.sqrt(self.Ix / max_inertia)
        dim_y = self.GEOMETRY_SCALE_FACTOR * np.sqrt(self.Iy / max_inertia)
        dim_z = self.GEOMETRY_SCALE_FACTOR * np.sqrt(self.Iz / max_inertia)
        
        # Create 8 vertices of the box (right-handed coordinate system)
        vertices = np.array([
            [-dim_x/2, -dim_y/2, -dim_z/2],  # 0: Left-bottom-rear
            [+dim_x/2, -dim_y/2, -dim_z/2],  # 1: Right-bottom-rear
            [+dim_x/2, +dim_y/2, -dim_z/2],  # 2: Right-top-rear
            [-dim_x/2, +dim_y/2, -dim_z/2],  # 3: Left-top-rear
            [-dim_x/2, -dim_y/2, +dim_z/2],  # 4: Left-bottom-front
            [+dim_x/2, -dim_y/2, +dim_z/2],  # 5: Right-bottom-front
            [+dim_x/2, +dim_y/2, +dim_z/2],  # 6: Right-top-front
            [-dim_x/2, +dim_y/2, +dim_z/2]   # 7: Left-top-front
        ])
        
        # Define 6 faces of the box (each face is a quadrilateral)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom (-z)
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top (+z)
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front (-y)
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back (+y)
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right (+x)
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left (-x)
        ]
        
        return faces, vertices, [dim_x, dim_y, dim_z]

    def _create_orientation_arrows(
        self,
        dimensions: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates arrow origins and directions for body-fixed axes.
        
        Generates arrows extending from the spacecraft body to indicate
        the orientation of the body-fixed coordinate frame.
        
        Args:
            dimensions: Spacecraft box dimensions [dim_x, dim_y, dim_z].
            
        Returns:
            Tuple containing:
                - origins: (3, 3) array of arrow starting points for X, Y, Z axes
                - directions: (3, 3) array of arrow direction vectors
        """
        dim_x, dim_y, dim_z = dimensions
        arrow_length = max(dimensions) * self.ARROW_LENGTH_MULTIPLIER
        
        # All arrows originate from the same corner of the spacecraft
        base_origin = np.array([-dim_x/2, -dim_y/2, -dim_z/2])
        origins = np.array([base_origin, base_origin, base_origin])
        
        # Arrow directions along each principal axis
        directions = np.array([
            [arrow_length, 0, 0],  # X-axis (red)
            [0, arrow_length, 0],  # Y-axis (green)
            [0, 0, arrow_length]   # Z-axis (blue)
        ])
        
        return origins, directions

    # =========================================================================
    # Plot Setup Methods
    # =========================================================================

    def _setup_plot(self) -> None:
        """Initializes all matplotlib figures, subplots, and UI elements.
        
        Creates a figure with:
        - 3D spacecraft visualization (top, spanning full width)
        - Four 2D data plots (bottom row: Euler angles, quaternions,
          angular rates, control inputs)
        - Pause/resume button
        - Frame slider for manual control
        """
        # Create figure with specified size
        self.fig = plt.figure(figsize=self.PLOT_FIGURE_SIZE)
        
        # Create grid layout: 2 rows (3D view, data plots), 4 columns
        grid_spec = self.fig.add_gridspec(
            2, 4,
            height_ratios=self.GRID_HEIGHT_RATIOS,
            width_ratios=self.GRID_WIDTH_RATIOS
        )
        
        # Main 3D plot spans entire top row
        self.ax_3d = self.fig.add_subplot(grid_spec[0, :], projection='3d')
        
        # Four data plots in bottom row
        self.ax_euler = self.fig.add_subplot(grid_spec[1, 0])
        self.ax_quat = self.fig.add_subplot(grid_spec[1, 1])
        self.ax_rates = self.fig.add_subplot(grid_spec[1, 2])
        self.ax_control = self.fig.add_subplot(grid_spec[1, 3])
        
        # Configure 3D plot appearance and limits
        self._configure_3d_plot()
        
        # Create and add spacecraft geometry
        self.faces, self.vertices, self.dimensions = self._create_spacecraft_geometry()
        self._initialize_3d_objects()
        
        # Setup all 2D data plots
        self._setup_data_plots()
        
        # Add information text display
        self._setup_info_text()
        
        # Add interactive controls
        self._setup_pause_button()
        self._setup_frame_slider()

    def _configure_3d_plot(self) -> None:
        """Configures 3D plot axes, labels, and title."""
        limits = self.AXIS_LIMITS
        self.ax_3d.set_xlim(limits)
        self.ax_3d.set_ylim(limits)
        self.ax_3d.set_zlim(limits)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('Spacecraft Attitude Animation')

    def _initialize_3d_objects(self) -> None:
        """Initializes 3D visualization objects in the main plot.
        
        Creates:
        - Spacecraft body (Poly3DCollection)
        - Body-fixed orientation arrows (red=X, green=Y, blue=Z)
        - Inertial reference frame arrows (semi-transparent)
        """
        # Create spacecraft body as semi-transparent polyhedron
        self.spacecraft_plot = Poly3DCollection(
            self.faces,
            alpha=self.SPACECRAFT_ALPHA,
            facecolor='gray',
            edgecolor='black'
        )
        self.ax_3d.add_collection3d(self.spacecraft_plot)
        
        # Initialize body-fixed orientation arrows (will be updated each frame)
        self.arrow_plots = []
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            # Create initial arrow (will be repositioned during animation)
            arrow = self.ax_3d.quiver(
                0, 0, 0, 1, 0, 0,
                color=color,
                arrow_length_ratio=self.ARROW_HEAD_RATIO
            )
            self.arrow_plots.append(arrow)
        
        # Add fixed inertial reference frame (semi-transparent)
        ref_length = self.REFERENCE_ARROW_LENGTH
        ref_alpha = self.REFERENCE_ARROW_ALPHA
        head_ratio = self.ARROW_HEAD_RATIO / 2  # Smaller heads for reference
        
        self.ax_3d.quiver(
            0, 0, 0, ref_length, 0, 0,
            color='red', alpha=ref_alpha, arrow_length_ratio=head_ratio
        )
        self.ax_3d.quiver(
            0, 0, 0, 0, ref_length, 0,
            color='green', alpha=ref_alpha, arrow_length_ratio=head_ratio
        )
        self.ax_3d.quiver(
            0, 0, 0, 0, 0, ref_length,
            color='blue', alpha=ref_alpha, arrow_length_ratio=head_ratio
        )

    def _setup_data_plots(self) -> None:
        """Configures all 2D data visualization plots.
        
        Creates four synchronized plots:
        1. Euler angles (roll, pitch, yaw) vs time
        2. Quaternion components (w, x, y, z) vs time
        3. Angular rates (ωx, ωy, ωz) vs time
        4. Control inputs (ux, uy, uz) vs time
        
        Each plot includes a black marker showing the current time step.
        """
        # Plot 1: Euler Angles (degrees)
        self.ax_euler.plot(self.time_data, self.euler_angles[0], 'r-', label='Roll (φ)')
        self.ax_euler.plot(self.time_data, self.euler_angles[1], 'g-', label='Pitch (θ)')
        self.ax_euler.plot(self.time_data, self.euler_angles[2], 'b-', label='Yaw (ψ)')
        self.euler_marker, = self.ax_euler.plot([], [], 'ko', markersize=self.MARKER_SIZE)
        self.ax_euler.set_xlabel('Time (s)')
        self.ax_euler.set_ylabel('Angle (deg)')
        self.ax_euler.set_title('Euler Angles')
        self.ax_euler.legend(fontsize=12)
        self.ax_euler.grid(True)
        
        # Plot 2: Quaternions
        self.ax_quat.plot(self.time_data, self.quaternions[0], 'k-', label='w')
        self.ax_quat.plot(self.time_data, self.quaternions[1], 'r-', label='x')
        self.ax_quat.plot(self.time_data, self.quaternions[2], 'g-', label='y')
        self.ax_quat.plot(self.time_data, self.quaternions[3], 'b-', label='z')
        self.quat_marker, = self.ax_quat.plot([], [], 'ko', markersize=self.MARKER_SIZE)
        self.ax_quat.set_xlabel('Time (s)')
        self.ax_quat.set_ylabel('Quaternion')
        self.ax_quat.set_title('Quaternions')
        self.ax_quat.legend(fontsize=12)
        self.ax_quat.grid(True)
        
        # Plot 3: Angular Rates (converted to deg/s for readability)
        self.ax_rates.plot(
            self.time_data,
            np.rad2deg(self.angular_rates[0]),
            'r-', label='ωx'
        )
        self.ax_rates.plot(
            self.time_data,
            np.rad2deg(self.angular_rates[1]),
            'g-', label='ωy'
        )
        self.ax_rates.plot(
            self.time_data,
            np.rad2deg(self.angular_rates[2]),
            'b-', label='ωz'
        )
        self.rates_marker, = self.ax_rates.plot([], [], 'ko', markersize=self.MARKER_SIZE)
        self.ax_rates.set_xlabel('Time (s)')
        self.ax_rates.set_ylabel('Rate (deg/s)')
        self.ax_rates.set_title('Angular Rates')
        self.ax_rates.legend(fontsize=12)
        self.ax_rates.grid(True)
        
        # Plot 4: Control Inputs
        self.ax_control.plot(self.time_data, self.control_inputs[0], 'r-', label='ux')
        self.ax_control.plot(self.time_data, self.control_inputs[1], 'g-', label='uy')
        self.ax_control.plot(self.time_data, self.control_inputs[2], 'b-', label='uz')
        self.control_marker, = self.ax_control.plot([], [], 'ko', markersize=self.MARKER_SIZE)
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Control')
        self.ax_control.set_title('Control Inputs')
        self.ax_control.legend(fontsize=12)
        self.ax_control.grid(True)

    def _setup_info_text(self) -> None:
        """Creates text display for current state information in 3D plot."""
        self.text_info = self.ax_3d.text2D(
            *self.INFO_TEXT_POSITION,
            '',
            transform=self.ax_3d.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=12
        )

    def _setup_pause_button(self) -> None:
        """Creates pause/resume button for animation control."""
        button_ax = plt.axes(self.BUTTON_POSITION)
        self.pause_button = Button(button_ax, 'Pause')
        self.pause_button.on_clicked(self._toggle_pause)

    def _setup_frame_slider(self) -> None:
        """Creates slider for manual frame selection."""
        slider_ax = plt.axes(self.SLIDER_POSITION)
        self.frame_slider = Slider(
            slider_ax,
            'Step',
            0,
            len(self.time_data) - 1,
            valinit=0,
            valfmt='%d',
            valstep=1
        )
        self.frame_slider.on_changed(self._update_from_slider)

    # =========================================================================
    # Animation Update Methods
    # =========================================================================

    def _update_spacecraft(self, step: int) -> None:
        """Updates spacecraft 3D visualization for current time step.
        
        Applies the quaternion rotation at the given step to both the
        spacecraft body geometry and the body-fixed orientation arrows.
        
        Args:
            step: Index of current time step in trajectory data.
        """
        if step >= len(self.time_data):
            return
        
        # Extract quaternion for current time step (scipy format: [x, y, z, w])
        quat = [
            self.quaternions[1][step],  # x
            self.quaternions[2][step],  # y
            self.quaternions[3][step],  # z
            self.quaternions[0][step]   # w
        ]
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(quat)
        rotation_matrix = rotation.as_matrix()
        
        # Rotate spacecraft body vertices
        rotated_faces = []
        for face in self.faces:
            rotated_face = [rotation_matrix @ vertex for vertex in face]
            rotated_faces.append(rotated_face)
        
        # Update spacecraft polyhedron
        self.spacecraft_plot.set_verts(rotated_faces)
        
        # Update body-fixed orientation arrows
        self._update_orientation_arrows(rotation_matrix)

    def _update_orientation_arrows(self, rotation_matrix: np.ndarray) -> None:
        """Updates body-fixed coordinate frame arrows with new orientation.
        
        Removes old arrows and creates new ones with the rotated orientation.
        
        Args:
            rotation_matrix: 3x3 rotation matrix representing current attitude.
        """
        # Get arrow geometry in body frame
        origins, directions = self._create_orientation_arrows(self.dimensions)
        colors = ['red', 'green', 'blue']
        
        # Remove existing arrows from plot
        for arrow in self.arrow_plots:
            arrow.remove()
        self.arrow_plots.clear()
        
        # Create new arrows with rotated orientations
        for i, (origin, direction, color) in enumerate(zip(origins, directions, colors)):
            # Rotate arrow origin and direction to inertial frame
            rotated_origin = rotation_matrix @ origin
            rotated_direction = rotation_matrix @ direction
            
            # Create arrow in 3D plot
            arrow = self.ax_3d.quiver(
                rotated_origin[0], rotated_origin[1], rotated_origin[2],
                rotated_direction[0], rotated_direction[1], rotated_direction[2],
                color=color,
                arrow_length_ratio=self.ARROW_HEAD_RATIO,
                linewidth=self.ARROW_LINE_WIDTH
            )
            self.arrow_plots.append(arrow)

    def _update_data_markers(self, step: int) -> None:
        """Updates position markers on all 2D data plots.
        
        Places a black marker at the current time on each of the four
        data plots to show synchronized state.
        
        Args:
            step: Index of current time step in trajectory data.
        """
        if step >= len(self.time_data):
            return
        
        current_time = self.time_data[step]
        
        # Update Euler angles marker (3 points: roll, pitch, yaw)
        euler_values = [self.euler_angles[i][step] for i in range(3)]
        self.euler_marker.set_data([current_time] * 3, euler_values)
        
        # Update quaternion marker (4 points: w, x, y, z)
        quat_values = [self.quaternions[i][step] for i in range(4)]
        self.quat_marker.set_data([current_time] * 4, quat_values)
        
        # Update angular rates marker (3 points, converted to deg/s)
        rate_values = [np.rad2deg(self.angular_rates[i][step]) for i in range(3)]
        self.rates_marker.set_data([current_time] * 3, rate_values)
        
        # Update control inputs marker (3 points: ux, uy, uz)
        control_values = [self.control_inputs[i][step] for i in range(3)]
        self.control_marker.set_data([current_time] * 3, control_values)

    def _update_info_text(self, step: int) -> None:
        """Updates information text display with current state values.
        
        Displays:
        - Step and time information
        - Euler angles (degrees)
        - Quaternion components
        - Angular rates (deg/s)
        - Control inputs
        - Spacecraft inertia properties
        
        Args:
            step: Index of current time step in trajectory data.
        """
        if step >= len(self.time_data):
            return
        
        current_time = self.time_data[step]
        
        # Format multiline information string
        info_text = (
            f"Step: {step}/{len(self.time_data) - 1}\n"
            f"Time: {current_time:.3f}s / {self.total_time:.3f}s\n"
            f"\n"
            f"Euler Angles (deg):\n"
            f"Roll:  {self.euler_angles[0][step]:8.2f}°\n"
            f"Pitch: {self.euler_angles[1][step]:8.2f}°\n"
            f"Yaw:   {self.euler_angles[2][step]:8.2f}°\n"
            f"\n"
            f"Quaternion:\n"
            f"w: {self.quaternions[0][step]:8.4f}\n"
            f"x: {self.quaternions[1][step]:8.4f}\n"
            f"y: {self.quaternions[2][step]:8.4f}\n"
            f"z: {self.quaternions[3][step]:8.4f}\n"
            f"\n"
            f"Angular Rates (deg/s):\n"
            f"ωx: {np.rad2deg(self.angular_rates[0][step]):8.2f}\n"
            f"ωy: {np.rad2deg(self.angular_rates[1][step]):8.2f}\n"
            f"ωz: {np.rad2deg(self.angular_rates[2][step]):8.2f}\n"
            f"\n"
            f"Control Inputs:\n"
            f"ux: {self.control_inputs[0][step]:8.3f}\n"
            f"uy: {self.control_inputs[1][step]:8.3f}\n"
            f"uz: {self.control_inputs[2][step]:8.3f}\n"
            f"\n"
            f"Inertia (kg⋅m²):\n"
            f"Ix: {self.Ix:.3f}  Iy: {self.Iy:.3f}  Iz: {self.Iz:.3f}"
        )
        
        self.text_info.set_text(info_text)

    def _animate_frame(self, frame: int) -> List:
        """Main animation callback function called by FuncAnimation.
        
        Updates all visualization components for the current frame unless
        manual control is active (slider being used).
        
        Args:
            frame: Frame number (0 to len(time_data)-1).
            
        Returns:
            Empty list (required by FuncAnimation blit interface).
        """
        if not self.manual_control:
            self.current_step = frame
            self.frame_slider.set_val(frame)
            
            # Update all visualization components
            self._update_spacecraft(frame)
            self._update_data_markers(frame)
            self._update_info_text(frame)
            
            # Set interval for next frame (real-time mode)
            if self.use_realtime and frame < len(self.frame_intervals) - 1:
                next_interval = self.frame_intervals[frame]
                self.animation.event_source.interval = next_interval
        
        # Reset manual control flag after processing
        self.manual_control = False
        return []

    # =========================================================================
    # Control Methods
    # =========================================================================

    def _toggle_pause(self, event) -> None:
        """Handles pause/resume button clicks.
        
        Args:
            event: Matplotlib button click event (unused).
        """
        if self.is_paused:
            self.animation.resume()
            self.pause_button.label.set_text('Pause')
            self.is_paused = False
        else:
            self.animation.pause()
            self.pause_button.label.set_text('Resume')
            self.is_paused = True

    def _update_from_slider(self, val: float) -> None:
        """Handles slider value changes for manual frame selection.
        
        Args:
            val: New slider value (frame number).
        """
        frame = int(self.frame_slider.val)
        self.current_step = frame
        self.manual_control = True
        
        # Update all visualization components
        self._update_spacecraft(frame)
        self._update_data_markers(frame)
        self._update_info_text(frame)
        
        # Redraw canvas
        self.fig.canvas.draw_idle()

    # =========================================================================
    # Public Interface Methods
    # =========================================================================

    def start_animation(self, repeat: bool = True) -> FuncAnimation:
        """Starts the spacecraft attitude animation.
        
        Creates a FuncAnimation object that updates the visualization at
        either constant or variable (real-time) intervals based on the
        use_realtime setting.
        
        Args:
            repeat: If True, animation loops continuously. If False, plays once.
            
        Returns:
            FuncAnimation object controlling the animation.
        """
        n_frames = len(self.time_data)
        
        # Determine initial frame interval
        if self.use_realtime and hasattr(self, 'frame_intervals'):
            initial_interval = self.frame_intervals[0]
            total_duration = sum(self.frame_intervals) / 1000
            print(f"✓ Animation starting with real-time intervals")
            print(f"  First interval: {initial_interval:.2f}ms")
            print(f"  Expected total duration: {total_duration:.2f}s")
        else:
            initial_interval = self.DEFAULT_INTERVAL_MS
            print(f"✓ Animation starting with constant {initial_interval}ms interval")
        
        # Create FuncAnimation object
        self.animation = FuncAnimation(
            self.fig,
            self._animate_frame,
            frames=n_frames,
            interval=initial_interval,
            blit=False,
            repeat=repeat
        )
        
        return self.animation

    def save_animation(self, filename: str = 'spacecraft_animation.gif', fps: int = 10) -> None:
        """Saves the animation as a GIF file.
        
        Args:
            filename: Output filename for the GIF.
            fps: Frames per second for the output GIF.
        """
        if self.animation is None:
            self.start_animation(repeat=False)
        
        print(f"Saving animation to {filename}...")
        self.animation.save(filename, writer='pillow', fps=fps)
        print("Animation saved!")

    def show(self) -> None:
        """Displays the animation window.
        
        Starts the animation and shows the matplotlib figure with all
        interactive controls.
        """
        self.start_animation()
        plt.show()


def main(
    suffix: str = 'BC',
    Ix: float = 1.0,
    Iy: float = 1.0,
    Iz: float = 1.0,
    use_realtime: bool = True
) -> None:
    """Main entry point for spacecraft attitude animation.
    
    Creates a SpacecraftAnimator instance and displays the visualization.
    
    Args:
        suffix: MATLAB file suffix identifying which trajectory to load.
        Ix: Spacecraft moment of inertia about X-axis (kg⋅m²).
        Iy: Spacecraft moment of inertia about Y-axis (kg⋅m²).
        Iz: Spacecraft moment of inertia about Z-axis (kg⋅m²).
        use_realtime: Whether to use real-time playback intervals.
    """
    animator = SpacecraftAnimator(suffix, Ix, Iy, Iz, use_realtime)
    animator.show()


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 5:
        # Full arguments: suffix Ix Iy Iz
        main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif len(sys.argv) == 2:
        # Only suffix provided
        print("⚠ Missing inertia moment arguments.")
        print("Usage: python animation.py derivative_supplier Ix Iy Iz")
        print("Using default values: Ix=1.0, Iy=1.0, Iz=1.0")
        main(sys.argv[1])
    else:
        # No arguments
        print("⚠ Missing arguments.")
        print("Usage: python animation.py derivative_supplier Ix Iy Iz")
        print("Using default values: derivative_supplier=BC, Ix=1.0, Iy=1.0, Iz=1.0")
        main()