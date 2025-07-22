import os
import sys

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import SE3, Quaternion
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

from .weighted_moving_filter import WeightedMovingFilter

# import numpy as np

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


class G1_29_ArmIK:  # noqa: N801
    def __init__(self, Unit_Test=False, Visualization=False, robot_config=None):  # noqa: N803
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                robot_config["ASSET_FILE"], robot_config["ASSET_ROOT"]
            )
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                robot_config["ASSET_FILE"], robot_config["ASSET_ROOT"]
            )  # for testing

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.15, -0.075, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.15, 0.075, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # Creating model and data for collision check
        self.geom_model = pin.buildGeomFromUrdf(
            self.reduced_robot.model,
            robot_config["ASSET_FILE"],
            pin.GeometryType.COLLISION,
            robot_config["ASSET_ROOT"],
        )

        # Creating collision pairs and filering out neighboring links
        self.geom_model.addAllCollisionPairs()
        # Get the kinematic adjacency from the model
        adjacent_pairs = {(self.reduced_robot.model.parents[i], i) for i in range(1, self.reduced_robot.model.njoints)}
        # Filter out neighboring links
        filtered_pairs = []
        for cp in self.geom_model.collisionPairs:
            link1 = self.geom_model.geometryObjects[cp.first].parentJoint
            link2 = self.geom_model.geometryObjects[cp.second].parentJoint
            if (link1, link2) not in adjacent_pairs and (link2, link1) not in adjacent_pairs:
                filtered_pairs.append(cp)
        self.geom_model.collisionPairs[:] = filtered_pairs
        self.data = self.reduced_robot.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        print("num collision pairs - initial:", len(self.geom_model.collisionPairs))
        print(f"Number of geometry objects: {len(self.reduced_robot.collision_model.geometryObjects)}")

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        print(self.cq.shape)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit
            )
        )
        self.opti.minimize(
            50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        # Attributes for interpolation
        self.current_L_tf = None
        self.current_R_tf = None
        self.current_L_orientation = None
        self.current_R_orientation = None
        self.speed_factor = 0.02  # You can adjust this as needed

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length=0.15, axis_width=10)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def check_self_collision(self, q):
        """Check for self-collisions in the given configuration."""
        # pin.updateGeometryPlacements(self.reduced_robot.model,
        # self.reduced_robot.data,
        # self.reduced_robot.collision_model,
        # self.collision_data, q)
        collision_detected = False

        # Compute all the collisions
        pin.computeCollisions(self.reduced_robot.model, self.data, self.geom_model, self.geom_data, q, False)

        # print("Start checking collision")
        # Print the status of collision for all collision pairs
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            cp = self.geom_model.collisionPairs[k]
            if cr.isCollision():
                collision_detected = True
                print(
                    "Collision detected between %s: %s and %s: %s",
                    cp.first,
                    self.geom_model.geometryObjects[cp.first].name,
                    cp.second,
                    self.geom_model.geometryObjects[cp.second].name,
                )

        return collision_detected

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        EE_efrc_L=None,  # noqa: N803
        EE_efrc_R=None,  # noqa: N803
        collision_check=True,
    ):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            _ = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # Check for self-collisions
            if collision_check & self.check_self_collision(sol_q):
                print("Self-collision detected. Rejecting solution.")
                return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv)
            )

            # Apply External Force to the EE
            external_force_L = pin.Force(EE_efrc_L)
            external_force_R = pin.Force(EE_efrc_R)
            J_L = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, sol_q, self.L_hand_id)
            J_R = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, sol_q, self.R_hand_id)
            tau_ext_L = J_L.T @ external_force_L.vector
            tau_ext_R = J_R.T @ external_force_R.vector

            tau_ext = np.concatenate((tau_ext_L[:7], tau_ext_R[-7:]))
            sol_tauff += tau_ext

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv)
            )

            print(
                "sol_q: %s \nmotorstate: \n%s \nleft_pose: \n%s \nright_pose: \n%s",
                sol_q,
                current_lr_arm_motor_q,
                left_wrist,
                right_wrist,
            )
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def set_initial_poses(self, L_tf, R_tf, L_orientation, R_orientation):  # noqa: N803
        """Set the initial poses for interpolation."""
        self.current_L_tf = L_tf.copy()
        self.current_R_tf = R_tf.copy()
        self.current_L_orientation = pin.Quaternion(L_orientation)
        self.current_R_orientation = pin.Quaternion(R_orientation)

    def get_q_tau(self, L_tf_target, R_tf_target, EE_efrc_L, EE_efrc_R):  # noqa: N803
        """Interpolate and solve IK for the given target poses."""
        # Interpolate positions and orientations
        self.current_L_tf = (1 - self.speed_factor) * self.current_L_tf + self.speed_factor * L_tf_target.translation
        self.current_R_tf = (1 - self.speed_factor) * self.current_R_tf + self.speed_factor * R_tf_target.translation

        self.current_L_orientation = self.current_L_orientation.slerp(
            self.speed_factor, pin.Quaternion(L_tf_target.rotation)
        )
        self.current_R_orientation = self.current_R_orientation.slerp(
            self.speed_factor, pin.Quaternion(R_tf_target.rotation)
        )

        L_tf_interpolated = pin.SE3(self.current_L_orientation.toRotationMatrix(), self.current_L_tf)
        R_tf_interpolated = pin.SE3(self.current_R_orientation.toRotationMatrix(), self.current_R_tf)

        # Solve IK
        sol_q, sol_tauff = self.solve_ik(
            L_tf_interpolated.homogeneous,
            R_tf_interpolated.homogeneous,
            EE_efrc_L=EE_efrc_L,
            EE_efrc_R=EE_efrc_R,
            collision_check=False,
        )
        return sol_q, sol_tauff

    def get_target_waypoint_error(self, current_q, L_tf_target, R_tf_target):  # noqa: N803
        current_q = np.asarray(current_q).reshape(-1)
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, current_q)

        L_current_pose = self.data.oMf[self.L_hand_id]
        R_current_pose = self.data.oMf[self.R_hand_id]
        L_current_pos = np.array(L_current_pose.translation).flatten()
        R_current_pos = np.array(R_current_pose.translation).flatten()

        L_target_pos = np.array(L_tf_target.translation).flatten()
        R_target_pos = np.array(R_tf_target.translation).flatten()

        error_L = L_current_pos - L_target_pos
        error_R = R_current_pos - R_target_pos

        return np.linalg.norm(np.concatenate((error_L, error_R)))

    def get_end_effector_poses(self, q):
        """
        Returns the current SE3 poses of the left and right end-effectors ("L_ee" and "R_ee")
        for a given joint configuration q.

        Args:
            q (np.ndarray): Joint configuration (shape: nq,)

        Returns:
            tuple: (L_ee_pose, R_ee_pose) where each is a pinocchio.SE3 object
        """
        q = np.asarray(q).reshape(-1)
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, q)

        L_ee_pose = self.data.oMf[self.L_hand_id]
        R_ee_pose = self.data.oMf[self.R_hand_id]

        return L_ee_pose, R_ee_pose

    def generate_waypoints_SE(self, start_pose: SE3, end_pose: SE3, num_points: int):  # noqa: N802
        """
        Generate SE3 waypoints interpolating from start to end pose.

        Args:
            start_pose (pin.SE3): Starting pose
            end_pose (pin.SE3): Ending pose
            num_points (int): Number of waypoints to generate

        Returns:
            list of pin.SE3: Interpolated SE3 poses
        """
        # Extract positions
        start_pos = start_pose.translation
        end_pos = end_pose.translation

        # Extract orientations as pin.Quaternions
        start_quat = Quaternion(start_pose.rotation)
        end_quat = Quaternion(end_pose.rotation)

        # Generate interpolation parameters
        alphas = np.linspace(0, 1, num_points)

        waypoints = []
        for alpha in alphas:
            # Interpolate position
            interp_pos = (1 - alpha) * start_pos + alpha * end_pos

            # Interpolate orientation using SLERP
            interp_quat = start_quat.slerp(alpha, end_quat)
            interp_rot = interp_quat.toRotationMatrix()

            # Create waypoint
            waypoint = SE3(interp_rot, interp_pos)
            waypoints.append(waypoint)

        return waypoints
