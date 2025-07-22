import os
import sys

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

from .robot_arm_ik import G1_29_ArmIK
from .weighted_moving_filter import WeightedMovingFilter

# import numpy as np

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


class G1_29_ArmIK_NoWrists(G1_29_ArmIK):  # noqa: N801
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
            # Add wrist joints
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
            "right_wrist_yaw_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_elbow_joint"),
                pin.SE3(np.eye(3), np.array([0.35, -0.075, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_elbow_joint"),
                pin.SE3(np.eye(3), np.array([0.35, 0.075, 0]).T),
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
        # self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            +
            # self.rotation_cost +
            0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
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

        self.nq = self.reduced_robot.model.nq
        self.nv = self.reduced_robot.model.nv

        self.init_data = np.zeros(self.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.nq)
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
            self.init_data = current_lr_arm_motor_q[-self.nq :]

        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            _ = self.opti.solve()
            sol_q = self.opti.value(self.var_q)

            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if collision_check and self.check_self_collision(sol_q):
                print("Self-collision detected. Rejecting solution.")
                return self.init_data, np.zeros(self.nv)

            v = (
                current_lr_arm_motor_dq[-self.nv :] * 0.0
                if current_lr_arm_motor_dq is not None
                else (sol_q - self.init_data) * 0.0
            )
            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.nv))

            # Apply external forces to 4 joints instead of 7
            external_force_L = pin.Force(EE_efrc_L)
            external_force_R = pin.Force(EE_efrc_R)
            J_L = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, sol_q, self.L_hand_id)
            J_R = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, sol_q, self.R_hand_id)
            tau_ext_L = J_L.T @ external_force_L.vector
            tau_ext_R = J_R.T @ external_force_R.vector

            tau_ext = np.concatenate((tau_ext_L[:4], tau_ext_R[-4:]))[: self.nv]
            sol_tauff += tau_ext

            if self.Visualization:
                self.vis.display(sol_q)

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info. {e}")
            sol_q = self.opti.debug.value(self.var_q)

            if sol_q.shape[0] == self.nq:
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            v = (
                current_lr_arm_motor_dq[-self.nv :] * 0.0
                if current_lr_arm_motor_dq is not None
                else (sol_q - self.init_data) * 0.0
            )
            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.nv))

            print(
                "sol_q: %s\nmotorstate: %s\nleft_pose: \n%s\nright_pose: \n%s",
                sol_q,
                current_lr_arm_motor_q,
                left_wrist,
                right_wrist,
            )
            if self.Visualization:
                self.vis.display(sol_q)

            if current_lr_arm_motor_q is not None:
                return current_lr_arm_motor_q[-self.nq :], np.zeros(self.nv)
            return np.zeros(self.nq), np.zeros(self.nv)
