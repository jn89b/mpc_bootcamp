import numpy as np
import matplotlib.pyplot as plt

import casadi as ca

from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.close_loop import CloseLoopSim
from optitraj.utils.report import Report
from optitraj.utils.data_container import MPCParams

from typing import List, Tuple, Dict


class ToyCar(CasadiModel):
    def __init__(self) -> None:
        super().__init__()
        self.dt_val: float = 0.1
        self.define_states()
        self.define_controls()
        self.define_state_space()

    def define_states(self) -> None:
        # positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.psi_f = ca.MX.sym('psi_f')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.psi_f
        )

        self.n_states = self.states.size()[0]

    def define_controls(self) -> None:
        self.u_vel = ca.MX.sym('u_vel')
        self.u_psi = ca.MX.sym('u_psi')

        self.controls = ca.vertcat(
            self.u_vel,
            self.u_psi
        )

        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        self.x_fdot = self.u_vel * ca.cos(self.psi_f)
        self.y_fdot = self.u_vel * ca.sin(self.psi_f)
        self.psi_fdot = self.u_psi

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.psi_fdot
        )

        name = 'dynamics'
        self.function = ca.Function(name,
                                    [self.states, self.controls],
                                    [self.z_dot])


class CarOptimalControl(OptimalControlProblem):
    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel) -> None:
        super().__init__(mpc_params,
                         casadi_model)

    def compute_dynamics_cost(self) -> ca.MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        # initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R

        x_final = self.P[self.casadi_model.n_states:]

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                + (states - x_final).T @ Q @ (states - x_final) \
                + controls.T @ R @ controls

        return cost

    def compute_total_cost(self) -> ca.MX:
        cost = self.compute_dynamics_cost()
        return cost


toycar = ToyCar()
state_limits_dict: dict = {
    'x_f':
        {'min': -1000, 'max': 1000},
    'y_f':
        {'min': -1000, 'max': 1000},
    'psi_f':
        {'min': -np.pi, 'max': np.pi}
}
control_limits_dict: dict = {
    'u_vel':
        {'min': 0, 'max': 10},
    'u_psi':
        {'min': -np.deg2rad(20), 'max': np.deg2rad(20)},
}
toycar.set_control_limits(control_limits_dict)
toycar.set_state_limits(state_limits_dict)

Q: np.diag = np.diag([1, 1, 0])
R: np.diag = np.diag([1, 1])
# we will now slot the MPC weights into the MPCParams class
mpc_params: MPCParams = MPCParams(Q=Q, R=R, N=15, dt=0.1)
car_opt_control: CarOptimalControl = CarOptimalControl(
    mpc_params=mpc_params, casadi_model=toycar
)

# Now we set our initial conditions
x0: np.array = np.array([-10,  # x
                         5,  # y
                         np.deg2rad(10)])

xF: np.array = np.array(
    [100,
     -100,
     np.deg2rad(45)]
)
u_0: np.array = np.array([0, 0])


def custom_stop_criteria(state: np.ndarray,
                         final_state: np.ndarray) -> bool:
    distance = np.linalg.norm(state[0:2] - final_state[0:2])
    if distance < 5.0:
        return True


closed_loop_sim: CloseLoopSim = CloseLoopSim(
    optimizer=car_opt_control, x_init=x0, x_final=xF, u0=u_0,
    N=500, log_data=True, stop_criteria=custom_stop_criteria)

closed_loop_sim.run()
report: Report = closed_loop_sim.report
states: Dict = report.current_state
controls: Dict = report.current_control
# we will now plot the trajectory
# plot a 3D trajectory
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(states['x_f'], states['y_f'])
ax.scatter(xF[0], xF[1], c='r', label='Goal')


# Plot the velocity and the heading as well as the controls
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.rad2deg(states['psi_f']), label='actual psi')
ax[0].plot(np.rad2deg(controls['u_psi']), label='cmd psi')
ax[1].plot(controls['u_vel'], label='velocity control')
for a in ax:
    a.legend()
plt.show()
