import numpy as np
import matplotlib.pyplot as plt

import casadi as ca
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.close_loop import CloseLoopSim
from optitraj.utils.report import Report
from optitraj.utils.data_container import MPCParams

from typing import List, Tuple, Dict

class Car(CasadiModel):
    """
    """
    def __init__(self):
        
        super().__init__()
        self.dt_val = 0.1
        self.define_states()
        self.define_controls()
        self.define_state_space()
        
    def define_states(self) -> None:
        """define the states of your system"""
        # positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.psi_f = ca.MX.sym('psi_f')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.psi_f,
        )

        self.n_states = self.states.size()[0]  # is a column vector

    def define_controls(self) -> None:
        """controls for your system"""
        self.u_psi = ca.MX.sym('u_psi')
        self.v_cmd = ca.MX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        """define the state space of your system"""
        self.x_fdot = self.v_cmd * ca.cos(self.psi_f)
        self.y_fdot = self.v_cmd * ca.sin(self.psi_f)
        self.psi_fdot = self.u_psi 

        # Define the state space equations
        self.dynamics = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.psi_fdot
        )

        # Create a CasADi function for the dynamics
        self.function = ca.Function('dynamics', 
                                    [self.states, 
                                     self.controls], 
                                    [self.dynamics])
        

class CarMPC(OptimalControlProblem):
    def __init__(self, mpc_params:MPCParams,
                 casadi_model:CasadiModel) -> None:
        super().__init__(mpc_params, casadi_model)
        
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


def main() -> None:
    """
    """
    car: Car = Car()
    control_limits: Dict = {
        'u_psi': {'min': -np.pi/4, 'max': np.pi/4},
        'v_cmd': {'min': 0.5, 'max': 10}
    }
    state_limits: Dict = {
        'x_f': {'min': -1000, 'max': 1000},
        'y_f': {'min': -1000, 'max': 1000},
        'psi_f': {'min': -np.pi, 'max': np.pi}
    }
    car.set_control_limits(control_limits)
    car.set_state_limits(state_limits)
    
    # now we will set the MPC weights for the plane
    # 0 means we don't care about the specific state variable 1 means we care about it
    Q: np.diag = np.diag([1, 1, 0.8])
    R: np.diag = np.diag([0.1, 0.1])

    # we will now slot the MPC weights into the MPCParams class
    mpc_params: MPCParams = MPCParams(Q=Q, R=R, N=15, dt=0.1)
    
    # formulate your optimal control problem
    plane_opt_control: CarMPC = CarMPC(
        mpc_params=mpc_params, casadi_model=car)
    
    # now set your initial conditions for this case its the plane
    x0: np.array = np.array([-25, 5, np.deg2rad(45)])
    xF: np.array = np.array([100, -100, np.deg2rad(0)])
    u_0: np.array = np.array([0, control_limits['v_cmd']['min']])

    def custom_stop_criteria(state: np.ndarray,
                             final_state: np.ndarray) -> bool:
        distance = np.linalg.norm(state[0:2] - final_state[0:2])
        if distance < 0.01:
            return True

    # we can now begin our simulation
    closed_loop_sim: CloseLoopSim = CloseLoopSim(
        optimizer=plane_opt_control, 
        x_init=x0, x_final=xF, u0=u_0,
        N=1000, log_data=True, 
        stop_criteria=custom_stop_criteria, 
        print_every=5)
    
    closed_loop_sim.run()
    print("simulation finished")
    report: Report = closed_loop_sim.report
    
    states: Dict = report.current_state
    # we will now plot the trajectory
    # plot a 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(states['x_f'], states['y_f'])
    ax.scatter(xF[0], xF[1], c='r', label='Goal')
    
    # plot the controls
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.rad2deg(report.current_state['psi_f']), label='psi_f')
    ax[0].plot(np.rad2deg(report.current_control['u_psi']), label='u_psi')
    ax[0].set_title('u_psi')
    ax[1].plot(report.current_control['v_cmd'])
    ax[1].set_title('v_cmd')
    for a in ax:
        a.legend()
        a.grid()
        
    plt.tight_layout()
    
    plt.show()


    
if __name__ == "__main__":
    main()