import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

class CanonicalSystem(object):
    """
    A class to represent a canonical system with a state vector and a time step.
    The state vector is initialized to zero and can be updated with a new state.
    """

    def __init__(self, ax=1.0, dt=0.01, pattern="exp", tau_y=0.6, tau=1.0, T=1.5, av=4.0, v_max=2.0):
        """
        Initializes the CanonicalSystem with parameters for the system dynamics.
        :param ax: Decay rate for the state.
        :param dt: Time step for the simulation.
        :param pattern: Type of step function, either "exp" for exponential or "discrete" for discrete.
        :param tau_y: Time constant for the state update.
        :param tau: Time constant for the system dynamics.
        :param T: Total time for the simulation.
        :param av: Coefficient for velocity dynamics.
        :param v_max: Maximum velocity.
        """
        self.av = av
        self.v_max = v_max
        self.s = 1.0
        self.ax = ax
        self.time_step = 0.0
        self.dt = dt
        self.pattern = pattern
        self.tau_y = tau_y
        self.tau = tau
        self.timesteps = int(T / dt)
        self.run_time = T
        self.v = 1.0
        self.dv = 0.0
        self.ds = 0.0
        self.v_max = 2.0
        if self.pattern == "exp":
            self.step = self.step_exp
        else:
            self.step = self.step_discrete

    def reset(self):
        if self.pattern == "exp":
            self.s = 0.0
            self.v = 1.0
        else:
            self.s = 1.0
            self.v = 1.0

    def step_exp(self):
        """
        Make a step in the system.
        """
        self.s += self.ds * self.dt * self.tau

        if self.s < 1.0:
            self.ds = 1 / self.tau_y
        else:
            self.ds = 0.0
        return self.s
    def step_v(self):
        self.v += self.dv * self.dt

        self.dv = - self.av * self.v * ( 1 - self.v / self.v_max)
        return self.v


    def step_discrete(self):
        self.s += (-self.ax * self.s) * self.tau * self.dt
        return self.s

    def rollout(self):
        self.s_track = np.zeros(self.timesteps)
        self.reset()

        for t in range(self.timesteps):
            self.s_track[t] = self.s
            self.step()
        return self.s_track.reshape(-1,1)
    
    def rollout_v(self):
        self.v_track = np.zeros(self.timesteps)
        self.reset()

        for t in range(self.timesteps):
            self.v_track[t] = self.step_v()
        return self.v_track.reshape(-1,1)

    def __test(self):
        """
        A simple test to visualize the system's behavior.
        """
        s_track = self.rollout()
        v_track = self.rollout_v()
        plt.plot(np.arange(self.timesteps) * self.dt, s_track)
        plt.plot(np.arange(self.timesteps) * self.dt, v_track)
        plt.legend(['s(t)', 'v(t)'])
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.title('Canonical System State Over Time')
        plt.grid()
        plt.show()
# CanonicalSystem()._CanonicalSystem__test()  # Run the test method to visualize the system's behavior.
# CanonicalSystem(pattern="discrete")._CanonicalSystem__test()

class DMPDG(object):
    """
    A class to represent a DMPDG (Dynamic Multi-Point Decision Graph) object.
    This class is a placeholder for future implementation.
    """

    def __init__(self, 
                 n_dmps=1, 
                 n_bfs=20, 
                 dt=0.01, 
                 ay=25.0, 
                 ax=1, 
                 tau_y=0.6, 
                 tau=1.0,
                 v_max=2.0, 
                 ag=6.0, 
                 av=4.0,
                 y0=1.0,
                 g=0.5,
                 pattern="exp",
                 dmp_type="vanilla",
                 T=1.5,
                 dim=None,
                 ):
        """
        Initialize the DMPDG object with optional arguments.
        """
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.ax = ax
        self.ay = ay
        self.by = self.ay / 4
        self.tau_y = tau_y
        self.tau = tau
        self.pattern = pattern
        self.cs = CanonicalSystem(ax=self.ax, 
                                  dt=self.dt, 
                                  tau_y=self.tau_y, 
                                  tau=self.tau, 
                                  pattern=self.pattern,
                                  T=T)

        # self.set_c()
        self.imitate = False
        self.gen_centers()
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        # self.set_h()

        self.v_max = v_max
        self.ag = ag
        self.av = av
        self.y0 = y0
        self.goal = g
        self.f = np.zeros(self.cs.timesteps)
        self.dmp_type = dmp_type

        if self.dmp_type == "vanilla":
            self.step = self.step_vanilla
        elif self.dmp_type == "delayed":
            self.step = self.step_dg
        self.dim = dim if dim is not None else n_dmps

    def __str__(self):
        """
        Return a string representation of the DMPDG object.
        """
        return "DMPDG Object"
    
    def __repr__(self):
        """
        Return a detailed string representation of the DMPDG object.
        """
        return f"DMPDG(n_dmps={self.n_dmps}, n_bfs={self.n_bfs}, dt={self.dt})"
    
    def set_h(self):
        self.h = np.zeros(self.n_bfs)
        for i in range(self.n_bfs - 1):
            self.h[i] = 1 / (self.c[i] - self.c[i + 1]) ** 2
    
    def set_c(self):
        self.c = np.zeros(self.n_bfs)
        for i in range(1, self.n_bfs + 1):
            self.c[i - 1] = np.exp(-self.ax *((i -1)/ (self.n_bfs -1)))
    
    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            c[n] = np.exp(-self.cs.ax * des_c[n])

        self.c = c.copy()


    def psi(self, s):
        return np.exp(-self.h * ((s - self.c) ** 2))
    
    def get_f_target(self, y_des: np.ndarray, dy_des: np.ndarray, ddy_des: np.ndarray):
        """
        Calculate the target force based on the desired trajectory.
        """
        # y_des, dy_des, ddy_des = self.imitate_trajectory(y_des)

        # f_target = np.zeros((self.n_dmps, self.n_bfs))
        if self.dmp_type == "delayed":
            v = self.cs.rollout_v()
            goal_d = self.rollout_goal_d()
            f_target = ddy_des - self.ay * (self.by * (goal_d - y_des) - dy_des) #/ v
            f_target = f_target.reshape(-1, 1)  / v
        elif self.dmp_type == "vanilla":
            f_target = ddy_des - self.ay*(self.by*(self.goal - y_des) - dy_des)
        return f_target#.reshape(-1, 1)

    def imitate_trajectory(self, y_des: np.ndarray):
        self.imitate = True
    
        self.y0 = y_des[0].copy()

        self.goal = self.goal_d= y_des[-1].copy()

        x = np.linspace([0], self.cs.run_time, y_des.shape[0])
        x_new = np.linspace(0, self.cs.run_time, self.cs.timesteps)

        y_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))
        dy_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))
        ddy_des_smooth = np.empty((self.cs.timesteps, y_des.shape[1]))

        for i in range(y_des.shape[1]):
            spl = make_smoothing_spline(x.flatten(), y_des[:, i])
            y_des_smooth[:, i] = spl(x_new).copy()
            dy_des_smooth[:, i] = np.gradient(y_des_smooth[:, i]) / self.dt
            ddy_des_smooth[:, i] = np.gradient(dy_des_smooth[:, i]) / self.dt

        self.f_target = self.get_f_target(y_des_smooth, dy_des_smooth, ddy_des_smooth)
        print(f"f_target shape: {self.f_target.shape}")
        self.w = self.gen_weights(self.f_target)
        self.f = self.forcing_term()
        self.reset()
        self.cs.reset()
        return y_des_smooth, dy_des_smooth, ddy_des_smooth
    
    # def gen_weights(self, f_target):
    #     s = self.cs.rollout()
    #     Psi = self.psi(s)
    #     print(Psi.shape, f_target.shape)
    #     weights =  (np.linalg.pinv(Psi) @ f_target) / Psi.sum(axis=0, keepdims=True).T
    #     return weights
    
    def get_phi_inv(self):
        # Normalize basis functions and multiply by s (phase)
        s = self.cs.rollout().reshape(-1, 1)  # [K, 1]
        psi_track = self.psi(s)
        basis_matrix = psi_track / np.sum(psi_track, axis=1, keepdims=True)  # normalized
        Phi = basis_matrix * s # [K, N]
        return np.linalg.pinv(Phi)

    # def gen_weights(self, f_target):
    #     # Normalize basis functions and multiply by s (phase)
    #     s = self.cs.rollout().reshape(-1, 1)  # [K, 1]
    #     psi_track = self.psi(s)
    #     basis_matrix = psi_track / np.sum(psi_track, axis=1, keepdims=True)  # normalized
    #     Phi = basis_matrix * s # [K, N]
    #     # weights = np.linalg.pinv(Phi) @ dmp.f_target  # [N, 1]
    #     weights = np.linalg.pinv(Phi) @ f_target  # [N, 1]
    #     return weights
    
    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.psi(x_track)
        # efficiently calculate BF weights using weighted linear regression
        weights = np.zeros((self.n_bfs, self.dim))
        # self.w_ =  torch.tensor((np.linalg.inv(psi_track.T @ psi_track) @ psi_track.T )@ f_target)
        # spatial scaling term
        k = self.goal - self.y0
        for i in range(self.dim):
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, None, b] * f_target[:, None, i])
                denom = np.sum(x_track ** 2 * psi_track[:, None, b])
                weights[b, i] = numer / denom
                if abs(k[i]) > 1e-5:
                    # print(i)
                    weights[b, i] /= k[i]
        
        return weights

    def reset(self):
        self.z = np.zeros(self.dim)  # reset z to zero
        if isinstance(self.y0, np.ndarray):
            self.y = self.y0.copy()
        else: 
            self.y = self.y0
        self.dy = np.zeros(self.dim)  # reset dy to zero
        self.dz = np.zeros(self.dim)  # reset dz to zero
        self.ddy = np.zeros(self.dim)  # reset ddy to zero
        if isinstance(self.goal, np.ndarray):
            self.goal_d = self.y0.copy()
        else:
            self.goal_d = self.y0
        self.dgoal_d = np.zeros(self.dim)  # reset dgoal_d to zero
        self.v = np.ones(self.dim)  # reset v to ones
        self.dv = np.zeros(self.dim)  # reset dv to zero
        self.cs.reset()
        
    def step_vanilla(self, i=0):
        if self.imitate:
            x = self.cs.step()
            psi = self.psi(x)
            f = np.dot(psi, self.w) * x * (self.goal - self.y0) / (np.sum(psi))
            # print(f)
            # f = f[0]
            self.f_[i] = f
        else:
            f = 0
        self.dz = (self.ay * (self.by * (self.goal - self.y) - self.z) + f) / self.tau_y

        self.dy = self.z / self.tau_y

        self.z = self.z + self.dz * self.dt
        self.y = self.y + self.dy * self.dt
        
        return self.y, self.dy, self.dz / self.tau_y

    def step_dg(self, i=0):
        if self.imitate:
            x = self.cs.step()
            psi = self.psi(x)
            f = np.dot(psi, self.w) * x * (self.goal - self.y0)
            sum_psi = np.sum(psi)
            if np.abs(sum_psi) > 1e-6:
                f /= sum_psi
            print(f)
            # f = f[0]
            # print(i, x, f)
            self.f_[i] = f
        else:
            f = 0

        self.v += self.dv * self.dt
        self.goal_d = self.goal_d + self.dgoal_d * self.dt
        self.z = self.z + self.dz * self.dt
        self.y = self.y + self.dy * self.dt
        
        self.dz = (self.ay * (self.by * (self.goal_d - self.y) - self.z) + self.v * f) / self.tau_y

        self.dy = self.z / self.tau_y
        self.dgoal_d = self.ag * (self.goal - self.goal_d) / self.tau_y
        self.dv = -self.av * self.v * (1 - (self.v/ self.v_max))
        
        
        return self.y, self.dy, self.dz / self.tau_y

    def forcing_term(self):
        Psi = self.psi(self.cs.rollout())
        sum_psi = (Psi.sum(axis=1, keepdims=True))
        sum_psi[np.where(abs(sum_psi) < 1e-6)[0]] = 1.0
        return ((Psi @ self.w) * self.cs.rollout()) #/ sum_psi

    def rollout_goal_d(self):
        goal_d = self.y0
        goal_d_rollout = np.zeros(self.cs.timesteps)
        dgoal_d = 0
        for i in range(self.cs.timesteps):
            goal_d  += dgoal_d * self.dt
            goal_d_rollout[i] = goal_d
            dgoal_d = self.ag * (self.goal - goal_d)
        return goal_d_rollout.reshape(-1, 1)
    
    def rollout(self):
        """
        Generate a rollout of the DMPDG system based on a desired trajectory.
        Args:
            y_des (np.ndarray): Desired trajectory to follow.   
        """
        y_rollout = np.zeros((self.cs.timesteps, self.dim))
        dy_rollout = np.zeros((self.cs.timesteps, self.dim))
        ddy_rollout = np.zeros((self.cs.timesteps, self.dim))
        self.f_ = np.empty_like(self.f)
        self.reset()
        if self.dmp_type == "delayed":
            self.goal_d_rollout = np.zeros(self.cs.timesteps)
            self.d_goal_d_rollout = np.zeros(self.cs.timesteps)
        for t in range(self.cs.timesteps):
            y, dy, ddy = self.step(i=t)
            y_rollout[t] = y
            dy_rollout[t] = dy
            ddy_rollout[t] = ddy
            if self.dmp_type == "delayed":
                self.goal_d_rollout[t] = self.goal_d
                self.d_goal_d_rollout[t] = self.dgoal_d
        return y_rollout, dy_rollout, ddy_rollout
    
    def __test(self):
        """
        A simple test to visualize the DMPDG system's behavior.
        """
        y, dy, ddy = self.rollout()
        fig , ax = plt.subplots(1, 3, figsize=(15, 5))
        t = np.arange(len(y)) * self.dt
        ax[0].plot(np.arange(len(y)) * self.dt, y, label='y')
        ax[0].set_title('y')
        ax[1].plot(np.arange(len(dy)) * self.dt, dy, label='dy', color='orange')
        ax[1].set_title('dy')
        ax[2].plot(np.arange(len(ddy)) * self.dt, ddy, label='ddy', color='green')
        ax[2].set_title('ddy')
        if self.dmp_type == "delayed":
            ax[0].plot(t, self.goal_d_rollout, '--')
            ax[1].plot(t, self.d_goal_d_rollout, '--')
        for a in ax:
            a.grid()
            a.legend()
            a.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t, self.psi(self.cs.rollout()), c="blue", linewidth=0.5)
        ax.plot(t, self.cs.rollout(), label='s(t)', linestyle='--')
        ax.plot(t, self.cs.rollout_v(), label='v(t)', linestyle=':')
        ax.set_title('Basis Functions')
        ax.set_xlabel('s')
        ax.set_ylabel('Activation')
        ax.grid()
        ax.legend()
        plt.tight_layout()
            
# dmp = DMPDG(n_dmps=1, n_bfs=20, dt=0.01, tau_y=.60, y0=1.0, g=0.5)
# DMPDG(pattern="discrete", dmp_type="delayed", T=1.0, dt=0.01, tau_y=.60, tau=1.0)._DMPDG__test()  # Run the test method to visualize the DMPDG system's behavior.

def make_arm_trajectory(points):
    traj0 = np.vstack([np.linspace(points[0], points[1], 50),
         np.linspace(points[1], points[2], 50)])
    new_traj0 = np.empty_like(traj0)
    x = np.linspace(0, 1, traj0.shape[0])
    for i in range(3):
          spl = make_smoothing_spline(x, traj0[:, i])
          new_traj0[:, i] = spl(np.linspace(0, 1, traj0.shape[0]))
    return new_traj0