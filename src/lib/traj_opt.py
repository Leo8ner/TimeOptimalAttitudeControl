import casadi as ca
import numpy as np

class Optimizer:
    def __init__(self, N_X, N_U, T_0, N_steps, f, F, margin, Q, R, D, M, X_i, X_f, 
                 X_max, U_max, U_rate_max, lb_U, ub_U, lb_X, ub_X, lb_dt, ub_dt):
        
        self.N_X        = int(N_X)                                         # Number of states----------------- , -
        self.N_U        = int(N_U)                                         # Number of controls----------------, -
        self.T_0        = T_0                                              # Initial time horizon--------------, s
        self.N_steps    = N_steps                                          # Number of time steps--------------, -
        self.f          = f                                                # Continuous dynamics function
        self.F          = F                                                # Discrete dynamics function
        self.Q          = Q                                                # State cost matrix-----------------, -
        self.R          = R                                                # Control cost matrix-------------- , -
        self.D          = D                                                # Control rate cost matrix----------, -
        self.M          = M                                                # Time horizon cost matrix-, -
        self.X_i        = X_i                                              # Initial state---------------------, [m, m, rad, m/s, m/s, rad/s, kg]
        self.X_f        = X_f                                              # Final state-----------------------, [m, m, rad, m/s, m/s, rad/s, kg]
        self.U_rate_max = U_rate_max                                       # Maximum control rate
        lb_X            = np.array(lb_X)
        ub_X            = np.array(ub_X)
        lb_U            = np.array(lb_U)
        ub_U            = np.array(ub_U)
        margin          = np.array(margin)
        self.lb_X       = np.tile(lb_X[:, np.newaxis], (1, N_steps + 1))   # State lower bounds----------------, [m, m, rad, m/s, m/s, rad/s, kg].T * (N_steps + 1)
        self.ub_X       = np.tile(ub_X[:, np.newaxis], (1, N_steps + 1))   # State upper bounds----------------, [m, m, rad, m/s, m/s, rad/s, kg].T * (N_steps + 1)
        self.lb_U       = np.tile(lb_U[:, np.newaxis], (1, N_steps))       # Control lower bounds--------------, [-, -, -].T * N_steps
        self.ub_U       = np.tile(ub_U[:, np.newaxis], (1, N_steps))       # Control upper bounds--------------, [-, -, -].T * N_steps
        self.lb_dt      = lb_dt                                            # Time horizon lower bound----------, s
        self.ub_dt      = ub_dt                                            # Time horizon upper bound----------, s
        self.margin     = ca.repmat(margin, N_X, 1)                        # Margin for the next state
        self.opti       = ca.Opti()                                        # Optimization problem
        X_max_mat       = ca.repmat(X_max, 1, N_steps + 1)                 # State normalization matrix--------, [m, m, rad, m/s, m/s, rad/s, kg].T * (N_steps + 1)
        self.X          = X_max_mat * self.opti.variable(N_X, N_steps + 1) # States----------------------------, [m, m, rad, m/s, m/s, rad/s, kg]
        U_max_mat       = ca.repmat(U_max, 1, N_steps)                     # Control normalization matrix------, [-, -, -].T * N_steps
        self.U          = U_max_mat * self.opti.variable(N_U, N_steps)     # Controls--------------------------, [-, -, -]
        self.T          = self.opti.variable()                             # Initializing time horizon---------, s
        self.dt         = self.T / N_steps                                 # Time step-------------------------, s
        self.opti.set_initial(self.T, T_0)                                 # Set time horizon initial guess
        self.X_guess    = ca.DM.zeros(N_X, N_steps + 1)                    # Initializing states trajectory guess
        self.init_guess = True                                             # Flag for parabolic initial guess
        self.inf_pr_max = float(margin)                                    # Maximum primal infeasibility
        self.inf_du_max = 1e10                                             # Maximum dual infeasibility

    def setup_problem(self):

        # State and control bounds
        self.opti.subject_to(self.opti.bounded(self.lb_U, self.U, self.ub_U))    # Control constraints
        self.opti.subject_to(self.opti.bounded(self.lb_X, self.X, self.ub_X))    # State constraints
        self.opti.subject_to(self.opti.bounded(self.lb_dt, self.dt, self.ub_dt)) # Time horizon constraints

        # Initial parabolic guess for the state trajectory
        if self.init_guess:
            t_values = np.linspace(0, self.T_0, self.N_steps + 1)
            for i, t in enumerate(t_values):
                for j in range(self.N_X):
                    if j == 6:
                        a = self.X_f[j] / self.T_0**2
                        self.X_guess[j,i] = self.parabola_generator(t, self.T_0, self.X_i[j], a_add = a, b = 0)
                    elif self.X_i[j] == 0:
                        self.X_guess[j,i] = 0
                    elif j == 4:
                        self.X_guess[j,i] = self.parabola_generator(t, self.T_0, self.X_i[j], a_m = -1, b = 0)
                    else:
                        self.X_guess[j,i] = self.parabola_generator(t, self.T_0, self.X_i[j])
            # Initialize all state variables
            self.opti.set_initial(self.X, self.X_guess)

        U_max = self.dt * self.U_rate_max
        objective = 0
        for k in range(self.N_steps):
            if k < self.N_steps - 1:
                # Control rate constraints
                self.opti.subject_to(
                self.opti.bounded(self.U[:, k] - U_max - self.margin[1:] , self.U[:, k + 1], U_max + self.U[:, k] + self.margin[1:])
                )
                U_rate = (self.U[:, k + 1] - self.U[:, k]) / self.dt                     # Control rate
            X_kp1 = self.F(self.X[:, k], self.U[:, k], self.dt)                        # Next state
            self.opti.subject_to(self.opti.bounded(X_kp1 - self.margin, self.X[:, k + 1], X_kp1 + self.margin)) # Dynamics constraints RK4
            objective += ca.bilin(self.R, self.U[:, k])                    # Control cost
            objective += ca.bilin(self.D, U_rate)                     # Control rate cost
        objective += ca.bilin(self.Q, self.X[:, -1])                 # State cost    
        objective += self.T * self.M                    # Final state cost
        self.opti.minimize(objective)
    
    def parabola_generator(self, t, x_f, y_0, a_m = 1, b = None, a_add = 0):
        c = y_0
        a = c / x_f**2 * a_m + a_add
        if b == None:
            b = -2 * a * x_f
        else:
            b = b
        return a * t**2 + b * t + c
    
    def set_initial_guess(self, X_guess, U_guess):
        self.init_guess = False
        self.X_guess = ca.DM(X_guess)
        self.opti.set_initial(self.X, self.X_guess)
        self.opti.set_initial(self.U, U_guess)

    def set_initial_input(self, U_guess):
        self.opti.subject_to(self.U[:,0] == U_guess)
        
    def get_initial_guess(self):
        return np.array(self.X_guess)

    def set_final_state(self, slack = ca.MX.zeros(7), x = True):
        self.slack = slack
        self.x_f = x
        if x:
            self.opti.subject_to(self.opti.bounded(-slack[:-1], self.X[:-1, -1], slack[:-1]))    # State constraints
        else:
            self.opti.subject_to(self.opti.bounded(-slack[1:-1], self.X[1:-1, -1], slack[1:-1])) # State constraints
    
    def set_initial_state(self, x = True):
        if x:
            self.opti.subject_to(self.X[:, 0] == self.X_i)
        else:
            self.opti.subject_to(self.X[1:, 0] == self.X_i[1:])

    def solve(self):

        self.setup_problem()

        # Set solver options
        p_opts = {"expand": True}
        s_opts = {
            "print_level": 0,
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "warm_start_init_point": "yes",
            "max_iter": 150,
            "tol": 10e-6,
            "dual_inf_tol": 10e-6,
            "constr_viol_tol": 10e-6,
            "compl_inf_tol": 10e-6,
            # "acceptable_tol": self.margin,
            # "acceptable_constr_viol_tol": self.margin,
            # "acceptable_dual_inf_tol": self.margin,
            # "acceptable_compl_inf_tol": self.margin
        }
        self.opti.solver("ipopt", p_opts, s_opts)  # Choose IPOPT as the solver


        # Track the best feasible iterate
        self.best_feasible_X = None
        self.best_feasible_U = None
        self.best_feasible_T = None
        self.cur_inf_pr = None
        self.cur_inf_du = None

        def store_feasible_iter(iteration):
            """Stores the last feasible iterate if constraints are satisfied."""
            nonlocal self
            try:
                stats = self.opti.stats()["iterations"]
                inf_pr = stats["inf_pr"][-1]
                inf_du = stats["inf_du"][-1]
                if abs(self.inf_pr_max - inf_pr) / self.inf_pr_max < 0.1 and inf_du < self.inf_du_max:
                    self.inf_pr_max = inf_pr
                    self.inf_du_max = inf_du
                    self.best_feasible_X = self.opti.debug.value(self.X)
                    self.best_feasible_U = self.opti.debug.value(self.U)
                    self.best_feasible_T = self.opti.debug.value(self.T)
                    #print("New more optimal iterate.")
                elif inf_pr < self.inf_pr_max:
                    self.inf_du_max = inf_du
                    self.inf_pr_max = inf_pr
                    self.best_feasible_X = self.opti.debug.value(self.X)
                    self.best_feasible_U = self.opti.debug.value(self.U)
                    self.best_feasible_T = self.opti.debug.value(self.T)
                    #print("New more feasible iterate.")

            except Exception as e:
                print(f"Failed to store feasible iterate: {e}")
                pass  # Ignore exceptions from `opti.debug.value()`

        # Add callback to track feasible solutions
        self.opti.callback(store_feasible_iter)

        try:
            sol    = self.opti.solve()                  # Solve the optimization problem
            opt_X  = self.opti.value(self.X)            # Extract the optimal state trajectory
            opt_U  = self.opti.value(self.U)            # Extract the optimal control trajectory
            opt_T  = self.opti.value(self.T)            # Extract the optimal time horizon
            status = 0
            print(f"Optimal solution found in {self.opti.stats()["iter_count"]} iterations.")
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            if self.best_feasible_X is not None:
                opt_X = self.best_feasible_X
                opt_U = self.best_feasible_U
                opt_T = self.best_feasible_T
                status = 1
                print("Using best feasible iterate.")
            else:
                opt_X = self.opti.debug.value(self.X)
                opt_U = self.opti.debug.value(self.U)
                opt_T = self.opti.debug.value(self.T)
                status = -1
                print("No feasible iterate found.")

        # Compute state derivatives for each time step
        opt_X_dot = np.zeros((self.N_X, self.N_steps))
        for k in range(self.N_steps):
            opt_X_dot[:, k] = np.array(self.f(opt_X[:, k], opt_U[:, k])).flatten()

        return opt_X, opt_U, opt_X_dot, opt_T, status
