import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
from itertools import product

class DataGenerator:
    def __init__(self):
        pass

    def lorenz_system(self, state, t, sigma, rho, beta):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    def generate_lorenz_data(self, initials, t, sigma, rho, beta):
        solution = odeint(self.lorenz_system, initials, t, args=(sigma, rho, beta))
        x_trajectory, y_trajectory, z_trajectory = solution[:, 0], solution[:, 1], solution[:, 2]
        input_df = pd.DataFrame({'x': x_trajectory[10:-1], 'y': y_trajectory[10:-1], 'z': z_trajectory[10:-1]})
        output_df = pd.DataFrame({'x': x_trajectory[11:], 'y': y_trajectory[11:], 'z': z_trajectory[11:]})
        return input_df, output_df

    def generate_actuated_pendulum_data(self, damping_coefficient=0.11, length=10.0, mass=1.0, gravity=9.81, disc=30):
            def actuated_pendulum(state, t, torque):
                theta, omega = state
                g = gravity
                L = length
                m = mass
                b = damping_coefficient

                dtheta_dt = omega
                domega_dt = (-g / L) * np.sin(theta) + (torque / (m * L**2)) - b * omega

                return [dtheta_dt, domega_dt]

            theta_range = np.linspace(-np.pi/2, np.pi/2, num=disc)
            omega_range = np.linspace(-1, 1, num=disc)
            torque_range = np.linspace(-1, 1, num=disc)

            combinations = np.array(list(product(theta_range, omega_range, torque_range)))

            input_lst = []
            output_lst = []

            for row in combinations:
                sol = odeint(actuated_pendulum, [row[0], row[1]], [0, 0.1], args=(row[2],))
                input_lst.append(row)
                output_lst.append(sol[-1])

            inputs = np.array(input_lst)
            outputs = np.array(output_lst)

            return inputs, outputs

    def generate_double_pendulum_data(self, discretize):
        def double_pendulum(y, t):
            theta1, omega1, theta2, omega2 = y
            m1 = 1.0
            m2 = 1.0
            l1 = 1.0
            l2 = 1.0
            g = 9.81

            theta1_dot = omega1
            theta2_dot = omega2

            delta = theta2 - theta1
            den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) * np.cos(delta)
            omega1_dot = ((m2 * l1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                        m2 * g * np.sin(theta2) * np.cos(delta) +
                        m2 * l2 * omega2 * omega2 * np.sin(delta) -
                        (m1 + m2) * g * np.sin(theta1)) / den1)

            den2 = (l2 / l1) * den1
            omega2_dot = ((-m2 * l2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                        (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                        (m1 + m2) * l1 * omega1 * omega1 * np.sin(delta) -
                        (m1 + m2) * g * np.sin(theta2)) / den2)

            return [theta1_dot, omega1_dot, theta2_dot, omega2_dot]

        theta1_range = np.linspace(0, 2*np.pi, discretize)
        theta2_range = np.linspace(0, 2*np.pi, discretize)
        omega1_range = np.linspace(-1, 1, discretize)
        omega2_range = np.linspace(-1, 1, discretize)
        combinations = np.array(list(product(theta1_range, theta2_range, omega1_range, omega2_range)))

        input_lst = []
        output_lst = []
        t = [0, 0.1]

        for row in combinations:
            sol = odeint(double_pendulum, row, t)
            input_lst.append(row)
            output_lst.append(sol[1])

        inputs = np.array(input_lst)
        outputs = np.array(output_lst)

        return inputs, outputs

    def generate_pendulum_data(self, disc):
            # Define the ODEs
            def damped_pendulum(y, t, b, c):
                theta, omega = y
                dydt = [omega, -b * omega - c * np.sin(theta)]
                return dydt

            # Define the parameters
            b = 0.1  # damping coefficient
            c = 1.0  # gravitational constant

            # Training data
            t_stepsize = 0.1 # stepsize = 0.1 seconds
            t_range = np.array([0, t_stepsize])

            # Create theta_range and omega_range arrays
            theta_range = np.linspace(-np.pi, np.pi, disc)
            omega_range = np.linspace(-1, 1, disc)

            # Create a grid of coordinates
            theta_grid, omega_grid = np.meshgrid(theta_range, omega_range)

            # Flatten the grid to obtain the list of all possible combinations
            points = np.column_stack((theta_grid.flatten(), omega_grid.flatten()))

            solutions = np.zeros((disc**2, 2))
            for i in range(len(points)):
                sol = odeint(damped_pendulum, points[i], t_range, args=(b, c))[1]
                solutions[i] = sol

            input_df = pd.DataFrame(points, columns=['theta', 'omega'])
            output_df = pd.DataFrame(solutions, columns=['theta', 'omega'])
            return input_df, output_df

    def generate_two_tank_system_data(self, disc):
        # Define the ODE system
        def tank_system(state, t, q):
            A1 = 1
            A2 = 1
            g = 9.81
            h1, h2 = state
            q1 = A1 * np.sqrt(2 * g * (h1-h2))
            q2 = A2 * np.sqrt(2 * g * h2)

            dh1_dt = (q - q1) / A1
            dh2_dt = (q1 - q2) / A2

            return [dh1_dt, dh2_dt]

        # Training data
        t_stepsize = 0.1  # stepsize = 0.1 seconds
        t_range = np.array([0, t_stepsize])

        # Create theta_range and omega_range arrays
        h1_range = np.linspace(20, 40, disc)
        h2_range = np.linspace(0, 20, disc)
        q_range = np.linspace(1, 20, int(disc/2))

        # Create a grid of coordinates
        X_test = np.array(list(product(h1_range, h2_range, q_range)))
        ground_truth = []
        for i in range(len(X_test)):
            y0 = [X_test[i, 0], X_test[i, 1]]
            sol = odeint(tank_system, y0, t_range, args=(X_test[i, 2],))
            ground_truth.append([sol[1, 0], sol[1, 1]])

        input_df = pd.DataFrame(X_test, columns=['h1', 'h2', "q"])
        output_df = pd.DataFrame(ground_truth, columns=['h1', 'h2'])
        return input_df, output_df

    def save_data_to_csv(self, inputs, outputs, system_name, data_type):
        dir_name = system_name
        os.makedirs(dir_name, exist_ok=True)
        prefix = os.path.join(dir_name, f'{system_name}_{data_type}')
        
        # Determine columns for inputs and outputs based on the system name
        if system_name == 'lorenz':
            input_columns = ['x', 'y', 'z']
            output_columns = ['x', 'y', 'z']
        elif system_name == 'double_pendulum':
            input_columns = ['theta1', 'theta2', 'omega1', 'omega2']
            output_columns = ['theta1', 'theta2', 'omega1', 'omega2']
        elif system_name == 'actuated_pendulum':
            input_columns = ['theta', 'omega', 'torque']
            output_columns = ['theta', 'omega']
        elif system_name == 'pendulum':
            input_columns = ['theta', 'omega']
            output_columns = ['theta', 'omega']
        elif system_name == 'two_tank_system':
            input_columns = ['h1', 'h2', 'q']
            output_columns = ['h1', 'h2']

        # Create DataFrames for inputs and outputs
        df_inputs = pd.DataFrame(inputs, columns=input_columns)
        df_outputs = pd.DataFrame(outputs, columns=output_columns)

        # Save DataFrames to CSV files
        df_inputs.to_csv(f'{prefix}_inputs.csv', index=False)
        df_outputs.to_csv(f'{prefix}_outputs.csv', index=False)

if __name__ == "__main__":
    generator = DataGenerator()

    # Generate and save Lorenz training data
    sigma = 10
    rho = 28
    beta = 8/3
    initials = [1.0, 0.0, 0.0]
    t = np.linspace(0, 1000, 10000)
    lorenz_train_inputs, lorenz_train_outputs = generator.generate_lorenz_data(initials, t, sigma, rho, beta)
    generator.save_data_to_csv(lorenz_train_inputs, lorenz_train_outputs, 'lorenz', 'train')

    # Generate and save Lorenz test data
    test_initials = [10, 20.0, 5.0]
    lorenz_test_inputs, lorenz_test_outputs = generator.generate_lorenz_data(test_initials, t, sigma, rho, beta)
    generator.save_data_to_csv(lorenz_test_inputs, lorenz_test_outputs, 'lorenz', 'test')

    val_initials = [-10.0, 42.0, -3.0]
    lorenz_val_inputs, lorenz_val_outputs = generator.generate_lorenz_data(val_initials, t, sigma, rho, beta)
    generator.save_data_to_csv(lorenz_val_inputs, lorenz_val_outputs, 'lorenz', 'validation')

    # Generate and save Actuated Pendulum training data
    actuated_pendulum_train_inputs, actuated_pendulum_train_outputs = generator.generate_actuated_pendulum_data()
    generator.save_data_to_csv(actuated_pendulum_train_inputs, actuated_pendulum_train_outputs, 'actuated_pendulum', 'train')

    # Generate and save Actuated Pendulum test data
    actuated_pendulum_test_inputs, actuated_pendulum_test_outputs = generator.generate_actuated_pendulum_data(disc=25)
    generator.save_data_to_csv(actuated_pendulum_test_inputs, actuated_pendulum_test_outputs, 'actuated_pendulum', 'test')

    # Generate and save Double Pendulum data
    double_pendulum_train_inputs, double_pendulum_train_outputs = generator.generate_double_pendulum_data(11)
    double_pendulum_test_inputs, double_pendulum_test_outputs = generator.generate_double_pendulum_data(9)
    generator.save_data_to_csv(double_pendulum_train_inputs, double_pendulum_train_outputs, 'double_pendulum', 'train')
    generator.save_data_to_csv(double_pendulum_test_inputs, double_pendulum_test_outputs, 'double_pendulum', 'test')

    # Generate and save Pendulum data
    pendulum_train_inputs, pendulum_train_outputs = generator.generate_pendulum_data(100)
    pendulum_test_inputs, pendulum_test_outputs = generator.generate_pendulum_data(70)
    generator.save_data_to_csv(pendulum_train_inputs, pendulum_train_outputs, 'pendulum', 'train')
    generator.save_data_to_csv(pendulum_test_inputs, pendulum_test_outputs, 'pendulum', 'test')

    # Generate and save Two Tank System data
    two_tank_system_train_inputs, two_tank_system_train_outputs = generator.generate_two_tank_system_data(30)
    two_tank_system_test_inputs, two_tank_system_test_outputs = generator.generate_two_tank_system_data(25)
    generator.save_data_to_csv(two_tank_system_train_inputs, two_tank_system_train_outputs, 'two_tank_system', 'train')
    generator.save_data_to_csv(two_tank_system_test_inputs, two_tank_system_test_outputs, 'two_tank_system', 'test')

