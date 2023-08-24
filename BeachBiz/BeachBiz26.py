import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def time_and_distance_sand_to_water(x3, x2, y1, y2, v_run, v_swim):
    distance_run = np.sqrt((x3 - x2) ** 2 + y2 ** 2)
    distance_swim = np.sqrt(x3 ** 2 + y1 ** 2)

    t_run = distance_run / v_run
    t_swim = distance_swim / v_swim

    return t_run + t_swim, distance_run + distance_swim, t_run, t_swim, distance_run, distance_swim


def time_and_distance_swim_run_swim_water_to_water(x3, x4, x2, y1, y2, v_run, v_swim):
    distance_swim1 = np.sqrt((x3 - x2) ** 2 + y2 ** 2)
    distance_run = abs(x3 - x4)
    distance_swim2 = np.sqrt(x4 ** 2 + y1 ** 2)

    t_swim1 = distance_swim1 / v_swim
    t_run = distance_run / v_run
    t_swim2 = distance_swim2 / v_swim

    return t_swim1 + t_run + t_swim2, \
           distance_swim1 + distance_run + distance_swim2


def time_and_distance_direct_swim_water_to_water(x2, y2, y1, v_swim):
    """Calculate time taken if one directly swims to the person in trouble."""
    distance_swim_direct = np.sqrt(x2**2 + (y2 - y1)**2)
    time_swim_direct = distance_swim_direct / v_swim
    print(f'time_swim_direct = {time_swim_direct}, distance_swim_direct = {distance_swim_direct}, ')
    return time_swim_direct, distance_swim_direct


def find_optimal_x3(x2, y1, y2, v_run, v_swim):
    fun = lambda x3: time_and_distance_sand_to_water(x3, x2, y1, y2, v_run, v_swim)[0]
    result = minimize(fun, x2, bounds=[(0, None)])
    return result.x[0], result.fun


def find_optimal_x3_x4(x2, y1, y2, v_run, v_swim):
    fun = lambda x: time_and_distance_swim_run_swim_water_to_water(x[0], x[1], x2, y1, y2, v_run, v_swim)[0]
    result = minimize(fun, [x2 / 2, x2 / 2], bounds=[(0, None), (0, None)])
    return result.x[0], result.x[1], result.fun


def plot_scenario(col, x2, y2, y1, v_sand, v_water, scenario, ax):

    scenario_col_name = f"v_sand = {v_sand} m/s, v_water = {v_water} m/s"
    print(f'  plot_scenario(scenario = {scenario}, col = {col}, scenario_col_name={scenario_col_name}')

    if scenario == "sand_to_water":
        opt_x3, opt_time = find_optimal_x3(x2, y1, y2, v_sand, v_water)
        print(f'    opt_time = {opt_time}')
        opt_time_again, corresponding_distance, _, _, _, _ = \
            time_and_distance_sand_to_water(opt_x3, x2, y1, y2, v_sand, v_water)
        print(f'    opt_time_again = {opt_time_again}')

        x3_values = np.linspace(0, x2 + 5, 400)
        times, distances, t_run, t_swim, d_run, d_swim = zip(
            *[time_and_distance_sand_to_water(x3, x2, y1, y2, v_sand, v_water) for x3 in x3_values])

        # Add scenario speeds to the top plot
        ax[0].set_title(scenario_col_name + "\nfunction terms plot")

        # sand_to_water plotting
        ax[0].plot(x3_values, times, label='total time')
        ax[0].plot(x3_values, t_run, 'r--', label='run time')
        ax[0].plot(x3_values, t_swim, 'b--', label='swim time')
        ax[0].plot(x3_values, d_run, 'r-.', label='run distance')
        ax[0].plot(x3_values, d_swim, 'b-.', label='swim distance')
        ax[0].axvline(opt_x3, color='black', linestyle='--', label=f'optimal x3: {opt_x3:.2f} m')
        ax[0].legend()

        ax[1].set_title("route plot")
        ax[1].set_aspect('equal', 'box')

        ax[1].fill_between([0, x2 + 5], 0, 15 + 10, color='lightblue', label='water')
        ax[1].fill_between([0, x2 + 5], -20, 0, color='yellow', label='beach')
        ax[1].plot([x2, opt_x3, 0], [y2, 0, y1], label='optimal path')
        ax[1].scatter([x2, opt_x3, 0], [y2, 0, y1], color=['blue', 'red', 'green'])
        ax[1].annotate('damsel', (0, y1), textcoords="offset points", xytext=(10, 0), ha='left', color='green')
        ax[1].annotate('in distress', (0, y1), textcoords="offset points", xytext=(10, -10), ha='left', color='green')

        ax[1].annotate('lifeguard', (x2, y2), textcoords="offset points", xytext=(10, +30), ha='right',
                       color='blue')

        # Determine position based on opt_x3 value
        horizontal_offset = -20 if opt_x3 > x2 / 2 else 20
        alignment = 'right' if opt_x3 > x2 / 2 else 'left'

        # Adjust vertical separation and place the top string in the water
        ax[1].annotate('optimal', (opt_x3, 2), textcoords="offset points",
                       xytext=(horizontal_offset, -3), ha=alignment, color='red')

        # Place the bottom string in the sand
        ax[1].annotate('entry point', (opt_x3, -5), textcoords="offset points",
                       xytext=(horizontal_offset, 7), ha=alignment, color='red')

        # For a python version = 3.9.7, I get a tpye = <class 'numpy.ndarray'>
        # while for a python version = 3.9, I get a type(opt_time) = <class 'float'>, so the following casts
        # all cases down to a float.
        if type(opt_time) is not float:
            opt_time = opt_time[0]

        result_color = 'blue'
        ax[1].text(2, -15, f"min. total time = {opt_time:.2f} s", color=result_color)
        ax[1].text(2, -18, f"corresp. distance = {corresponding_distance:.2f} m", color=result_color)

        ax[1].legend()



    elif scenario == "water_to_water":
        from scipy.optimize import minimize

        # Objective function for time
        def time_objective(params, x2, y1, y2, v_sand, v_water):
            x3, x4 = params
            time, _ = time_and_distance_swim_run_swim_water_to_water(x3, x4, x2, y1, y2, v_sand, v_water)
            print(f"### x3 = {x3}, x4 = {x4}, time = {time}")
            return time

        # Objective function for distance
        def distance_objective(params, x2, y1, y2, v_sand, v_water):
            x3, x4 = params
            _, distance = time_and_distance_swim_run_swim_water_to_water(x3, x4, x2, y1, y2, v_sand, v_water)
            print(f"### x3 = {x3}, x4 = {x4}, distance = {distance}")
            return distance

        dummy = 0
        distances, times = \
            zip(*[time_and_distance_swim_run_swim_water_to_water(x, dummy, x2, y1, y2, v_sand, v_water) for x in
                  np.linspace(0, x2 + 5, 400)])

        # Now finding optimal x3 and x4 for time
        initial_guess = [x2 / 4 , 3 * x2 / 4]  # Guessing midpoint values for x3 and x4
        bounds = [(0, x2 + 5), (0, x2 + 5)]
        result_time = minimize(time_objective, initial_guess, args=(x2, y1, y2, v_sand, v_water), bounds=bounds)

        if result_time.success:
            x3_time_opt, x4_time_opt = result_time.x
        else:
            raise ValueError("time optimization failed")

        # Finding optimal x3 and x4 for distance
        result_distance = minimize(distance_objective, initial_guess, args=(x2, y1, y2, v_sand, v_water),
                                   bounds=bounds,
                                   options={'maxiter': 1000,
                                            'ftol': 0.01, # is in meter, 1 cm is good enough for beach purposes!
                                            'gtol': 0.001
                                            })
        if result_distance.success:
            x3_dist_opt, x4_dist_opt = result_distance.x
        else:
            raise ValueError("Distance optimization failed")

        ax[0].set_title(scenario_col_name)
        n_dimensions = 2
        if n_dimensions == 1:
            ax[0].plot(np.linspace(0, x2 + 5, 400), times, label='total time')
            ax[0].legend()
            n_plot_rows = 1
        else:
            assert n_dimensions == 2
            # Calculating contours for water_to_water scenario
            x_values = np.linspace(0, x2 + 5, 100)
            times = np.zeros((len(x_values), len(x_values)))
            distances = np.zeros((len(x_values), len(x_values)))
            for i, x3 in enumerate(x_values):
                for j, x4 in enumerate(x_values):
                    times[i, j], distances[i, j] = \
                        time_and_distance_swim_run_swim_water_to_water(x3, x4, x2, y1, y2, v_sand, v_water)

            # Contour plot for total distance
            ax[0].set_aspect('equal', 'box')
            contour_dist = ax[0].contour(x_values, x_values, distances, 50, cmap='viridis')
            ax[0].clabel(contour_dist, inline=True, fontsize=8)
            ax[0].set_title(scenario_col_name + "\ntotal distance plot")
            ax[0].set_xlabel('x4')
            ax[0].set_ylabel('x3')

            # Contour plot for total time
            ax[1].set_aspect('equal', 'box')
            contour_time = ax[1].contour(x_values, x_values, times, 50, cmap='plasma')
            ax[1].clabel(contour_time, inline=True, fontsize=8)
            ax[1].set_title("total time plot")
            ax[1].set_xlabel('x4')
            ax[1].set_ylabel('x3')

            n_plot_rows = 2

            ax[0].axhline(x3_dist_opt, color='r', linestyle='--', label=f'dist optimal x3 = {x3_dist_opt:.2f}')
            ax[0].axvline(x4_dist_opt, color='purple', linestyle='--', label=f'dist optimal x4 = {x4_dist_opt:.2f}')

            ax[1].axhline(x3_time_opt, color='r', linestyle='--', label=f'time optimal x3 = {x3_time_opt:.2f}')
            ax[1].axvline(x4_time_opt, color='purple', linestyle='--', label=f'time optimal x4 = {x4_time_opt:.2f}')

        i = n_plot_rows
        ax[i].set_aspect('equal', 'box')

        # water to water: indirect
        opt_x3, opt_x4, optimal_time_indirect = find_optimal_x3_x4(x2, y1, y2, v_sand, v_water)
        print(f'    optimal_time_indirect = {optimal_time_indirect}')
        optimal_time_indirect_again, corresponding_distance_indirect = \
            time_and_distance_swim_run_swim_water_to_water(opt_x3, opt_x4, x2, y1, y2, v_sand, v_water)
        print(f'    optimal_time_indirect_again = {optimal_time_indirect_again}')

        # water to water: direct
        time_swim_direct, corresponding_distance_swim_direct = \
            time_and_distance_direct_swim_water_to_water(x2, y2, y1, v_water)
        print(f'    time_swim_direct = {time_swim_direct}')

        direct_is_faster = time_swim_direct < optimal_time_indirect
        if direct_is_faster:
            ax[i].set_title("direct route is faster")
            result_color = 'blue'
            optimal_time = time_swim_direct
            corresponding_distance = corresponding_distance_swim_direct
        else:
            ax[i].set_title("indirect route is faster")
            result_color = 'orange'
            optimal_time = optimal_time_indirect
            corresponding_distance = corresponding_distance_indirect

        ax[i].text(2, -15, f"min. total time = {optimal_time:.2f} s", color=result_color)
        ax[i].text(2, -18, f"corresp. distance = {corresponding_distance:.2f} m", color=result_color)

        ax[i].fill_between([0, x2 + 5], 0, 15 + 10, color='lightblue', label='water')
        ax[i].fill_between([0, x2 + 5], -20, 0, color='yellow', label='beach')

        # Direct path coordinates
        direct_x = [x2, 0]
        direct_y = [y2, y1]

        # Indirect path coordinates
        indirect_x = [x2, opt_x3, opt_x4, 0]
        indirect_y = [y2, 0, 0, y1]

        if direct_is_faster:
            print('    direct_is_faster')
            direct_style = '-'  # solid
            indirect_style = '--' # dashed
        else:
            print('    indirect_is_faster')
            direct_style = '--' # dashed
            indirect_style = '-' # solid
        ax[i].plot(direct_x, direct_y, label='best direct path',
                   linestyle=direct_style)  # Solid line for optimal path
        ax[i].plot(indirect_x, indirect_y, label='best indirect path',
                   linestyle=indirect_style)
        ax[i].scatter([opt_x3, opt_x4], [0, 0], color=['red', 'purple'],
                      marker='o')  # Points for the non-optimal path

        ax[i].scatter([x2, 0], [y2, y1], color=['blue', 'green'])

        ax[i].annotate('damsel', (0, y1), textcoords="offset points", xytext=(10, 0), ha='left', color='green')
        ax[i].annotate('in distress', (0, y1), textcoords="offset points", xytext=(10, -10), ha='left', color='green')

        ax[i].annotate('lifeguard', (x2, y2), textcoords="offset points", xytext=(0, -10), ha='right',
                       color='blue')

        # Annotations for x3 and x4 values
        ax[i].annotate(f'x3={opt_x3:.2f}', (opt_x3, -2), textcoords="offset points",
                       xytext=(10, -10), ha='center', va='top', color='red')
        ax[i].annotate(f'x4={opt_x4:.2f}', (opt_x4, -5), textcoords="offset points",
                       xytext=(10, -10), ha='center', va='top', color='purple')

        ax[i].legend(loc='upper right')


# Scenario parameters
scenarios = [(5, 2), (3, 2), (2, 2), (2, 3)]
n_cols = len(scenarios)

test_sand_to_water = True
if test_sand_to_water:
    x2, y2, y1 = 50, -20, 15
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    # Test the scenarios for sand_to_water
    for col, (v_sand, v_water) in enumerate(scenarios):
        plot_scenario(col, x2, y2, y1, v_sand, v_water, "sand_to_water", axes[:, col])
    plt.tight_layout()
    plt.show()
    fig.savefig("sand_to_water.pdf", bbox_inches='tight')
    fig.savefig("sand_to_water.png", bbox_inches='tight')

test_water_to_water = True
if test_water_to_water:
    x2, y2, y1 = 50, +5, 15

    n_rows = 2
    do_distance_plots = True
    if do_distance_plots:
        n_rows += 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    # Test the scenarios for water_to_water
    for col, (v_sand, v_water) in enumerate(scenarios):
        plot_scenario(col, x2, y2, y1, v_sand, v_water, "water_to_water", axes[:, col])
    plt.tight_layout()
    plt.show()
    fig.savefig("water_to_water.pdf", bbox_inches='tight')
    fig.savefig("water_to_water.png", bbox_inches='tight')
