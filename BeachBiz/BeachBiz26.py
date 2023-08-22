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
        ax[0].set_title(scenario_col_name + "\nFunction Terms Plot")

        # sand_to_water plotting
        ax[0].plot(x3_values, times, label='Total time')
        ax[0].plot(x3_values, t_run, 'r--', label='Run time')
        ax[0].plot(x3_values, t_swim, 'b--', label='Swim time')
        ax[0].plot(x3_values, d_run, 'r-.', label='Run distance')
        ax[0].plot(x3_values, d_swim, 'b-.', label='Swim distance')
        ax[0].axvline(opt_x3, color='black', linestyle='--', label=f'Optimal x3: {opt_x3:.2f} m')
        ax[0].legend()

        ax[1].set_title("Route Plot")
        ax[1].set_aspect('equal', 'box')

        ax[1].fill_between([0, x2 + 5], -20, 0, color='yellow', label='Beach')
        ax[1].fill_between([0, x2 + 5], 0, 15 + 10, color='lightblue', label='Water')
        ax[1].plot([x2, opt_x3, 0], [y2, 0, y1], label='Optimal path')
        ax[1].scatter([x2, opt_x3, 0], [y2, 0, y1], color=['blue', 'red', 'green'])
        ax[1].annotate('Person in trouble', (0, y1), textcoords="offset points", xytext=(10, 0), ha='left')
        ax[1].annotate('Optimal', (opt_x3, 2), textcoords="offset points", xytext=(0, -15), ha='center',
                       color='black')
        ax[1].annotate('entry point', (opt_x3, -5), textcoords="offset points", xytext=(0, 5), ha='center',
                       color='black')

        # For a python version = 3.9.7, I get a tpye = <class 'numpy.ndarray'>
        # while for a python version = 3.9, I get a type(opt_time) = <class 'float'>, so the following casts
        # all cases down to a float.
        if type(opt_time) is not float:
            opt_time = opt_time[0]
        ax[1].text(2, -15, f"Optimal total time = {opt_time:.2f} s", color='green')
        ax[1].text(2, -18, f"corresponding distance = {corresponding_distance:.2f} m", color='green')

        ax[1].legend()

    elif scenario == "water_to_water":
        opt_x3, opt_x4, _ = find_optimal_x3_x4(x2, y1, y2, v_sand, v_water)
        distances, times = \
            zip(*[time_and_distance_swim_run_swim_water_to_water(x, opt_x4, x2, y1, y2, v_sand, v_water) for x in
                                 np.linspace(0, x2 + 5, 400)])

        ax[0].set_title(scenario_col_name)
        n_dimensions = 2

        if n_dimensions == 1:
            ax[0].plot(np.linspace(0, x2 + 5, 400), times, label='Total time')
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
            ax[0].set_title(scenario_col_name + "\nTotal Dist. Plot")
            ax[0].set_xlabel('x4')
            ax[0].set_ylabel('x3')

            # Contour plot for total time
            ax[1].set_aspect('equal', 'box')
            contour_time = ax[1].contour(x_values, x_values, times, 50, cmap='plasma')
            ax[1].clabel(contour_time, inline=True, fontsize=8)
            ax[1].set_title("Total Time Plot")
            ax[1].set_xlabel('x4')
            ax[1].set_ylabel('x3')
            ##
            n_plot_rows = 2

        i = n_plot_rows
        ax[i].set_title("Route Plot")
        ax[i].set_aspect('equal', 'box')

        # 1
        opt_x3, opt_x4, optimal_time_indirect = find_optimal_x3_x4(x2, y1, y2, v_sand, v_water)
        print(f'    optimal_time_indirect = {optimal_time_indirect}')
        optimal_time_indirect_again, corresponding_distance_indirect = \
            time_and_distance_swim_run_swim_water_to_water(opt_x3, opt_x4, x2, y1, y2, v_sand, v_water)
        print(f'    optimal_time_indirect_again = {optimal_time_indirect_again}')

        # 2
        time_swim_direct, corresponding_distance_swim_direct = \
            time_and_distance_direct_swim_water_to_water(x2, y2, y1, v_water)
        print(f'    time_swim_direct = {time_swim_direct}')

        direct_is_faster = time_swim_direct < optimal_time_indirect
        if direct_is_faster:
            optimal_time = time_swim_direct
            corresponding_distance = corresponding_distance_swim_direct
        else:
            optimal_time = optimal_time_indirect
            corresponding_distance = corresponding_distance_indirect

        ax[i].text(2, -15, f"Optimal total time = {optimal_time:.2f} s", color='green')
        ax[i].text(2, -18, f"corresponding distance = {corresponding_distance:.2f} m", color='green')

        ax[i].fill_between([0, x2 + 5], -20, 0, color='yellow', label='Beach')
        ax[i].fill_between([0, x2 + 5], 0, 15 + 10, color='lightblue', label='Water')

        if direct_is_faster:
            print('    direct_is_faster')
            ax[i].plot([x2, 0], [y2, y1], label='Optimal path')
        else:
            print('    indirect_is_faster')
            ax[i].plot([x2, opt_x3, opt_x4, 0], [y2, 0, 0, y1], label='Optimal path')
            ax[i].scatter([opt_x3, opt_x4], [0, 0], color=['red', 'purple'])

        ax[i].scatter([x2, 0], [y2, y1], color=['blue', 'green'])


        ax[i].annotate('Person in trouble', (0, y1), textcoords="offset points", xytext=(10, 0), ha='left')

        # Annotations for x3 and x4 values
        ax[i].annotate(f'x3={opt_x3:.2f}', (opt_x3, -2), textcoords="offset points",
                       xytext=(10, -10), ha='center', va='top', color='red')
        ax[i].annotate(f'x4={opt_x4:.2f}', (opt_x4, -5), textcoords="offset points",
                       xytext=(10, -10), ha='center', va='top', color='red')

        ax[i].legend()

scenarios = [(5, 2), (3, 2), (2, 2), (2, 3)]
n_cols = len(scenarios)


# Scenario parameters
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
