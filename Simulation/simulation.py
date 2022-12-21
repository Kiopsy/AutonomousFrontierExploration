# Final Project Simulation
from fileinput import filename
from environment import Env
from constants import c
from graphics import GraphicsWindow
import pandas as pd

# Variables to set up environment:

# Choosing which from: random walk, greedy, or wayfront_detection
frontier_algo = c.GREEDY
# Show graphics: 1 or 0
show_graphics = 0
# layout filename
filename = 'Layouts/Layout4/layout4_int.txt'

def main():
    # Define filenames
    '''
    max_robots = 4
    filenames = []
    for layout_num in [1, 4]:
        for num_robots in range(1, max_robots + 1):
            filenames.append('Layouts/Layout' + str(layout_num) + '/layout' + str(layout_num) + '_' + str(num_robots) + '_int.txt')
    '''

    # instantiating the environment & running simulations
    # Define the max number of robots on which you hope to test
    max_robots = 2
    # Create list for data
    all_data = []
    # Run simulation for each algo, on each number of robots <= max_robots, and on each layout number in layout_nums
    for algo in [c.WAVEFRONT, c.GREEDY]:
        # Create string version for dataframe
        algorithm = "WFD" if algo == c.WAVEFRONT else "Greedy"
        #for filename in filenames:
        layout_nums = [1]
        for layout_num in layout_nums:
            for num_robots in range(1, max_robots + 1):
                print("Testing " + algorithm + " on layout #" + str(layout_num) + " and with " + str(num_robots) + " robots.")
                filename = 'Layouts/Layout' + str(layout_num) + '/layout' + str(layout_num) + '_' + str(num_robots) + '_int.txt'
                env = Env(filename, algo, show_graphics)
                area = env.grid.size
                # Get total number of time steps
                timesteps = env.run_simulation()
                # Add row to list/dataframe
                all_data.append([algorithm, area, num_robots, timesteps])
        print('==========Finished testing ' + algorithm + ' algorithm.==========')
        print()
    df = pd.DataFrame(all_data, columns=['Algorithm', 'Map Area (Cells^2)', 'Number of Robots', 'Time to Completion (Time Steps)'])
    print(df)
    print("Writing to CSV...")
    df.to_csv('Data/simulation_data.csv')
    print("Finished!")
    print("")
    
    
if __name__ == "__main__":
    main()