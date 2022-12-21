import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatch
import matplotlib.colors as mcol
import matplotlib.animation as manim
from constants import c

# Define RGB triples for colors
RED = [255, 0, 0]
LIGHTRED = [255, 192, 203]
GRAY = [128, 128, 128]
BLACK = [0, 0, 0]
GREEN = [0, 255, 0]
LIGHTGREEN = [152, 251, 152]
WHITE = [255, 255, 255]

# Define character values and corresponding colors and labels
COLORMAP = {c.FREE: [WHITE, "Free Space"],
            c.WALL: [GRAY, "Wall"],
            c.ROBOT: [RED, "Robot"],
            c.PREV_PATH: [LIGHTRED, "Previous Path"],
            c.NEXT_PATH: [LIGHTGREEN, "Next Path"],
            c.FRONTIER_CELL: [GREEN, "Frontier Cell"],
            c.UNEXPLORED: [BLACK, "Unexplored Space"],            
            }

class GraphicsWindow:
    def __init__(self, dynamic_grid):
        self.env = dynamic_grid

    def convert(self):
        env = self.env
        # Made 2D grid a 3D one
        extra_dim = np.zeros_like(env)
        env = np.dstack((env, extra_dim, extra_dim))
        # Replace representations with colors
        for key, value in COLORMAP.items():
            env[env[:, :, 0] == key] = np.array(value[0])

        self.env = env
        
    def show_env(self, timestep, algo): 
        self.convert()
        env = self.env

        # Define image
        im = plt.imshow(self.env, interpolation='none')
        color_vals = np.unique(np.array(env).ravel())
        # Get unique colors within map
        pixel_info = [val for val in COLORMAP.values()]
        # Create color patches for legend
        color_patches = []
        for i in range(len(pixel_info)):
            color_patches.append(mpatch.Patch(color=mcol.to_rgba(im.norm(pixel_info[i][0])), label=pixel_info[i][1]))
        # Use patches as handles for legend
        plt.legend(handles=color_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.title('Robot Map\n' + algo + ', Time Step = ' + str(timestep))
        #plt.show()
        plt.savefig(os.getcwd() + '/Frames/Frame_' + str(timestep) + '.png', bbox_inches='tight')


        
        
        
        

        






