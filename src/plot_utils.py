import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os

from PIL import Image


nfl_image = "../documents/images/NFL-logo.png"


def draw_graph(G, ax=None):
    '''Function to plot teams as graphs, with ball.'''
    colors = {-1: "tab:blue", 1:'tab:orange', 0:'brown'}
    node_colors = []
    pos_dict = {}
    for node, attr in G.nodes(data=True):
        team_def = attr["team_redeffinition"]
        node_colors.append(colors[team_def])
        pos_dict[node] = np.array([attr["pos_x"],attr["pos_y"]])
    edges = G.edges(data=True)
    weights = [v[-1]['weight'] for v in edges]
    pos = pos_dict
    ec = nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.9, ax=ax)


def create_folder(filename, path="../figures/"):
    complete_path = os.path.join(path, filename)
    if os.path.exists(complete_path):
        return None
    os.mkdir(complete_path)
    return complete_path


def plot_field(subplots=2, figsize=(15,10)):
  '''Function to plot the football field. '''
  stadium_limit_dimension_y =  (0, 53.3333333)
  stadium_limit_dimension_x = (0, 120)
  color__green = "#586d3d"
  color__white = "#DADADA"
  color__brown = "#663831"
  color__yellow = "#cbb67c"
  text_position = 5
  portery_yards = 6.166667
  numbers = {20:'1 0', 30:'2 0', 40:'3 0', 50:'4 0', 60:'5 0', 70:'4 0', 80:'3 0', 90:'2 0', 100:'1 0'}

  fig, ax = plt.subplots(subplots, figsize=figsize)
  ax[0].set_xlim(stadium_limit_dimension_x)
  ax[0].set_ylim(stadium_limit_dimension_y)
  ax[0].vlines(
      [i for i in range(0, max(stadium_limit_dimension_x), 10)],
      min(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y),
      color=color__white,
      alpha=0.5
  )

  ax[0].vlines(
      [i for i in range(10, max(stadium_limit_dimension_x)-10, 5)],
      min(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y),
      color=color__white,
      alpha=0.3
  )

  vlines2 = [i for i in range(10, max(stadium_limit_dimension_x)-10+1, 1)]
  vlines2_size=1
  ax[0].vlines(
      vlines2,
      min(stadium_limit_dimension_y),
      min(stadium_limit_dimension_y) + vlines2_size,
      color=color__white,
      alpha=0.3
  )
  ax[0].vlines(
      vlines2,
      max(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y) - vlines2_size,
      color=color__white,
      alpha=0.5
  )

  vlines3_size = 0.5
  medium_vlines_pos = (min(stadium_limit_dimension_y) + max(stadium_limit_dimension_y))/2 + portery_yards/2
  medium_vlines_neg = (min(stadium_limit_dimension_y) + max(stadium_limit_dimension_y))/2 - portery_yards/2
  ax[0].vlines(
      vlines2,
      medium_vlines_pos + vlines3_size,
      medium_vlines_pos - vlines3_size,
      color=color__white,
      alpha=0.3
  )
  ax[0].vlines(
      vlines2,
      medium_vlines_neg + vlines3_size,
      medium_vlines_neg - vlines3_size,
      color=color__white,
      alpha=0.3
  )
  for p, n in numbers.items():
    ax[0].text(p, min(stadium_limit_dimension_y) + text_position, n, horizontalalignment='center', verticalalignment='center', color=color__white, rotation=0)
    ax[0].text(p, max(stadium_limit_dimension_y) - text_position, n, horizontalalignment='center', verticalalignment='center', color=color__white, rotation=180)

  triangle_pos = [(i + (i-5))/2 for i in range(20, max(stadium_limit_dimension_x)//2-10+1, 10)]
  ax[0].scatter(triangle_pos, [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos), marker='<', s=10, color=color__white)
  ax[0].scatter(triangle_pos, [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos), marker='<', s=10, color=color__white)

  triangle_pos2 = [(i + (i+5))/2 for i in range(70, max(stadium_limit_dimension_x)-10, 10)]
  ax[0].scatter(triangle_pos2, [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos2), marker='>', s=10, color=color__white)
  ax[0].scatter(triangle_pos2, [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos2), marker='>', s=10, color=color__white)
  ax[0].set_facecolor(color__green)
  ax[0].set_xticks([])
  ax[0].set_yticks([])
  logosize = 7
  extent = (
      stadium_limit_dimension_x[1]//2-logosize,
      stadium_limit_dimension_x[1]//2+logosize,
      stadium_limit_dimension_y[1]//2-logosize,
      stadium_limit_dimension_y[1]//2+logosize
  )
  ax[0].imshow(plt.imread(nfl_image), aspect='auto', extent=extent, alpha=0.5)
  return fig, ax

def save_figs(information, filename, show=False, save=False, stop_counter=None):
    complete_path = create_folder(filename=filename)
    if complete_path is None:
        return None

    game_id = information.gameId.unique()[0]
    play_id = information.playId.unique()[0]
    play_direction = information.playDirection_x.unique()[0]

    counter=0
    for idx, row in information.iterrows():
        increasing_information = information.loc[:idx]
        time_component = pd.to_datetime(increasing_information.time, format='ISO8601')
        time_component = (time_component - time_component.min()).dt.total_seconds()

        fig, ax = plot_field(subplots=2, figsize=(15,10))
        
        draw_graph(row.graphs, ax=ax[0])
    
        ax[1].plot(
            increasing_information.time_component, 
            increasing_information.model_pred,
            color='blue', 
            marker='o',
            markerfacecolor='k',
            markersize=3,
            label='Model score',
        )
        #ax[1].plot(
        #    increasing_frame_sample_tracking.time, 
        #    increasing_frame_sample_tracking.manual_cutpoint, 
        #    color='gray', 
        #    linestyle='--', 
        #    label='pass prediction threshold')
        
        increasing_information_events = increasing_information[
            increasing_information.event.notna()
        ]
    
        ax[1].vlines(
            increasing_information_events.time_component, 
            [0 - 0.0] * increasing_information_events.shape[0], 
            [1 - 0.05] * increasing_information_events.shape[0],
            color='k',
            alpha=0.8,
        )
    
        counter_text = 0
        for _, row_event in increasing_information_events.iterrows():
            counter_text +=1
            ax[1].text(
                row_event.time_component, 
                counter_text%2 - (-1)**counter_text*0.05,
                row_event.event, 
                size=5,  
                horizontalalignment='center', 
                verticalalignment='center',
            )
        ax[1].set_ylim(-0.15, 1.1)
    
        plt.suptitle('Play development over time.')
        ax[1].set_title('Overall play behavior')
        ax[1].set_title(f'Model prediction [game:{game_id}, play:{play_id}]')
        ax[1].set_xlabel('Seconds from ball snap')
        ax[1].set_ylabel('Probability of Pass Forward prediction \n(caught) at  the next 0.5 second.')
        ax[1].grid(linestyle='--')
    
        ax[1].legend()
        if show:
            plt.show()
        if save:
            plt.savefig(f'{complete_path}/test_{str(counter).zfill(3)}.png')
        plt.close(fig)
        counter += 1
        if stop_counter:
            if counter==stop_counter:
                break


def create_gif(image_paths, output_gif_path, fps=5):
  images = [Image.open(image_path) for image_path in image_paths]
  images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:],
    fps=fps,
    loop=0 # 0 means infinite loop
  )


def save_gif(filename, fps=5):
    image_paths = [os.path.join(f'../figures/{filename}', p) for p in sorted(os.listdir(f'../figures/{filename}'))]
    filename_split = filename.split('_')[-1]
    output_gif_path = f"../outputs/output_{filename_split}.gif"
    create_gif(image_paths, output_gif_path, fps=fps)
