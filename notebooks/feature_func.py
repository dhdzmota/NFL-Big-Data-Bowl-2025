import numpy as np
import os
import pandas as pd
import scipy.spatial as ss

from sklearn.preprocessing import LabelEncoder

elemental_columns = [
    'nflId',
    'x',
    'y',
    'club',
    's',
    'a',
    'dir',
    'event',
    'weight',
    'position',
    'position_group', 
    'position_subgroup',
    'dis', 
    'o',
    'gameId',
    'playId',
    'frameId',
]


def add_prefix_to_dataframe(df, prefix):
    rename_dict = {col: f'{prefix}_{col}' for col in df.columns}
    df = df.rename(columns=rename_dict)
    return df


def get_path(*args):
    """Util function to get path"""
    ## Function to get the path
    path = os.path.join(*args)
    return path


def add_player_info_to_tracking(tracking_df, players_df, play_df):
    """Merge dataframes information from players and play to tracking_df"""
    play_df = play_df.reset_index()
    tracking_df.loc[tracking_df.nflId.notna(), 'nflId'] = tracking_df[tracking_df.nflId.notna()].nflId.astype('int')
    tracking_w_player_info = tracking_df.merge(players_df, on='nflId', how='left')[elemental_columns]
    tracking_w_player_info = tracking_w_player_info.merge(
        play_df[['gameId', 'playId', 'team_deffinition']].drop_duplicates(['gameId', 'playId']), 
        on=['gameId', 'playId'], how='inner'
    )
    
    tracking_w_player_info['team_redeffinition'] = tracking_w_player_info.apply(
        lambda row: row.team_deffinition.get(row.club), axis=1
    )
    # Filling ball null values:
    tracking_w_player_info.position = tracking_w_player_info.position.fillna('ball')
    tracking_w_player_info.position_group = tracking_w_player_info.position_group.fillna('ball')
    tracking_w_player_info.position_subgroup = tracking_w_player_info.position_subgroup.fillna('ball')
    tracking_w_player_info.loc[tracking_w_player_info.club=='football', 'weight'] =  1.11
    return tracking_w_player_info


def encode_position_tracking_df_w_player_info(tracking_w_player_info_df, train=False, encoders=None):
    if train: 
        lb_player_position = LabelEncoder()
        lb_player_position_group = LabelEncoder()
        lb_player_position_subgroup = LabelEncoder()
        lb_player_position.fit(tracking_w_player_info_df.position.unique())
        lb_player_position_group.fit(tracking_w_player_info_df.position_group.unique())
        lb_player_position_subgroup.fit(tracking_w_player_info_df.position_subgroup.unique())
    if encoders is None:
        encoders = (lb_player_position, lb_player_position_group, lb_player_position_subgroup)
    tracking_w_player_info_df["encoded_position"] = encoders[0].transform(tracking_w_player_info_df.position)
    tracking_w_player_info_df["encoded_position_group"] = encoders[1].transform(tracking_w_player_info_df.position_group)
    tracking_w_player_info_df["encoded_position_subgroup"] = encoders[2].transform(tracking_w_player_info_df.position_subgroup)
    if train:
        return tracking_w_player_info_df, encoders[0], encoders[1], encoders[2]
    return tracking_w_player_info_df


def compute_individual_players_info_for_tracking_df(tracking_w_player_info_df):
    trpi = tracking_w_player_info_df.copy()
    trpi.loc[trpi.fillna(0).dir==90, 'dir'] = 91
    trpi.loc[trpi.fillna(0).dir==270, 'dir'] = 269
    trpi['current_position_influence'] = trpi.apply(influence_function_actual_pos, axis=1)
    trpi.drop(columns=['position', 'position_group', 'position_subgroup'], inplace=True)
    # For each player we can compute their physical values:
    trpi['action'] = trpi.weight * trpi.s * trpi.dis
    trpi['ke'] = 1/2 * trpi.weight * trpi.s ** 2
    trpi['p'] = trpi.weight * trpi.s
    trpi['px'] = trpi.weight * trpi.s * np.cos(np.deg2rad(trpi.dir))
    trpi['py'] = trpi.weight * trpi.s * np.sin(np.deg2rad(trpi.dir))
    trpi['fx'] = trpi.weight * trpi.a * np.cos(np.deg2rad(trpi.dir))
    trpi['fy'] = trpi.weight * trpi.a * np.sin(np.deg2rad(trpi.dir))
    trpi['f'] = trpi.weight * trpi.a
    trpi['work'] = trpi.f * trpi.dis
    return trpi


def influence_function(analysis_unit):
    """Returns the influence function for a specific player"""
    mean_x = analysis_unit.x + 0.5*analysis_unit.s* np.cos(np.deg2rad(analysis_unit.dir))
    mean_y = analysis_unit.y + 0.5*analysis_unit.s* np.sin(np.deg2rad(analysis_unit.dir))
    
    max_speed = 20 #Max speed proposed in paper over the tracking data
    influence_radius = 3  # Influence in yards of the player. (max dis)
    s_rat = (analysis_unit.s/max_speed)**2
    
    sx = (influence_radius + influence_radius * s_rat) / 2
    sy = (influence_radius - influence_radius * s_rat) / 2
    
    diag_cov_comp1 = sx**2*np.cos(np.deg2rad(analysis_unit.dir))**2
    diag_cov_comp2 = sy**2*np.cos(np.deg2rad(analysis_unit.dir))**2
    
    cov_matrix_det = np.cos(np.deg2rad(analysis_unit.dir))**4 * sx ** 2 * sy ** 2
    
    def matrix_multiplication_result(pos_x, pos_y):
        x_comp = sx ** 2 * np.cos(np.deg2rad(analysis_unit.dir)) ** 2 * (pos_x - mean_x) ** 2
        y_comp = sy ** 2 * np.cos(np.deg2rad(analysis_unit.dir)) ** 2 * (pos_y - mean_y) ** 2
        tot_comp = x_comp + y_comp
        return tot_comp
    influence_sign = analysis_unit.team_redeffinition
    
    func = lambda pos_x, pos_y: influence_sign * (1 / np.sqrt(2*np.pi*cov_matrix_det) * np.exp((-1/2)* matrix_multiplication_result(pos_x, pos_y)))
    return func


def influence_function_actual_pos(analysis_unit):
    """Returns the influence for a specific player on its current position"""

    pos_x = analysis_unit.x
    pos_y = analysis_unit.y
    
    mean_x = analysis_unit.x + 0.5*analysis_unit.s* np.cos(np.deg2rad(analysis_unit.dir))
    mean_y = analysis_unit.y + 0.5*analysis_unit.s* np.sin(np.deg2rad(analysis_unit.dir))
    
    max_speed = 20 #Max speed proposed in paper over the tracking data
    influence_radius = 3  # Influence in yards of the player. (max dis)
    s_rat = (analysis_unit.s/max_speed)**2
    
    sx = (influence_radius + influence_radius * s_rat) / 2
    sy = (influence_radius - influence_radius * s_rat) / 2
    
    diag_cov_comp1 = sx**2*np.cos(np.deg2rad(analysis_unit.dir))**2
    diag_cov_comp2 = sy**2*np.cos(np.deg2rad(analysis_unit.dir))**2
    
    cov_matrix_det = np.cos(np.deg2rad(analysis_unit.dir))**4 * sx ** 2 * sy ** 2
    
    x_comp = sx ** 2 * np.cos(np.deg2rad(analysis_unit.dir)) ** 2 * (pos_x - mean_x) ** 2
    y_comp = sy ** 2 * np.cos(np.deg2rad(analysis_unit.dir)) ** 2 * (pos_y - mean_y) ** 2
    matrix_multiplication_result = x_comp + y_comp
    influence_sign = analysis_unit.team_redeffinition
    interaction =  influence_sign * (1 / np.sqrt(2*np.pi*cov_matrix_det) * np.exp((-1/2)* matrix_multiplication_result))
    return interaction

def get_general_and_team_info(tracking_w_player_info):
    operation_dict = {
        'dis':'sum', 
        'ke': 'sum',
        'f': 'sum', 
        'fx': 'sum', 
        'fy': 'sum', 
        'p': 'sum',
        'px': 'sum', 
        'py': 'sum',
        'work': 'sum',
        'action':'sum',
        'x': 'mean',
        'y': 'mean',
        'current_position_influence':'mean',
    }
    general_results = tracking_w_player_info.groupby(['gameId', 'playId', 'frameId']).agg(operation_dict)
    # Update dict
    operation_dict['team_redeffinition'] = 'first'
    
    team_results = tracking_w_player_info[
        tracking_w_player_info.club != 'football'
    ].groupby(
        ['gameId', 'playId', 'frameId', 'club']
    ).agg(
        operation_dict
    )
    
    team_possesion = team_results[
        team_results.team_redeffinition == 1
    ].drop(['team_redeffinition'], axis=1).reset_index(level=-1, drop=True)
    team_defensive = team_results[
        team_results.team_redeffinition == -1
    ].drop(['team_redeffinition'], axis=1).reset_index(level=-1, drop=True)

    team_diff = (
        team_possesion -
        team_defensive
    )
    team_diff = team_diff.rename(
        columns={col: f'{col}__difference' for col in team_diff.columns}
    )
    general_results = general_results.rename(
        columns={col: f'{col}__general' for col in general_results.columns}
    )

    team_possesion = team_possesion.rename(
        columns={col: f'{col}__possesion' for col in team_possesion.columns}
    )
    team_defensive = team_defensive.rename(
        columns={col: f'{col}__defensive' for col in team_defensive.columns}
    )    
    results = pd.concat([
        general_results, 
        team_diff, 
        team_possesion[['x__possesion', 'y__possesion']], 
        team_defensive[['x__defensive', 'y__defensive']], 
    ], axis=1)
    return results


def most_influence_info(tracking_w_player_info_df):
    most_influence_defensive = tracking_w_player_info_df[
        tracking_w_player_info_df.team_redeffinition == -1
    ].sort_values(
        'current_position_influence', ascending=True
    ).groupby(
        ['gameId', 'playId', 'frameId']
    ).agg(
        {'encoded_position': 'first', 'encoded_position_subgroup': 'first', 'current_position_influence': 'first'}
    ).rename(
        columns={
            'encoded_position':'most_influence_encoded_position_defensive', 
            'encoded_position_subgroup': 'most_influence_encoded_position_subgroup_defensive',
            'current_position_influence':'most_influence_defensive',
        }
    )
    
    most_influence_possesion = tracking_w_player_info_df[
        tracking_w_player_info_df.team_redeffinition == 1
    ].sort_values(
        'current_position_influence', ascending=False
    ).groupby(
        ['gameId', 'playId', 'frameId']
    ).agg(
        {'encoded_position': 'first', 'encoded_position_subgroup': 'first', 'current_position_influence': 'first'},
    ).rename(
        columns={
            'encoded_position':'most_influence_encoded_position_possesion', 
            'encoded_position_subgroup': 'most_influence_encoded_position_subgroup_possesion',
            'current_position_influence':'most_influence_possesion',
        }
    )
    most_influence = pd.concat([most_influence_defensive, most_influence_possesion], axis=1)
    return most_influence
        

def get_metrics_distance_between_team_players(df):
    distances = ss.distance_matrix(
        df[df.team_redeffinition == -1][['x','y']],
        df[df.team_redeffinition == 1][['x', 'y']]
    )
    distance_metrics = {
        'distance_between_teams_min': np.min(distances),
        'distance_between_teams_max': np.max(distances),
        'distance_between_teams_mean': np.mean(distances),
        'distance_between_teams_std': np.std(distances),
    }
    return distance_metrics


def general_vs_team_distance_metrics_info(tracking_w_player_info_df):
    general_vs_team_distance_metrics = tracking_w_player_info_df.groupby(['gameId', 'playId', 'frameId']).apply(get_metrics_distance_between_team_players)
    general_vs_team_distance_metrics = pd.json_normalize(general_vs_team_distance_metrics).set_index(general_vs_team_distance_metrics.index)
    return general_vs_team_distance_metrics
