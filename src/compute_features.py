import os
import pandas as pd
import numpy as np

from functools import reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


from feature_func import (
    get_path,
    add_player_info_to_tracking,
    add_prefix_to_dataframe,
    compute_individual_players_info_for_tracking_df,
    encode_position_tracking_df_w_player_info,
    get_general_and_team_info,
    general_vs_team_dist_metrics_info,
    most_influence_info,
    not_futball_columns,
)
from graph_feature import (
    get_graph_with_nodes_attrs,
    get_feathergraph_embedding,
)


NB_TRACKS = 8
SECONDS = 0.5
RANDOM_SEED = 42

BASE_PATH = os.path.join(os.getcwd(), '../')
DATA_PATH = get_path(BASE_PATH, "data")
ORIGINAL_DATA_PATH = get_path(DATA_PATH, "original")
FINAL_DATA_PATH = get_path(DATA_PATH, "final")

GAMES_PATH = get_path(ORIGINAL_DATA_PATH, "games.csv")
PLAYERS_PATH = get_path(ORIGINAL_DATA_PATH, "players.csv")
PLAYS_PATH = get_path(ORIGINAL_DATA_PATH, "plays.csv")
TRACKING_PATH = get_path(ORIGINAL_DATA_PATH, "tracking")
POSITION_DICT_PATH = get_path(ORIGINAL_DATA_PATH, "position_dictionary.csv")
FINAL_FEATURES_PATH = get_path(FINAL_DATA_PATH, 'final_features.csv')


def read_data(nb_tracks=1):
    "Initial function to read all the datasets from the corresponding paths"
    games_df = pd.read_csv(GAMES_PATH)
    players_df = pd.read_csv(PLAYERS_PATH)
    play_df = pd.read_csv(PLAYS_PATH)
    position_dictionary_df = pd.read_csv(POSITION_DICT_PATH, index_col=0)
    tracking_path_list = [f for f in os.listdir(TRACKING_PATH) if '.csv' in f]
    tracks = tracking_path_list[: nb_tracks]
    tracking_df = pd.concat([
        pd.read_csv(get_path(TRACKING_PATH, f)) for f in tracks
    ])
    return games_df, players_df, play_df, position_dictionary_df, tracking_df


def slight_data_modifications(tracking_df, players_df):
    """Function that fundamentally changes some information of tracking_df and
    players_df."""
    # Change info from tracking_df
    tracking_df.loc[
        tracking_df.event == 'autoevent_ballsnap', 'event'
    ] = 'ball_snap'
    tracking_df.loc[
        tracking_df.event == 'autoevent_passforward', 'event'
    ] = 'pass_forward'
    tracking_df.loc[
        tracking_df.event == 'autoevent_passinterrupted', 'event'
    ] = 'pass_outcome_incomplete'
    # Include info to players_df from constructed al dict.
    players_df['position_group'] = players_df.position.map(position_dictionary_df.T.to_dict()).apply(lambda x: x['group'])
    players_df['position_subgroup'] = players_df.position.map(position_dictionary_df.T.to_dict()).apply(lambda x: x['subgroup'])


def information_after_snap(tracking_df):
    """Function that creates an after snap df and a plays df with the
    corresponding response variable."""
    after_snap_tracking_df = tracking_df[tracking_df.frameType == 'AFTER_SNAP']
    # Now we keep the relevant events.
    after_snap_events = after_snap_tracking_df[
        after_snap_tracking_df.event.notna()]

    # Of the relevant events, we can make a list for each game_id, and play_id
    all_events_reduced = after_snap_events[
        after_snap_events.club == 'football'  # Reduce the possibilities.
        ].groupby(
        ['gameId', 'playId', 'frameType']
    ).agg(
        {'event': lambda x: list(x)}
    ).reset_index()

    # We can now filter the game and plays that have
    # (pass_forward or pass_arrived) with the
    # corresponding pass_outcome_caught.
    plays_with_response_variable = all_events_reduced[
        all_events_reduced.event.apply(
            lambda x: (
                ('pass_forward' in x or 'pass_arrived' in x) and
                ("pass_outcome_caught" in x or "pass_outcome_touchdown" in x)
            )
        )
    ].drop(['frameType', 'event'], axis=1)

    plays_with_response_variable['is_response_variable_candidate'] = 1
    return_dfs = (
        after_snap_tracking_df,
        plays_with_response_variable
    )
    return return_dfs


def get_event_ocurrence_df(after_snap_tracking_df, plays_with_response_variable):
    """Response variable and game_id plays_id that will be defined as the
    #core of the proyect."""
    cols = ['gameId', 'playId', 'frameId', 'time', 'playDirection', 'event']
    # Selecting only the events (including those where nothing happens).
    event_occurence = after_snap_tracking_df.drop_duplicates(
        ['gameId', 'playId', 'frameId']
    )[cols]

    # Where there exists no event, lets fill it with none.
    event_occurence = event_occurence.fillna('None')
    event_occurence.time = pd.to_datetime(
        event_occurence.time, format='ISO8601'
    )

    # Lets get a future event:
    event_occurence['current_time'] = event_occurence.time - pd.Timedelta(
        seconds=SECONDS
    )
    event_occurence['future_event'] = event_occurence.event

    future_event_occurence = event_occurence[
        ['gameId', 'playId', 'frameId', 'current_time', 'future_event']
    ]
    event_occurence = event_occurence.drop(
        ['current_time', 'future_event'], axis=1
    )
    event_occurence = event_occurence.merge(
        future_event_occurence[
            ['current_time', 'future_event', 'playId', 'gameId']
        ],
        left_on=['time', 'playId', 'gameId'],
        right_on=['current_time', 'playId', 'gameId'],
        how='left'
    )
    event_occurence['is_pass_forward'] = (
            event_occurence.future_event == 'pass_forward'
    ).astype('int')

    event_occurence = event_occurence.merge(
        plays_with_response_variable,
        on=['gameId', 'playId'],
        how='left'
    )

    event_occurence.loc[
        (event_occurence.is_pass_forward == 1) &
        (event_occurence.is_response_variable_candidate == 1), 'target'
    ] = 1
    event_occurence['target'] = event_occurence['target'].fillna(0)
    return event_occurence


def get_core_dataset(event_occurence):
    """Get the core dataset to train"""
    # Selecting only the most important envents events.
    relevant_event_occurence = event_occurence[
        event_occurence.future_event != 'None'].dropna()
    # Selecting some of the non-important envents events.
    non_relevant_event_occurence = event_occurence[
        event_occurence.future_event == 'None'
    ].sample(
        random_state=RANDOM_SEED, n=relevant_event_occurence.shape[0] // 3
    )
    # Merge all information in one dataframe. This will be the core of the proyect.
    sample_event_occurence = pd.concat(
        [relevant_event_occurence, non_relevant_event_occurence]
    )
    return sample_event_occurence


def get_current_event_stuff(reorgainzed_sample_event_occurence):
    current_event_stuff = reorgainzed_sample_event_occurence.set_index(
        ['gameId', 'playId', 'frameId'])[['playDirection', 'target']]
    current_event_stuff[
        'playDirection'] = current_event_stuff.playDirection.map(
        {'right': 0, 'left': 1})
    return current_event_stuff


def get_reduced_play_df(reorgainzed_sample_event_occurence, play_df, games_df):
    reduced_play_df = reorgainzed_sample_event_occurence[
        ['gameId', 'playId', 'frameId']
    ].merge(
        play_df,
        on=['gameId', 'playId'],
        how='left'
    )
    reduced_play_df = reduced_play_df.merge(
        games_df[
            ['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']
        ],
        on='gameId',
        how='inner'
    )

    reduced_play_df['team_deffinition'] = reduced_play_df.apply(
        lambda x: {x.possessionTeam: 1, x.defensiveTeam: -1}, axis=1
    )
    reduced_play_df['gameClock_time'] = reduced_play_df.gameClock.str.split(
        ':'
    ).apply(lambda x: int(x[0]) * 60 + int(x[1]))

    home_posseses = reduced_play_df['homeTeamAbbr'] == reduced_play_df[
        'possessionTeam'
    ]
    visit_posseses = reduced_play_df['visitorTeamAbbr'] == reduced_play_df[
        'possessionTeam'
    ]

    home_deffends = reduced_play_df['homeTeamAbbr'] == reduced_play_df[
        'defensiveTeam'
    ]
    visit_deffends = reduced_play_df['visitorTeamAbbr'] == reduced_play_df[
        'defensiveTeam'
    ]
    psPS = 'preSnapPossessionScore'
    psDS = 'preSnapDefensiveScore'
    reduced_play_df.loc[home_posseses, psPS] = reduced_play_df[
        home_posseses
    ].preSnapHomeScore
    reduced_play_df.loc[visit_posseses, psPS] = reduced_play_df[
        visit_posseses
    ].preSnapVisitorScore

    reduced_play_df.loc[home_deffends, psDS] = reduced_play_df[
        home_deffends
    ].preSnapHomeScore
    reduced_play_df.loc[visit_deffends, psDS] = reduced_play_df[
        visit_deffends
    ].preSnapVisitorScore
    psPTWP = 'preSnapPossessionTeamWinProbability'
    psDTWP = 'preSnapDefensiveTeamWinProbability'
    reduced_play_df.loc[home_posseses, psPTWP] = reduced_play_df[
        home_posseses
    ].preSnapHomeTeamWinProbability
    reduced_play_df.loc[visit_posseses, psPTWP] = reduced_play_df[
        visit_posseses
    ].preSnapVisitorTeamWinProbability

    reduced_play_df.loc[home_deffends, psDTWP] = reduced_play_df[
        home_deffends].preSnapHomeTeamWinProbability
    reduced_play_df.loc[visit_deffends, psDTWP] = reduced_play_df[
        visit_deffends].preSnapVisitorTeamWinProbability

    reduced_play_df['preSnapScore_difference'] = (
            reduced_play_df[psPS] - reduced_play_df[psDS]
    )
    reduced_play_df['preSnapTeamWinProbability_difference'] = (
            reduced_play_df[psPTWP] - reduced_play_df[psDTWP]
    )

    reduced_play_df['receiverAlignment_0'] = (
        reduced_play_df.receiverAlignment.fillna(
            '0x0'
        ).str.split('x').apply(lambda x: x[0]).astype('int')
    )
    reduced_play_df['receiverAlignment_1'] = (
        reduced_play_df.receiverAlignment.fillna(
            '0x0'
        ).str.split('x').apply(lambda x: x[1]).astype('int')
    )
    play_features_columns = [
        'gameId',  # Id column
        'playId',  # Id column
        'frameId',  # Id column
        'quarter',
        'down',
        'yardsToGo',
        'yardlineNumber',
        'absoluteYardlineNumber',
        'preSnapScore_difference',
        'preSnapTeamWinProbability_difference',
        'expectedPoints',
        'playClockAtSnap',
        'receiverAlignment_0',
        'receiverAlignment_1',
        'gameClock_time',
        'team_deffinition',
    ]
    reduced_play_df = reduced_play_df[
        play_features_columns
    ].set_index(
        ['gameId', 'playId', 'frameId']
    )
    return reduced_play_df


def get_reduced_tracking_w_player_info(
    tracking_df,
    reorgainzed_sample_event_occurence,
    players_df,
    reduced_play_df
):
    reduced_tracking_df = tracking_df.merge(
        reorgainzed_sample_event_occurence[
            ['gameId', 'playId', 'frameId', 'target']
        ],
        on=['gameId', 'playId', 'frameId'],
        how='right',
    )
    reduced_tracking_w_player_info_df = add_player_info_to_tracking(
        reduced_tracking_df, players_df, reduced_play_df
    )
    ds = encode_position_tracking_df_w_player_info(
        reduced_tracking_w_player_info_df, train=True, encoders=None
    )
    reduced_tracking_w_player_info_df = ds[0]
    lb_player_position = ds[1]
    lb_player_position_group = ds[2]
    lb_player_position_subgroup = ds[3]
    position_encoders = (
        lb_player_position,
        lb_player_position_group,
        lb_player_position_subgroup
    )

    rtwpi_df = compute_individual_players_info_for_tracking_df(
        reduced_tracking_w_player_info_df
    )
    return rtwpi_df, position_encoders

def get_unique_gameId_playId(df):
    unique_df = df[
        ['gameId', 'playId']
    ].drop_duplicates()
    return unique_df


def compute_before_snap_stuff(
    tracking_df,
    unique_sample_event_occurence,
    players_df,
    reduced_play_df
):
    before_snap_tracking_df = tracking_df[
        tracking_df.frameType != 'AFTER_SNAP'
    ]

    event_list_before_snap = before_snap_tracking_df[
        before_snap_tracking_df.club == 'football'
        ].sort_values(['gameId', 'playId', 'frameId']).groupby(
        ['gameId', 'playId']
    ).event.apply(
        lambda x: list(x.dropna())
    )
    event_list_before_snap = event_list_before_snap.reset_index().merge(
        unique_sample_event_occurence, on=['gameId', 'playId'], how='right'
    ).set_index(['gameId', 'playId'])

    presnap_event_encoder = LabelEncoder()
    presnap_event_encoder.fit(
        event_list_before_snap.event.astype('str').unique()
    )
    event_list_before_snap['encoded_event'] = presnap_event_encoder.transform(
        event_list_before_snap.event.astype('str')
    )

    before_snap_ls_bs = before_snap_tracking_df[
        before_snap_tracking_df.event.isin(['line_set', 'ball_snap'])
    ]

    before_snap_ls_bs = before_snap_ls_bs.groupby(
        ['gameId', 'playId', 'event', 'displayName']
    ).first().reset_index()

    before_snap_ls = before_snap_ls_bs[
        before_snap_ls_bs.event == 'line_set'
        ].merge(
        unique_sample_event_occurence, on=['gameId', 'playId'],
        how='inner'
    )

    before_snap_bs = before_snap_ls_bs[
        before_snap_ls_bs.event == 'ball_snap'
        ].merge(
        unique_sample_event_occurence, on=['gameId', 'playId'],
        how='inner'
    )

    BS_ls_w_player_info = add_player_info_to_tracking(
        before_snap_ls,
        players_df,
        reduced_play_df
    )
    BS_ls_w_player_info = encode_position_tracking_df_w_player_info(
        BS_ls_w_player_info, encoders=position_encoders
    )
    BS_ls_w_player_info = compute_individual_players_info_for_tracking_df(
        BS_ls_w_player_info
    )

    BS_bs_w_player_info = add_player_info_to_tracking(
        before_snap_bs,
        players_df,
        reduced_play_df
    )
    BS_bs_w_player_info = encode_position_tracking_df_w_player_info(
        BS_bs_w_player_info, encoders=position_encoders
    )
    BS_bs_w_player_info = compute_individual_players_info_for_tracking_df(
        BS_bs_w_player_info
    )

    return event_list_before_snap, BS_ls_w_player_info, BS_bs_w_player_info


def get_general_reduced_tracking_w_player_info(
    reduced_tracking_w_player_info_df,
    before_snap_ls_w_player_info,
    before_snap_bs_w_player_info
):
    grtwpi_df = get_general_and_team_info(
        reduced_tracking_w_player_info_df
    )

    line_set_grtwpi_df= add_prefix_to_dataframe(
        df=get_general_and_team_info(before_snap_ls_w_player_info),
        prefix="line_set"
    ).reset_index(drop=True, level=-1)

    ball_snap_grtwpi_df = add_prefix_to_dataframe(
        df=get_general_and_team_info(before_snap_bs_w_player_info),
        prefix="ball_snap"
    ).reset_index(drop=True, level=-1)
    return grtwpi_df, line_set_grtwpi_df, ball_snap_grtwpi_df


def get_most_influence_info(
    reduced_tracking_w_player_info_df,
    before_snap_ls_w_player_info,
    before_snap_bs_w_player_info
):
    most_influence_info_df = most_influence_info(
        reduced_tracking_w_player_info_df)

    line_set_most_influence_info = add_prefix_to_dataframe(
        df=most_influence_info(before_snap_ls_w_player_info),
        prefix="line_set"
    ).reset_index(drop=True, level=-1)

    ball_snap_most_influence_info = add_prefix_to_dataframe(
        df=most_influence_info(before_snap_bs_w_player_info),
        prefix="ball_snap"
    ).reset_index(drop=True, level=-1)

    return (
        most_influence_info_df,
        line_set_most_influence_info,
        ball_snap_most_influence_info
    )


def get_football_info(
    reduced_tracking_w_player_info_df,
    general_reduced_tracking_w_player_info_df,
):
    grtwpi_df = general_reduced_tracking_w_player_info_df
    football_info_df = reduced_tracking_w_player_info_df[
        reduced_tracking_w_player_info_df.club == 'football'
    ].set_index(
        ['gameId', 'playId', 'frameId']
    )
    football_info_df.drop(not_futball_columns, axis=1, inplace=True)
    football_info_df['distance_general'] = (
        (grtwpi_df.x__general - football_info_df.x) ** 2 +
        (grtwpi_df.y__general - football_info_df.y) ** 2
    )
    football_info_df['distance_possesion_team'] = (
        (football_info_df.x - grtwpi_df.x__possesion) ** 2 +
        (football_info_df.y - grtwpi_df.y__possesion) ** 2
    )
    football_info_df['distance_defensive_team'] = (
            (football_info_df.x - grtwpi_df.x__defensive) ** 2 +
            (football_info_df.y - grtwpi_df.y__defensive) ** 2
    )

    aprtp = reduced_tracking_w_player_info_df[
        reduced_tracking_w_player_info_df.club != 'football'
        ].set_index(['gameId', 'playId', 'frameId'])[
        ['x', 'y', 'encoded_position', 'club']]

    aprtp['distance_to_football'] = list(
        (football_info_df.x - aprtp.x) ** 2 +
        (football_info_df.y - aprtp.y) ** 2
    )
    aprtp.reset_index(inplace=True)

    all_players_reduced_tracking_positions = aprtp.merge(
        reduced_play_df.reset_index()[
            ['gameId', 'playId', 'frameId', 'team_deffinition']],
        on=['gameId', 'playId', 'frameId'], how='left'
    )

    all_players_reduced_tracking_positions[
        'team_redeffinition'] = all_players_reduced_tracking_positions.apply(
        lambda row: row.team_deffinition[row.club], axis=1
    )

    min_distances_to_ball = all_players_reduced_tracking_positions.sort_values(
        ['gameId', 'playId', 'frameId', 'distance_to_football']
    ).drop_duplicates(
        ['gameId', 'playId', 'frameId', 'team_redeffinition'], keep='first'
    )

    min_distances_to_ball_defensive = min_distances_to_ball[
        min_distances_to_ball.team_redeffinition == -1
        ].set_index(['gameId', 'playId', 'frameId'])[
        ['encoded_position', 'distance_to_football']]

    min_distances_to_ball_possesion = min_distances_to_ball[
        min_distances_to_ball.team_redeffinition == 1
        ].set_index(['gameId', 'playId', 'frameId'])[
        ['encoded_position', 'distance_to_football']]
    min_distances_to_ball_defensive.rename(
        columns={col: f'min_{col}_defensive' for col in
                 min_distances_to_ball_defensive.columns}, inplace=True
    )
    min_distances_to_ball_possesion.rename(
        columns={col: f'min_{col}_possesion' for col in
                 min_distances_to_ball_possesion.columns}, inplace=True
    )
    football_info_df.rename(
        columns={col: f'football_{col}' for col in football_info_df.columns},
        inplace=True
    )
    football_info_df.drop(
        ['football_team_deffinition', 'football_team_redeffinition'], axis=1,
        inplace=True
    )

    return (
        football_info_df,
        min_distances_to_ball_possesion,
        min_distances_to_ball_defensive
    )


def get_graph_whole_team(reduced_tracking_w_player_info_df):
    graphs_whole_team = reduced_tracking_w_player_info_df.groupby(
        ['gameId', 'playId', 'frameId']
    ).apply(
        get_graph_with_nodes_attrs, no_negatives=True
    )
    return graphs_whole_team


def get_embeddings_w_dim_red_graph(graphs_whole_team):
    graphs_whole_team_feather_embeddings = graphs_whole_team.apply(
        get_feathergraph_embedding
    )

    graphs_whole_team_feather_embeddings_df = pd.DataFrame(
        np.concatenate(
            graphs_whole_team_feather_embeddings.to_numpy()).reshape(
            graphs_whole_team_feather_embeddings.shape[0],
            len(graphs_whole_team_feather_embeddings.iloc[0])),
        index=graphs_whole_team_feather_embeddings.index,
        columns=[f'feather_embeddings_{i}' for i in
                 range(len(graphs_whole_team_feather_embeddings.iloc[0]))],
    )
    pca_model = PCA(n_components=1000)
    pca_model.fit(graphs_whole_team_feather_embeddings_df.fillna(0))
    real_n_components = (abs(np.gradient(
        np.log(pca_model.explained_variance_))) > 0.1).sum()
    pca_model = PCA(n_components=real_n_components * 2)
    pca_model.fit(graphs_whole_team_feather_embeddings_df.fillna(0))

    graphs_whole_team_feather_embeddings_df_reduced = pd.DataFrame(
        pca_model.transform(graphs_whole_team_feather_embeddings_df.fillna(0)),
        index=graphs_whole_team_feather_embeddings_df.index,
        columns=[f'whole_team_feather_embeeding_reduced_{col}' for col in
                 range(pca_model.n_components)]
    )
    return graphs_whole_team_feather_embeddings_df_reduced, pca_model


def feature_computation():
    pass



if __name__ == '__main__':
    # We are using ds as placeholder for data
    ds = read_data(nb_tracks=1)
    games_df, players_df, play_df, position_dictionary_df, tracking_df = ds
    slight_data_modifications(tracking_df, players_df)
    ds = information_after_snap(tracking_df)
    after_snap_tracking_df, plays_with_response_variable = ds
    event_occurence = get_event_ocurrence_df(
        after_snap_tracking_df, plays_with_response_variable
    )
    sample_event_occurence = get_core_dataset(event_occurence)
    reorgainzed_sample_event_occurence = sample_event_occurence.sample(
        frac=1, random_state=RANDOM_SEED
    )
    current_event_stuff = get_current_event_stuff(
        reorgainzed_sample_event_occurence
    )
    reduced_play_df = get_reduced_play_df(
        reorgainzed_sample_event_occurence, play_df, games_df
    )

    ds = get_reduced_tracking_w_player_info(
        tracking_df,
        reorgainzed_sample_event_occurence,
        players_df,
        reduced_play_df
    )
    reduced_tracking_w_player_info_df, position_encoders = ds

    unique_sample_event_occurence = get_unique_gameId_playId(
        sample_event_occurence
    )
    ds = compute_before_snap_stuff(
        tracking_df,
        unique_sample_event_occurence,
        players_df,
        reduced_play_df
    )
    event_list_before_snap, BS_ls_w_player_info, BS_bs_w_player_info = ds
    before_snap_ls_w_player_info = BS_ls_w_player_info
    before_snap_bs_w_player_info = BS_bs_w_player_info

    ds = get_general_reduced_tracking_w_player_info(
        reduced_tracking_w_player_info_df,
        before_snap_ls_w_player_info,
        before_snap_bs_w_player_info
    )
    general_reduced_tracking_w_player_info_df =  ds[0]
    #line_set_general_reduced_tracking_w_player_info_df = ds[1]
    #ball_snap_general_reduced_tracking_w_player_info_df = ds[2]

    ds = get_most_influence_info(
        reduced_tracking_w_player_info_df,
        before_snap_ls_w_player_info,
        before_snap_bs_w_player_info
    )
    most_influence_info_df = ds[0]
    #line_set_most_influence_info = ds[1]
    #ball_snap_most_influence_info = ds[2]

    general_vs_team_distance_metrics_df = general_vs_team_dist_metrics_info(
        reduced_tracking_w_player_info_df
    )

    ds = get_football_info(
        reduced_tracking_w_player_info_df,
        general_reduced_tracking_w_player_info_df,
    )
    football_info_df = ds[0]
    min_distances_to_ball_possesion = ds[1]
    min_distances_to_ball_defensive = ds[2]

    graphs_whole_team = get_graph_whole_team(reduced_tracking_w_player_info_df)
    ds = get_embeddings_w_dim_red_graph(graphs_whole_team)
    graphs_whole_team_feather_embeddings_df_reduced = ds[0]
    pca_model = ds[1]

    results_df_list = [
        # Initial play context
        reduced_play_df.drop('team_deffinition', axis=1),
        # General tracking information
        general_reduced_tracking_w_player_info_df,
        # Influence information
        most_influence_info_df,
        # Team and general distance metrics
        general_vs_team_distance_metrics_df,
        # Ball features without bs and ls dfs:
        football_info_df,
        min_distances_to_ball_possesion,
        min_distances_to_ball_defensive,
        # presnap events:
        event_list_before_snap[['encoded_event']],
        # target and direction
        current_event_stuff,
        # Graph features
        graphs_whole_team_feather_embeddings_df_reduced,
    ]

    all_features = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how='left'
        ), results_df_list
    )

    all_features.to_csv(FINAL_FEATURES_PATH)
