import os
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

from functools import reduce

from feature_func import *
from graph_feature import *
from plot_utils import *

FILEPATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(FILEPATH, '../')
DATA_PATH = get_path(BASE_PATH, "data")
ORIGINAL_DATA_PATH = get_path(DATA_PATH, "original")
GAMES_PATH = get_path(ORIGINAL_DATA_PATH, "games.csv")
PLAYERS_PATH = get_path(ORIGINAL_DATA_PATH, "players.csv")
PLAYS_PATH = get_path(ORIGINAL_DATA_PATH, "plays.csv")
TRACKING_PATH = get_path(ORIGINAL_DATA_PATH, "tracking")
POSITION_DICT_PATH = get_path(ORIGINAL_DATA_PATH, "position_dictionary.csv")

MODEL_PATH = get_path(BASE_PATH, "models")
FINAL_MODEL_PATH = get_path(MODEL_PATH, "final_model")
ENCODER0_PATH = get_path(FINAL_MODEL_PATH, 'lb_player_position.pkl')
ENCODER1_PATH = get_path(FINAL_MODEL_PATH, 'lb_player_position_group.pkl')
ENCODER2_PATH = get_path(FINAL_MODEL_PATH, 'lb_player_position_subgroup.pkl')
ENCODER3_PATH = get_path(FINAL_MODEL_PATH, 'presnap_event_encoder.pkl')
DIMREDU0_PATH = get_path(FINAL_MODEL_PATH, 'pca_model.pkl')
FINAL_MODEL_FILE = get_path(FINAL_MODEL_PATH, "model.pkl")

LIMIT_FRAMES = 50
GAME_ID = 2022110300
PLAY_ID = 2185


def initial_data_modifications(
    tracking_df, players_df, position_dictionary_df
):
    tracking_df.loc[
        tracking_df.event == 'autoevent_ballsnap', 'event'] = 'ball_snap'
    tracking_df.loc[
        tracking_df.event == 'autoevent_passforward', 'event'] = 'pass_forward'
    tracking_df.loc[
        tracking_df.event == 'autoevent_passinterrupted', 'event'] = 'pass_outcome_incomplete'

    players_df['position_group'] = players_df.position.map(
        position_dictionary_df.T.to_dict()).apply(lambda x: x['group'])
    players_df['position_subgroup'] = players_df.position.map(
        position_dictionary_df.T.to_dict()).apply(lambda x: x['subgroup'])

def compute_features(df):
    initial_data_modifications(
        tracking_df, players_df, position_dictionary_df
    )
    """df is partially a tracking_df"""

    after_snap_tracking_df = df[df.frameType == 'AFTER_SNAP']
    before_snap_tracking_df = df[df.frameType != 'AFTER_SNAP']

    cols = ['gameId', 'playId', 'frameId', 'time', 'playDirection', 'event']
    event_occurence = \
    after_snap_tracking_df.drop_duplicates(['gameId', 'playId', 'frameId'])[
        cols]
    current_event_stuff = \
    event_occurence.set_index(['gameId', 'playId', 'frameId'])[
        ['playDirection']]
    current_event_stuff[
        'playDirection'] = current_event_stuff.playDirection.map(
        {'right': 0, 'left': 1})

    reduced_play_df = play_df.merge(current_event_stuff.reset_index(level=-1),
                                    on=['gameId', 'playId'], how='right')
    reduced_play_df = reduced_play_df.merge(
        games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on='gameId',
        how='left')
    reduced_play_df['team_deffinition'] = reduced_play_df.apply(
        lambda x: {x.possessionTeam: 1, x.defensiveTeam: -1}, axis=1)
    reduced_play_df['gameClock_time'] = reduced_play_df.gameClock.str.split(
        ':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

    home_posseses = reduced_play_df['homeTeamAbbr'] == reduced_play_df[
        'possessionTeam']
    visit_posseses = reduced_play_df['visitorTeamAbbr'] == reduced_play_df[
        'possessionTeam']

    home_deffends = reduced_play_df['homeTeamAbbr'] == reduced_play_df[
        'defensiveTeam']
    visit_deffends = reduced_play_df['visitorTeamAbbr'] == reduced_play_df[
        'defensiveTeam']

    reduced_play_df.loc[home_posseses, 'preSnapPossessionScore'] = \
    reduced_play_df[home_posseses].preSnapHomeScore
    reduced_play_df.loc[visit_posseses, 'preSnapPossessionScore'] = \
    reduced_play_df[visit_posseses].preSnapVisitorScore

    reduced_play_df.loc[home_deffends, 'preSnapDefensiveScore'] = \
    reduced_play_df[home_deffends].preSnapHomeScore
    reduced_play_df.loc[visit_deffends, 'preSnapDefensiveScore'] = \
    reduced_play_df[visit_deffends].preSnapVisitorScore

    reduced_play_df.loc[home_posseses, 'preSnapPossessionTeamWinProbability'] = \
    reduced_play_df[home_posseses].preSnapHomeTeamWinProbability
    reduced_play_df.loc[
        visit_posseses, 'preSnapPossessionTeamWinProbability'] = \
    reduced_play_df[visit_posseses].preSnapVisitorTeamWinProbability

    reduced_play_df.loc[home_deffends, 'preSnapDefensiveTeamWinProbability'] = \
    reduced_play_df[home_deffends].preSnapHomeTeamWinProbability
    reduced_play_df.loc[visit_deffends, 'preSnapDefensiveTeamWinProbability'] = \
    reduced_play_df[visit_deffends].preSnapVisitorTeamWinProbability

    reduced_play_df['preSnapScore_difference'] = reduced_play_df[
                                                     'preSnapPossessionScore'] - \
                                                 reduced_play_df[
                                                     'preSnapDefensiveScore']
    reduced_play_df['preSnapTeamWinProbability_difference'] = reduced_play_df[
                                                                  'preSnapPossessionTeamWinProbability'] - \
                                                              reduced_play_df[
                                                                  'preSnapDefensiveTeamWinProbability']

    reduced_play_df[
        'receiverAlignment_0'] = reduced_play_df.receiverAlignment.fillna(
        '0x0').str.split('x').apply(lambda x: x[0]).astype('int')
    reduced_play_df[
        'receiverAlignment_1'] = reduced_play_df.receiverAlignment.fillna(
        '0x0').str.split('x').apply(lambda x: x[1]).astype('int')
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
    reduced_play_df = reduced_play_df[play_features_columns]
    reduced_tracking_w_player_info_df = add_player_info_to_tracking(df,
                                                                    players_df,
                                                                    reduced_play_df)

    reduced_tracking_w_player_info_df = encode_position_tracking_df_w_player_info(
        reduced_tracking_w_player_info_df, encoders=position_encoders
    )
    reduced_tracking_w_player_info_df = compute_individual_players_info_for_tracking_df(
        reduced_tracking_w_player_info_df)

    unique_sample_event_occurence = df[['gameId', 'playId']].drop_duplicates()

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

    event_list_before_snap.loc[~event_list_before_snap.event.astype(str).isin(
        presnap_event_encoder.classes_), 'encoded_event'] = -1
    event_list_before_snap.loc[
        event_list_before_snap.event.astype(str).isin(
            presnap_event_encoder.classes_), 'encoded_event'
    ] = presnap_event_encoder.transform(
        event_list_before_snap[
            event_list_before_snap.event.astype(str).isin(
                presnap_event_encoder.classes_)
        ].event.astype('str')
    )

    # event_list_before_snap['encoded_event'] = presnap_event_encoder.transform(event_list_before_snap.event.astype('str'))

    before_snap_ls_bs = before_snap_tracking_df[
        before_snap_tracking_df.event.isin(['line_set', 'ball_snap'])
    ]
    # There might be some duplicates, with this we remove them.
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

    before_snap_ls_w_player_info = add_player_info_to_tracking(before_snap_ls,
                                                               players_df,
                                                               reduced_play_df)
    before_snap_ls_w_player_info = encode_position_tracking_df_w_player_info(
        before_snap_ls_w_player_info, encoders=position_encoders)
    before_snap_ls_w_player_info = compute_individual_players_info_for_tracking_df(
        before_snap_ls_w_player_info)

    before_snap_bs_w_player_info = add_player_info_to_tracking(before_snap_bs,
                                                               players_df,
                                                               reduced_play_df)
    before_snap_bs_w_player_info = encode_position_tracking_df_w_player_info(
        before_snap_bs_w_player_info, encoders=position_encoders)
    before_snap_bs_w_player_info = compute_individual_players_info_for_tracking_df(
        before_snap_bs_w_player_info)

    general_reduced_tracking_w_player_info_df = get_general_and_team_info(
        reduced_tracking_w_player_info_df)

    line_set_general_reduced_tracking_w_player_info_df = add_prefix_to_dataframe(
        df=get_general_and_team_info(before_snap_ls_w_player_info),
        prefix="line_set"
    ).reset_index(drop=True, level=-1)

    ball_snap_general_reduced_tracking_w_player_info_df = add_prefix_to_dataframe(
        df=get_general_and_team_info(before_snap_bs_w_player_info),
        prefix="ball_snap"
    ).reset_index(drop=True, level=-1)

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

    general_vs_team_distance_metrics_df = general_vs_team_dist_metrics_info(
        reduced_tracking_w_player_info_df
    )
    football_info_df = reduced_tracking_w_player_info_df[
        reduced_tracking_w_player_info_df.club == 'football'].set_index(
        ['gameId', 'playId', 'frameId'])
    not_futball_columns = [
        'nflId',
        'dir',
        'event',
        'weight',
        'encoded_position',
        'o',
        'action',
        'ke',
        'fx',
        'fy',
        'f',
        'px',
        'py',
        'work',
        'club',
        # 'team_deffinition',
        # 'team_redeffinition',
        'encoded_position_group',
        'encoded_position_subgroup',
        'current_position_influence',
    ]
    football_info_df.drop(not_futball_columns, axis=1, inplace=True)
    football_info_df['distance_general'] = (
            (
                        general_reduced_tracking_w_player_info_df.x__general - football_info_df.x) ** 2 +
            (
                        general_reduced_tracking_w_player_info_df.y__general - football_info_df.y) ** 2
    )
    football_info_df['distance_possesion_team'] = (
            (
                        football_info_df.x - general_reduced_tracking_w_player_info_df.x__possesion) ** 2 +
            (
                        football_info_df.y - general_reduced_tracking_w_player_info_df.y__possesion) ** 2
    )
    football_info_df['distance_defensive_team'] = (
            (
                        football_info_df.x - general_reduced_tracking_w_player_info_df.x__defensive) ** 2 +
            (
                        football_info_df.y - general_reduced_tracking_w_player_info_df.y__defensive) ** 2
    )

    all_players_reduced_tracking_positions = reduced_tracking_w_player_info_df[
        reduced_tracking_w_player_info_df.club != 'football'
        ].set_index(['gameId', 'playId', 'frameId'])[
        ['x', 'y', 'encoded_position', 'club']]

    all_players_reduced_tracking_positions['distance_to_football'] = list(
        (football_info_df.x - all_players_reduced_tracking_positions.x) ** 2 +
        (football_info_df.y - all_players_reduced_tracking_positions.y) ** 2
    )

    all_players_reduced_tracking_positions.reset_index(inplace=True)

    all_players_reduced_tracking_positions = all_players_reduced_tracking_positions.merge(
        reduced_play_df[['gameId', 'playId', 'frameId', 'team_deffinition']],
        on=['gameId', 'playId', 'frameId'], how='left'
    )

    all_players_reduced_tracking_positions.loc[
        all_players_reduced_tracking_positions.team_deffinition.notna(), 'team_redeffinition'
    ] = all_players_reduced_tracking_positions.loc[
        all_players_reduced_tracking_positions.team_deffinition.notna()
    ].apply(
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
        inplace=True)
    football_info_df.drop(
        ['football_team_deffinition', 'football_team_redeffinition'], axis=1,
        inplace=True)

    graphs_whole_team = reduced_tracking_w_player_info_df.groupby(
        ['gameId', 'playId', 'frameId']
    ).apply(
        get_graph_with_nodes_attrs, no_negatives=True
    )


    graphs_whole_team_feather_embeddings = graphs_whole_team.apply(
        get_feathergraph_embedding)

    graphs_whole_team_feather_embeddings_df = pd.DataFrame(
        np.concatenate(
            graphs_whole_team_feather_embeddings.to_numpy()).reshape(
            graphs_whole_team_feather_embeddings.shape[0],
            len(graphs_whole_team_feather_embeddings.iloc[0])),
        index=graphs_whole_team_feather_embeddings.index,
        columns=[f'feather_embeddings_{i}' for i in
                 range(len(graphs_whole_team_feather_embeddings.iloc[0]))],
    )

    graphs_whole_team_feather_embeddings_df_reduced = pd.DataFrame(
        pca_model.transform(graphs_whole_team_feather_embeddings_df.fillna(0)),
        index=graphs_whole_team_feather_embeddings_df.index,
        columns=[f'whole_team_feather_embeeding_reduced_{col}' for col in
                 range(pca_model.n_components)]
    )




    # graphs_noline_additional_feather_embeddings_df_reduced = pd.DataFrame(
    #    pca_model3.transform(graphs_noline_additional_feather_embeddings_df.fillna(0)),
    #    index=graphs_noline_additional_feather_embeddings_df.index,
    #    columns = [f'no_line_feather_embeeding_reduced_{col}' for col in range(pca_model3.n_components)]
    # )

    results_df_list = [
        # Initial play context
        reduced_play_df.set_index(['gameId', 'playId', 'frameId']).drop(
            'team_deffinition', axis=1),
        # General tracking information
        general_reduced_tracking_w_player_info_df,
        # line_set_general_reduced_tracking_w_player_info_df,
        # ball_snap_general_reduced_tracking_w_player_info_df,
        # Influence information
        most_influence_info_df,
        # line_set_most_influence_info,
        # ball_snap_most_influence_info,
        # Team and general distance metrics
        general_vs_team_distance_metrics_df,
        # line_set_general_vs_team_distance_metrics_df,
        # ball_snap_general_vs_team_distance_metrics_df,
        # Ball features without bs and ls dfs:
        football_info_df,
        min_distances_to_ball_possesion,
        min_distances_to_ball_defensive,

        # presnap events:
        event_list_before_snap[['encoded_event']],

        # target and direction
        current_event_stuff,

        # Graph features
        # graphs_whole_team_feather_embeddings_df
        graphs_whole_team_feather_embeddings_df_reduced,
        # graphs_noline_additional_feather_embeddings_df
        # graphs_noline_additional_feather_embeddings_df_reduced,
        # graphs_change_bs_ls_feather_embeddings_df_reduced,
    ]
    all_features = reduce(
        lambda left, right: pd.merge(left, right, left_index=True,
                                     right_index=True, how='left'),
        results_df_list)

    return all_features, graphs_whole_team


def get_all_information(test_df):
    features, graphs = compute_features(test_df)
    graphs.name='graphs'
    predictions = model.predict_proba(features)[:, 1]
    information = test_df[test_df.club=='football'].merge(features, on=['gameId', 'playId', 'frameId'], how='right')
    information['model_pred'] = predictions
    information = information.merge(graphs.reset_index(), on=['gameId', 'playId', 'frameId'], how='left')
    information['time_component'] = pd.to_datetime(information.time, format='ISO8601')
    time_component = (
        information.set_index(['gameId','playId', 'frameId']).time_component -
        information.groupby(['gameId', 'playId']).time_component.min()
    ).dt.total_seconds().reset_index()
    information = information.drop(['time_component'], axis=1).merge(time_component.reset_index(), on=['gameId', 'playId', 'frameId'], how='left')
    return information


games_df = pd.read_csv(GAMES_PATH)
players_df = pd.read_csv(PLAYERS_PATH)
play_df = pd.read_csv(PLAYS_PATH)
position_dictionary_df = pd.read_csv(POSITION_DICT_PATH, index_col=0)
tracking_df = pd.concat([
    pd.read_csv(
        get_path(TRACKING_PATH, f)
    )for f in os.listdir(TRACKING_PATH)[-1:]
])

desired_df = tracking_df[
    (tracking_df.gameId==GAME_ID)  &
    (tracking_df.playId==PLAY_ID)
]

with open(ENCODER0_PATH, 'rb') as f:
    lb_player_position = pickle.load(f)

with open(ENCODER1_PATH, 'rb') as f:
    lb_player_position_group = pickle.load(f)

with open(ENCODER2_PATH, 'rb') as f:
    lb_player_position_subgroup = pickle.load(f)

with open(ENCODER3_PATH, 'rb') as f:
    presnap_event_encoder = pickle.load(f)

with open(DIMREDU0_PATH, 'rb') as f:
    pca_model = pickle.load(f)

with open(FINAL_MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

position_encoders = (
    lb_player_position,
    lb_player_position_group,
    lb_player_position_subgroup
)

information = get_all_information(desired_df)

features, graphs  = compute_features(desired_df)
predictions = model.predict_proba(features)[:, 1]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)
topNfeatures = 10
feature_columns = features.columns

args_max = pd.DataFrame(shap_values, columns=feature_columns).abs().T.apply(lambda x: len(feature_columns)-1-np.argsort(np.argsort(x)), axis=0).reset_index(drop=True)
top_features = args_max.iloc[:topNfeatures].T
top_features = top_features.apply(lambda x: feature_columns[x]).rename(columns={col: f'important_feature{col}' for col in top_features.columns}).set_index(features.index)
top_features_values = top_features.apply(
    lambda row: pd.Series(
        features.loc[row.name][[row[col] for col in top_features.columns]].values,
        index=top_features.columns
    ), axis=1
)

top_features_w_val = top_features + '= '+ top_features_values.round(3).astype('string')

limited_top_features = top_features.head(LIMIT_FRAMES)

top_n_features_list =[
    limited_top_features.important_feature0.value_counts(),
    limited_top_features.important_feature1.value_counts(),
    limited_top_features.important_feature2.value_counts(),
    limited_top_features.important_feature3.value_counts(),
    limited_top_features.important_feature4.value_counts(),
    limited_top_features.important_feature5.value_counts(),
    limited_top_features.important_feature6.value_counts(),
    limited_top_features.important_feature7.value_counts(),
    limited_top_features.important_feature8.value_counts(),
    limited_top_features.important_feature9.value_counts(),
]
top_n_features_real = pd.concat(
    top_n_features_list
).groupby(level=0).sum().sort_values(ascending=False)

top_n_features_real = top_n_features_real.head(11)

predictions = information['model_pred']
print(predictions)

# def plot_shap_feature(f, padding=0, limit_frames=LIMIT_FRAMES):
#     features_f = features[f].head(limit_frames)
#     information_f = information.head(limit_frames)
#     # Min max scaling of the feature to plot
#     features_f = (features_f - features_f.min())/(features_f.max()-features_f.min()) - 0.5
#     plt.plot(
#         information_f.time_component.to_list(),
#         (features_f + padding).to_list(),
#         color='k',
#         linewidth=1
#     )
#     plt.scatter(
#         information_f.time_component,
#         (features_f + padding).to_list(),
#         c=pd.DataFrame(shap_values, columns=feature_columns).head(limit_frames)[f].to_list(),
#         s=20,
#         cmap='cool',
#         vmax=pd.DataFrame(shap_values, columns=feature_columns).head(limit_frames)[top_n_features_real.index].max().max(),
#         vmin=pd.DataFrame(shap_values, columns=feature_columns).head(limit_frames)[top_n_features_real.index].min().min(),
#     )
#
# plt.figure(figsize=(10,10))
#
# padding_coef = -2
#
# for i, indx in enumerate(top_n_features_real.index):
#     f = indx
#     plot_shap_feature(f, padding=i*padding_coef)
#
# tick_loc = list(range(0, padding_coef*len(top_n_features_real), padding_coef))
# plt.yticks(tick_loc, top_n_features_real.index)
# plt.xlabel('Seconds after snap')
# plt.title('Most relevant features over time')
# plt.colorbar(label='Shap Values')
# plt.grid(axis='y', linestyle='--')
# plt.savefig('./figures/img-test0.png')
# plt.close()
#
# plt.figure()
# draw_graph(graphs.iloc[0])
# plt.savefig('./figures/img-test.png')
