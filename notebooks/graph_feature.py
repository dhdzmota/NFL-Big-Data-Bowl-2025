import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial as ss
from sklearn.preprocessing import scale

def angle_matrix(x, y, p=2, threshold=1000000):
  x = np.asarray(x)
  y = np.asarray(y)
  x = x[:,np.newaxis,:]
  y = y[np.newaxis,:,:]
  x_y = y-x
  arctan_vals = np.arctan2(x_y[:,:,1], x_y[:,:,0])*180/np.pi
  arctan_vals[arctan_vals < 0] = arctan_vals[arctan_vals < 0] + 360
  return arctan_vals


def get_analysis_unit_without_ball(analysis_unit):
  analysis_unit_without_ball = analysis_unit[analysis_unit.club!='football']
  return analysis_unit_without_ball


def get_look_matrix(analysis_unit, not_ball=True):
  analysis_unit = analysis_unit.copy()
  if not_ball:
    analysis_unit = get_analysis_unit_without_ball(analysis_unit)
  # Get the analysis unit without the ball.

  # Get the angle matrix, depending in positions.
  angles_according_to_position = angle_matrix(
      analysis_unit[['x', 'y']],
      analysis_unit[['x', 'y']]
      )
  # Compare if positions are aligned to orientation or direction.
  angles_according_to_position_o = angles_according_to_position - np.asarray(analysis_unit.o)
  angles_according_to_position_dir = angles_according_to_position - np.asarray(analysis_unit.dir)
  # Normalize the direction with cosine
  # (if aligned then 1, if not then 0, and any values between that, also negative alignment is possible).
  look_o = np.cos(np.deg2rad(angles_according_to_position_o)) # Relevance depending in direction
  look_dir = np.cos(np.deg2rad(angles_according_to_position_dir))
  # Sum the contribution of both alingments, at 50%50.
  look = (look_o + look_dir)/2
  look = np.nan_to_num(look, nan=0)
  look = np.matrix.transpose(look)
  return look


def get_inverse_distance_matrix_weighted_by_look(analysis_unit, not_ball=False):
  inverse_distance_matrix = get_inverse_distance_matrix(analysis_unit, not_ball=not_ball, power=1)
  look = get_look_matrix(analysis_unit, not_ball=not_ball)
  #inverse_distance_matrix_weighted = inverse_distance_matrix + 10*look
  inverse_distance_matrix_weighted = inverse_distance_matrix*look
  return inverse_distance_matrix_weighted


def get_team_positions(analysis_unit, not_ball=True):
  analysis_unit = analysis_unit.copy()
  if not_ball:
    analysis_unit = get_analysis_unit_without_ball(analysis_unit)
  pos = analysis_unit[['x', 'y']].reset_index(drop=True)
  pos['xy'] = pos.apply(lambda x: np.array([x.x, x.y]), axis=1)
  pos = pos['xy'].to_dict()
  return pos


def get_inverse_distance_matrix(analysis_unit, power=1, not_ball=True):
  analysis_unit = analysis_unit.copy()
  if not_ball:
    analysis_unit = get_analysis_unit_without_ball(analysis_unit)
  # Get distance matrix
  unit_distance_matrix = ss.distance_matrix(
      analysis_unit[['x', 'y']],
      analysis_unit[['x', 'y']])
  # Divide 1 over the values by a power
  inverse_distance_matrix = 1/unit_distance_matrix**(power)
  # Replace infinite with 0
  inverse_distance_matrix[inverse_distance_matrix==np.inf] = 0
  inverse_distance_matrix = np.nan_to_num(inverse_distance_matrix, nan=0)
  return inverse_distance_matrix


def get_team_graph(analysis_unit, not_ball=False):
  matrix = get_inverse_distance_matrix_weighted_by_look(analysis_unit, not_ball=not_ball)
  G = nx.from_numpy_array(matrix, create_using=nx.MultiDiGraph()) # Must be multidigraph
  return G


def get_team_graph_no_negative(analysis_unit, not_ball=False):
  matrix = get_inverse_distance_matrix_weighted_by_look(analysis_unit, not_ball=not_ball)
  matrix[matrix<0] = 0
  G = nx.from_numpy_array(matrix, create_using=nx.MultiDiGraph()) # Must be multidigraph
  return G


def get_graph_with_nodes_attrs(analysis_unit, not_ball=False, no_negatives=False):
  if no_negatives:
    g = get_team_graph_no_negative(analysis_unit, not_ball=not_ball)
  else:
    g = get_team_graph(analysis_unit, not_ball=False)

  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).action.to_dict(), 'action')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).ke.to_dict(), 'ke')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).p.to_dict(), 'p')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).f.to_dict(), 'f')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).work.to_dict(), 'work')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).team_redeffinition.fillna(0).to_dict(), 'team_redeffinition')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).x.to_dict(), 'pos_x')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).y.to_dict(), 'pos_y')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).s.to_dict(), 'speed')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).a.to_dict(), 'acceleration')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).dis.to_dict(), 'distance')
  nx.set_node_attributes(g, analysis_unit.reset_index(drop=True).encoded_position.to_dict(), 'encoded_position')
  return g


def get_laplacian_matrix(g, weight='weight'):
    # Laplacian = adjacency - degree
    adj_matrix = nx.adjacency_matrix(g, weight=weight).toarray()
    degree_matrix = np.diag(np.array((g.degree(weight=weight)))[:,1])
    laplacian_matrix = adj_matrix - degree_matrix
    return laplacian_matrix


# Embedding fgsd:
def get_fgsd_embedding(g, replicability_nb=3):
  laplacian_matrix = get_laplacian_matrix(g, weight='weight') 
  inverted_laplacian_matrix = np.linalg.pinv(laplacian_matrix)
  ones = np.ones(laplacian_matrix.shape[0])
  S = np.outer(np.diag(inverted_laplacian_matrix), ones) + np.outer(ones, np.diag(inverted_laplacian_matrix)) - 2 * inverted_laplacian_matrix
  hist, _ = np.histogram(S.flatten(), bins=22*replicability_nb)
  hist = hist/hist.sum()
  return hist


def get_laplacian_embedding(g, replicability_nb=2):
  laplacian_matrix = get_laplacian_matrix(g, weight='weight') # Laplacian = adjacency - degree
  hist, _ = np.histogram(laplacian_matrix.flatten(), bins=22*replicability_nb)
  hist = hist/hist.sum()
  return hist


def get_adjacent_embedding(g, replicability_nb=2):
  laplacian_matrix = nx.adjacency_matrix(g, weight='weight').toarray() # Laplacian = adjacency - degree
  hist, _ = np.histogram(laplacian_matrix.flatten(), bins=22*replicability_nb)
  hist = hist/hist.sum()
  return hist


# Embedding feather
def get_feathergraph_embedding(g):
  adjacency_matrix = nx.adjacency_matrix(g, weight='weight').toarray()
  degree_inverse = np.diag(1/np.array((g.degree(weight='weight')))[:,1])
  normalized_adjacency_matrix = degree_inverse.dot(adjacency_matrix)

  node_features = pd.DataFrame(
      dict(g.nodes(data=True))
  ).T

  scaled_node_features = pd.DataFrame(
      scale(node_features, axis=0),
      index=node_features.index,
      columns=node_features.columns
  )
  #scaled_node_features = node_features

  theta = np.linspace(np.pi//2, 2*np.pi+np.pi//2, 360//24)
  outer_scaled_node_features = np.outer(scaled_node_features, theta).reshape(g.number_of_nodes(), -1)
  outer_scaled_node_features_transformed = np.concatenate([np.cos(outer_scaled_node_features), np.sin(outer_scaled_node_features)], axis=1)
  modified_outer_scaled_node_features_transformed = outer_scaled_node_features_transformed.copy()

  feature_blocks = []
  for _ in range(5):
    modified_outer_scaled_node_features_transformed = normalized_adjacency_matrix.dot(modified_outer_scaled_node_features_transformed)
    feature_blocks.append(modified_outer_scaled_node_features_transformed)

  feature_blocks = np.concatenate(feature_blocks, axis=1)
  feather_feature_blocks =  np.mean(feature_blocks, axis=0)
  return feather_feature_blocks


def compute_graph_structure_features(g):
  gnew = g.copy()
  additional_network_features = pd.DataFrame()
  additional_network_features['betweenness_centrality'] = nx.centrality.betweenness_centrality(g, weight='weight')
  additional_network_features['percolation_centrality_action'] = nx.centrality.percolation_centrality(g, weight='weight', attribute='action')
  additional_network_features['percolation_centrality_ke'] = nx.centrality.percolation_centrality(g, weight='weight', attribute='ke')
  additional_network_features['percolation_centrality_p'] = nx.centrality.percolation_centrality(g, weight='weight', attribute='p')
  additional_network_features['percolation_centrality_f'] = nx.centrality.percolation_centrality(g, weight='weight', attribute='f')
  additional_network_features['percolation_centrality_work'] = nx.centrality.percolation_centrality(g, weight='weight', attribute='work')
  additional_network_features['betweeness_residual_action_percolation'] = (
      additional_network_features.betweenness_centrality - additional_network_features.percolation_centrality_action
  )
  additional_network_features['betweeness_residual_ke_percolation'] = (
      additional_network_features.betweenness_centrality - additional_network_features.percolation_centrality_ke
  )
  additional_network_features['betweeness_residual_p_percolation'] = (
      additional_network_features.betweenness_centrality - additional_network_features.percolation_centrality_p
  )  
  additional_network_features['betweeness_residual_f_percolation'] = (
      additional_network_features.betweenness_centrality - additional_network_features.percolation_centrality_f
  )
  additional_network_features['betweeness_residual_work_percolation'] = (
      additional_network_features.betweenness_centrality - additional_network_features.percolation_centrality_work
  )
  additional_network_features['load_centrality'] = nx.centrality.load_centrality(g, weight='weight')
  additional_network_features['harmonic_centrality'] = nx.centrality.harmonic_centrality(g, distance='weight')
  additional_network_features['average_neighbor_degree'] = nx.assortativity.average_neighbor_degree(g, weight='weight')
  additional_network_features['average_degree_connectivity'] = nx.assortativity.average_degree_connectivity(g, weight='weight')
  additional_network_features['pagerank'] = nx.pagerank(g, weight='weight')
    
  additional_network_features.drop(
      ['percolation_centrality_action',
      'percolation_centrality_ke',
      'percolation_centrality_p',
      'percolation_centrality_f',
      'percolation_centrality_work'], axis=1, 
      inplace=True
  )
  additional_network_features = additional_network_features.to_dict()
  for key, value in additional_network_features.items():
      nx.set_node_attributes(gnew, value, key)
  return gnew
