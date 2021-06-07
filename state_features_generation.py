# -*- coding: utf-8 -*-
# author: @peggy4444


from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from spectral_clustering import compute_pressure_features

events=pd.read_excel('104_games_synced.xlsx')


# Create action space

def action_type(row):
    if 'Non attacking pass' in  row['action_name']:
        return 'Pass'
    if 'Attacking pass accurate' in  row['action_name']:
        return 'Pass'
    if 'Diagonal passes' in  row['action_name']:
        return 'Pass'
    if 'Inaccurate extra attacking pass' in  row['action_name']:
        return 'Pass'
    if 'Loss of a ball' in  row['action_name']:
        return 'Badd ball control'
    if 'Ball recovery' in  row['action_name']:
        return 'Ball receiving'
    if 'Pass interceptions' in  row['action_name']:
        return 'Interception'
    if 'Attacking pass inaccurate' in  row['action_name']:
        return 'Pass'
    if 'Successfull dribbling' in  row['action_name']:
        return 'Dribble'
    if 'Foul' in  row['action_name']:
        return 'Foul'
    if 'Accurate crossing from set piece' in  row['action_name']:
        return 'Cross'
    if 'Ball out of the field' in  row['action_name']:
        return 'Ball out'
    if 'Inaccurate set-piece cross' in  row['action_name']:
        return 'Cross'
    if 'Inaccurate blocked cross' in  row['action_name']:
        return 'Cross'
    if 'Cross interception' in  row['action_name']:
        return 'Interception'
    if 'Tackle' in  row['action_name']:
        return 'Tackle'
    if 'Free ball pick up' in  row['action_name']:
        return 'Ball receiving'
    if 'Bad ball control' in  row['action_name']:
        return 'Bad ball control'
    if 'Crosses inaccurate' in  row['action_name']:
        return 'Cross'
    if 'Key interception' in  row['action_name']:
        return 'Interception'
    if 'Extra attacking pass accurate' in  row['action_name']:
        return 'Pass'
    if 'Inaccurate key pass' in  row['action_name']:
        return 'Pass'
    if 'Wide shot' in  row['action_name']:
        return 'Shot'
    if 'Unsuccessfull dribbling' in  row['action_name']:
        return 'Dribble'
    if 'Shot blocked' in  row['action_name']:
        return 'Shot'
    if 'Shots blocked' in  row['action_name']:
        return 'Shot'
    if 'Misplaced crossing from set piece with a shot' in  row['action_name']:
        return 'Shot'
    if 'Accurate crossing from set piece with a shot' in  row['action_name']:
        return 'Shot'
    if 'Clearance' in  row['action_name']:
        return 'Clearance'
    if 'Good interception of goalkeeper' in  row['action_name']:
        return 'Interception'
    if 'Shot on target' in  row['action_name']:
        return 'Shot'
    if 'Crosses accurate' in  row['action_name']:
        return 'Cross'
    if 'Key assist' in  row['action_name']:
        return 'Assist'
    if 'Non attacking pass inaccurate' in  row['action_name']:
        return 'Pass'
    if 'Half time' in  row['action_name']:
        return 'Half time'
    if '2nd half start' in  row['action_name']:
        return '2nd half start'
    if 'Yellow card' in  row['action_name']:
        return 'Yellow card'
    if 'Pass into offside' in  row['action_name']:
        return 'Pass'
    if 'Offside' in  row['action_name']:
        return 'Offside'
    if 'Accurate key pass' in  row['action_name']:
        return 'Pass'
    if 'Shot blocked by field player' in  row['action_name']:
        return 'Shot'
    if 'Goal' in  row['action_name']:
        return 'Goal'
    if 'Accurate crossing from set piece with a goal' in  row['action_name']:
        return 'Cross'
    if 'Non attacking pass inaccurate' in  row['action_name']:
        return 'Pass'
    if 'Match end' in  row['action_name']:
        return 'Match end'
    if 'Bad interception of goalkeeper' in  row['action_name']:
        return 'Interception'
    if 'Team pressing unsuccessful' in  row['action_name']:
        return 'Press'
    if 'Team pressing successful' in  row['action_name']:
        return 'Press'
    if 'Assist' in  row['action_name']:
        return 'Assist'
    if 'Set piece cross with goal' in  row['action_name']:
        return 'Cross'
    if 'Fouling inside the box when not in danger, including handball' in  row['action_name']:
        return 'Foul'
    if 'Extra attacking pass Assist' in  row['action_name']:
        return 'Pass'
    if 'Shot into the bar/post' in  row['action_name']:
        return 'Shot'
    if 'Red card' in  row['action_name']:
        return 'Red card'
    if 'Cross' in  row['action_name']:
        return 'Cross'
    if 'Own goal' in  row['action_name']:
        return 'Own goal'
    if 'Misplaced crossing from set piece with a goal' in  row['action_name']:
        return 'Cross'
    if 'Diagonal pass' in  row['action_name']:
        return 'Pass'
    if 'Pass behind a player' in  row['action_name']:
        return 'Pass'    
    
    return 'Other'


event['action_type']= event.apply (lambda row: action_type(row), axis=1)

event['action_type'].unique()


# Create action results

def action_result(row):
    # return 1 for accurate and successful actions, 0 for unseuccessful and inaccurate actions
    if 'Non attacking pass' in  row['action_name']:
        return '1'
    if 'Attacking pass accurate' in  row['action_name']:
        return '1'
    if 'Diagonal passes' in  row['action_name']:
        return '1'
    if 'Inaccurate extra attacking pass' in  row['action_name']:
        return '0'
    if 'Loss of a ball' in  row['action_name']:
        return '1'
    if 'Ball recovery' in  row['action_name']:
        return '1'
    if 'Pass interceptions' in  row['action_name']:
        return '1'
    if 'Attacking pass inaccurate' in  row['action_name']:
        return '0'
    if 'Successfull dribbling' in  row['action_name']:
        return '1'
    if 'Foul' in  row['action_name']:
        return '1'
    if 'Accurate crossing from set piece' in  row['action_name']:
        return '1'
    if 'Ball out of the field' in  row['action_name']:
        return '1'
    if 'Inaccurate set-piece cross' in  row['action_name']:
        return '0'
    if 'Inaccurate blocked cross' in  row['action_name']:
        return '0'
    if 'Cross interception' in  row['action_name']:
        return '1'
    if 'Tackle' in  row['action_name']:
        return '1'
    if 'Free ball pick up' in  row['action_name']:
        return '1'
    if 'Bad ball control' in  row['action_name']:
        return '1'
    if 'Crosses inaccurate' in  row['action_name']:
        return '0'
    if 'Key interception' in  row['action_name']:
        return '1'
    if 'Extra attacking pass accurate' in  row['action_name']:
        return '1'
    if 'Inaccurate key pass' in  row['action_name']:
        return '0'
    if 'Wide shot' in  row['action_name']:
        return '1'
    if 'Unsuccessfull dribbling' in  row['action_name']:
        return '0'
    if 'Shot blocked' in  row['action_name']:
        return '0'
    if 'Shots blocked' in  row['action_name']:
        return '0'
    if 'Misplaced crossing from set piece with a shot' in  row['action_name']:
        return '0'
    if 'Accurate crossing from set piece with a shot' in  row['action_name']:
        return '1'
    if 'Clearance' in  row['action_name']:
        return '1'
    if 'Good interception of goalkeeper' in  row['action_name']:
        return '1'
    if 'Shot on target' in  row['action_name']:
        return '1'
    if 'Crosses accurate' in  row['action_name']:
        return '1'
    if 'Key assist' in  row['action_name']:
        return '1'
    if 'Non attacking pass inaccurate' in  row['action_name']:
        return '0'
    if 'Half time' in  row['action_name']:
        return '1'
    if '2nd half start' in  row['action_name']:
        return '1'
    if 'Yellow card' in  row['action_name']:
        return '1'
    if 'Pass into offside' in  row['action_name']:
        return '0'
    if 'Offside' in  row['action_name']:
        return '1'
    if 'Accurate key pass' in  row['action_name']:
        return '1'
    if 'Shot blocked by field player' in  row['action_name']:
        return '0'
    if 'Goal' in  row['action_name']:
        return '1'
    if 'Accurate crossing from set piece with a goal' in  row['action_name']:
        return '1'
    if 'Non attacking pass inaccurate' in  row['action_name']:
        return '0'
    if 'Match end' in  row['action_name']:
        return '1'
    if 'Bad interception of goalkeeper' in  row['action_name']:
        return '0'
    if 'Team pressing unsuccessful' in  row['action_name']:
        return '0'
    if 'Team pressing successful' in  row['action_name']:
        return '1'
    if 'Assist' in  row['action_name']:
        return '1'
    if 'Set piece cross with goal' in  row['action_name']:
        return '1'
    if 'Fouling inside the box when not in danger, including handball' in  row['action_name']:
        return '1'
    if 'Extra attacking pass Assist' in  row['action_name']:
        return '1'
    if 'Shot into the bar/post' in  row['action_name']:
        return '1'
    if 'Red card' in  row['action_name']:
        return '1'
    if 'Cross' in  row['action_name']:
        return '1'
    if 'Own goal' in  row['action_name']:
        return '1'
    if 'Misplaced crossing from set piece with a goal' in  row['action_name']:
        return '0'
    if 'Diagonal pass' in  row['action_name']:
        return '1'
    if 'Pass behind a player' in  row['action_name']:
        return '1'
    
    
    return 'Other'

event['action_result']= event.apply (lambda row: action_result(row), axis=1)

event['action_result'].unique()


# Generate time_ramaining features

end_first_half = event[event.half == 1][['match_id','second']].groupby('match_id', as_index=False).max()

end_second_half = event[event.half == 2][['match_id','second']].groupby('match_id', as_index=False).max()

event= pd.merge(event, end_first_half[['match_id','second']].rename(columns={'second':'half_max_second'}), on='match_id')

event= pd.merge(event, end_second_half[['match_id','second']].rename(columns={'second':'2_half_max_second'}), on='match_id')

def time_remaining(row):
    if row['half']==1:
        return int(row['half_max_second']) - int(row['second'])
    if row['half']==2:
        return int(row['2_half_max_second']) - int(row['second'])
    
event['time_remaining'] = event.apply(time_remaining,axis=1)

# Generate angle and distance to goal features

PITCH_LENGTH = 105
PITCH_WIDTH = 68

for side in ['pos', 'pos_dest']:
    # Normalize the X location
    key_x = f'{side}_x'
    event[f'{key_x}_norm'] = event[key_x] / PITCH_LENGTH

    # Normalize the Y location
    key_y = f'{side}_y'
    event[f'{key_y}_norm'] = event[key_y] / PITCH_WIDTH

GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2

for side in ['pos', 'pos_dest']:
    diff_x = GOAL_X - event[f'{side}_x']
    diff_y = abs(GOAL_Y - event[f'{side}_y'])
    event[f'{side}_distance_to_goal'] = np.sqrt(diff_x ** 2 + diff_y ** 2)
    event[f'{side}_angle_to_goal'] = np.divide(diff_x, diff_y, 
                                                    out=np.zeros_like(diff_x), 
                                                    where=(diff_y != 0))

# One-hot actions

def add_action_type_dummies(event):
    return event.merge(pd.get_dummies(event['action_type']), how='left',
                             left_index=True, right_index=True)
event= add_action_type_dummies(event)

event.columns

event_columns= event[['action_id',
       'action_name', 'player_id', 'player_name', 'team_id', 'team_name',
       'position_id', 'position_name','half', 'zone_name','possession_id', 'possession_name', 'possession_team_id',
       'possession_team_name', 'possession_time', 'possession_number','match_id','action_type', 'action_result','time_remaining','pos_distance_to_goal',
       'pos_angle_to_goal', 'pos_dest_distance_to_goal',
       'pos_dest_angle_to_goal']]

# Pressure features
def add_pressure_features(event, clusters):
    pressures= compute_pressure_features(clusters)
    return event.merge(pressure), how='left',
                             left_index=True, right_index=True)
event= add_pressure_features(event, clusters)

# all features

feature_set1= event[['match_id', 'half', 'time_remaining',
                     'pos_distance_to_goal', 'pos_angle_to_goal', 'action_result',
       'Clearance', 'Cross', 'Dribble', 'Foul', 'Goal',
       'Interception', 'Offside', 'Own goal', 'Pass', 'Press','Shot', 'Tackle']]

feature_set2= event[['match_id', 'half', 'time_remaining',
                     'pos_distance_to_goal', 'pos_angle_to_goal', 'action_result',
                     'Assist',
       'Clearance', 'Cross', 'Dribble', 'Foul', 'Goal',
       'Interception', 'Offside', 'Own goal', 'Pass', 'Press','Shot', 'Tackle' 'possessors_locations', 'opponents_locations']]


feature_set3= event[['match_id', 'half', 'time_remaining',
                     'pos_distance_to_goal', 'pos_angle_to_goal', 'action_result',
                     'Clearance', 'Cross', 'Dribble', 'Foul', 'Goal',
       'Interception', 'Offside', 'Own goal', 'Pass', 'Press','Shot', 'Tackle', 'pressure_1', 'pressure_2','pressure_3']]

event_features

event_features.isnull().sum()

event_features=event_features.fillna(0)

event_columns.shape

event_columns.columns


# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

