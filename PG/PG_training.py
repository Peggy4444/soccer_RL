# -*- coding: utf-8 -*-

# author: @peggy4444


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')



from google.colab import files
uploaded = files.upload()

import pandas as pd
data=pd.read_csv('off_line_data_to_pg.csv')



class PG:
  def __init__(self, path=None):
    self.state_shape=(20,) # the state space for state type (III), change it to 17, and 61 for other proposed state types.
    self.action_shape=4 # the action space
    self.gamma=0.99 # decay rate of past observations
    self.alpha=1e-4 # learning rate in the policy gradient
    self.learning_rate=0.01 # learning rate in deep learning
    
    if not path:
      self.model=self._create_model() #build model
    else:
      self.model=self.load_model(path) #import model

    # record observations
    self.states=[]
    self.gradients=[] 
    self.rewards=[]
    self.target_probs=[]
    self.behave_probs=[]
    self.discounted_rewards=[]
    self.total_behave_rewards=[]
  
  def _create_model(self):
    ''' builds the model using keras'''
    model=Sequential()

    # input shape is of observations
    model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
    #model.add(Dropout(0.5))
    # introduce a relu layer 
    model.add(Dense(12, activation="relu"))
    #model.add(Dropout(0.5))    

    # output shape is according to the number of action
    # The softmax function outputs a probability distribution over the actions
    model.add(Dense(self.action_shape, activation="softmax")) 
    model.compile(loss="categorical_crossentropy",
            optimizer=Adam(lr=self.learning_rate))
        
    return model

  def hot_encode_action(self, action):
    '''encoding the actions into a binary list'''

    action_encoded=np.zeros(self.action_shape, np.float32)
    action_encoded[action]=1

    return action_encoded
  
  def remember(self, state, action, action_prob, behavior_probs, reward):
    '''stores observations'''
    encoded_action=self.hot_encode_action(action)
    self.gradients.append(encoded_action-action_prob)
    self.states.append(state)
    self.rewards.append(reward)
    self.probs.append(action_prob)
    self.behave_probs.append(behavior_probs)

  
  def get_target_prob(self, state):
    '''samples the next action based on the policy probabilty distribution 
      of the actions'''

    # transform state
    state=state.reshape([1, state.shape[0]])
    # get action probably
    action_probability_distribution=self.model.predict(state).flatten()
    # norm action probability distribution
    action_probability_distribution/=np.sum(action_probability_distribution)  

    return action_probability_distribution


  def get_discounted_rewards(self, rewards): 
    ''' \gamma ^ t * reward'''
    
    discounted_rewards=[]
    cumulative_total_return=0
    # iterate the rewards backwards and and calc the total return 
    for reward in rewards[::-1]:      
      cumulative_total_return=(cumulative_total_return*self.gamma)+reward
      discounted_rewards.insert(0, cumulative_total_return)

    # normalize discounted rewards
    mean_rewards=np.mean(discounted_rewards)
    std_rewards=np.std(discounted_rewards)
    norm_discounted_rewards=(discounted_rewards-
                          mean_rewards)/(std_rewards+1e-7) # avoiding zero div
    
    return norm_discounted_rewards


  def update_policy(self):

    states=np.vstack(self.states)
    gradients=np.vstack(self.gradients)
    rewards=np.vstack(self.rewards)
    discounted_rewards=self.get_discounted_rewards(rewards)
    gradients= gradients*discounted_rewards* (self.target_probs)/(self.behave_probs).      #importance weight
    gradients=self.alpha*np.vstack([gradients])+self.target_probs

    history=self.model.train_on_batch(states, gradients)
    
    self.states, self.probs, self.behave_probs, self.gradients, self.rewards=[], [], [], [], []

    return history


  def train(self, episodes):

    
    total_behave_rewards=np.zeros(episodes)
    for episode in range(data.episod.nunique()):
      episode_reward=0
      # each episode is a new game env
      # get an action and record the game state & reward per episode
      array=data.values
      X = np.asarray(array).astype(np.float32)
      i=0
      while X[i][1] == episode:
        
        state=X[i][6:27]
        target_prob=self.get_target_prob(state)
        action=int(X[i][0])
        next_state=X[i+1][6:27]
        reward=X[i][27]
        behavior_probs=X[i][2:6]

        self.remember(state, action, target_prob, behavior_probs, reward)
        state=next_state
        episode_reward+=reward 
        i=+1

          # update policy 

      history=self.update_policy()
        
      total_behave_rewards[episode]=episode_reward
      
    self.total_behave_rewards=total_behave_rewards



  def save_model(self):
    '''saves the moodel // do after training'''
    self.model.save('PG_model.h5')
  
  def load_model(self, path):
    '''loads a trained model from path'''
    return load_model(path)

team=PG(array)

team.train(10)

plt.title('PG Behavioral Reward')
plt.xlabel('Episode')
plt.ylabel('Average behavioral reward (Episode length)')
plt.plot(team.total_behavioral_rewards)
plt.show()















