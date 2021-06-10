
# -*- coding: utf-8 -*-

# author: @peggy4444

data=pd.read_csv('data_to_ope.csv')


class ImportanceSampling:


    def ope_is(array):
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy.
        References:
        - Sutton, R. S. & Barto, A. G. Reinforcement Learning: An Introduction. Chapter 5.5.
        - https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
        - http://videolectures.net/deeplearning2017_thomas_safe_rl/
        :return: the evaluation score
        """
    
        per_episode_w_i = []
        for episode in range(data.episod.nunique()):
          array=data.values
          X = np.asarray(array).astype(np.float32)
          i=0
          while X[i][1] == episode:
  
            optimal_probs= X[i][6:10]
            discounted_reward=X[i][31]
            behavior_probs=X[i][2:6]
            w_i *= optimal_probs / \
                      behavior_probs
            i+=1
          per_episode_w_i.append(w_i)


        total_w_i_sum_across_episodes = sum(per_episode_w_i)

        wis = 0
        if total_w_i_sum_across_episodes != 0:
          i=0
          while X[i][1] == episode:
            discounted_reward=X[i][31]
            ope_is_rew += per_episode_w_i[i] * discounted_reward
            i+=1
          ope_reward /= total_w_i_sum_across_episodes

        return ope_is_reward
