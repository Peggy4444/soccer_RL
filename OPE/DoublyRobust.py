# -*- coding: utf-8 -*-

# author: @peggy4444



data=pd.read_csv('data_to_ope.csv')

class DoublyRobust:


    def ope_dr(array) :
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        Paper: https://arxiv.org/pdf/1511.03722.pdf
        :return: the evaluation score
        """

        # Sequential Doubly Robust


        
        per_episode_seq_dr = []
        for episode in range(data.episod.nunique()):
          array=data.values
          X = np.asarray(array).astype(np.float32)
          i=0
          episode_seq_dr = 0
          while X[i][1] == episode:

            optimal_probs= X[i][6:10]
            discounted_reward=X[i][31]
            behavior_probs=X[i][2:6]
            v_value_q_model_based=X[i][34]
            q_value=X[i][33]
            reward=X[i][30]

            rho = optimal_probs / \
                      behavior_probs

            episode_seq_dr = v_value_q_model_based + rho * (reward + discount_factor
                                                                                   * episode_seq_dr - q_value)
                                                                                  
              
    
          per_episode_seq_dr.append(episode_seq_dr)

      ope_dr = np.array(per_episode_seq_dr).mean()

      return ope_dr
