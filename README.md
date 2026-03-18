# rl-from-scratch
Implementations of Deep Q-Network (DQN) variants from scratch using PyTorch, trained on the [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment.                                                             
   
## Implementations                                                                                                                                                                                                                                                                                            
  - **DQN** — standard Deep Q-Network with experience replay and a target network                                                                                                                                                                       
  - **Dueling DQN** — splits the Q-network into separate value and advantage streams 
  - **GRU-DQN** — recurrent variant that processes sequences of states through a GRU layer                                                                                                                                                              
                                                                                                                                                                                                                                                        
## Experiments                                                                                                                                                                                                                                                       
  - Baseline DQN training                                                                                                                                                                                                                               
  - Hyperparameter sensitivity analysis (target update frequency, replay memory size, epsilon decay, learning rate)
  - DQN vs Dueling DQN comparison                                                                                                                                                                                                                       
  - DQN vs GRU-DQN comparison
 
### instal reuirements
pip install -r requirements.txt -q 

### DQN baseline                                                              
python experiments/dqn_baseline.py  

### DQN hyperparameter tuning                                                                                                                                                                                                                    
python experiments/dqn_hyperparam_tuning.py

### DQN vs Dueling DQN                                                                                                                                                                                                                           
python experiments/compare_dueling.py  

### DQN vs GRU DQN                                                            
python experiments/compare_gru.py  


## Results                                                                                                                                                                                                                                            
           
### DQN Baseline                                                                                                                                                                                                                                      
![DQN Baseline](figures/dqn_acrobot_baseline.png)                                                                                                                                                                                                     
                                                                                                                                                                                                                                                        
### DQN vs Dueling DQN                                                                                                                                                                                                                              
![DQN vs Dueling DQN](figures/dqn_vs_dueling_dqn_comparison.png)                                                                                                                                                                                      
                                                                                                                                                                                                                                                        
### DQN vs GRU DQN                                                                                                                                                                                                                                    
![DQN vs GRU DQN](figures/dqn_vs_gru_dqn_acrobot.png)                                                                                                                                                                                                 
                                                                                                                                                                                                                                                        
### Hyperparameter Sensitivity                                                                                                                                                                                                                        
![Target Update Frequency](figures/acrobot_target_update_sensitivity.png)                                                                                                                                                                             
![Replay Memory Size](figures/acrobot_memory_size_sensitivity.png)                                                                                                                                                                                    
![Epsilon Decay Steps](figures/acrobot_epsilon_decay_sensitivity.png)                                                                                                                                                                                 
![Learning Rate](figures/acrobot_learning_rate_sensitivity.png)  
