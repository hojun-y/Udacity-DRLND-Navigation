config = dict()
config['replay_size'] = 20000
config['lr'] = 0.001
config['discount_factor'] = 0.99
config['batch_size'] = 32
config['gradient_clip'] = 0.8

config['history_len'] = 1
config['train_start'] = 2000
config['target_score'] = 13
config['sync_target_every'] = 250
config['epsilon_decay'] = 0.999
config['epsilon_lower_bound'] = 0.1

config['fc1'] = 25
config['fc2'] = 25
config['loss_type'] = 'huber'

config['weights_save_path'] = 'save/weights.data'
config['plot_save_path'] = 'save/reward.png'
config['rewards_save_path'] = 'save/reward.dmp'

config['print_every'] = 1
