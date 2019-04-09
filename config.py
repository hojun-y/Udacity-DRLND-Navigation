config = dict()
config['replay_size'] = 80000
config['lr'] = 3e-4
config['discount_factor'] = 0.99
config['batch_size'] = 64
config['gradient_clip'] = 0.8

config['history_len'] = 1
config['train_start'] = 30000
config['target_score'] = 13
config['sync_target_every'] = 300
config['epsilon_decay'] = 0.995
config['epsilon_lower_bound'] = 0.1

config['fc1'] = 64
config['fc2'] = 64
config['loss_type'] = 'huber'

config['weights_save_path'] = 'save/weights.data'
config['plot_save_path'] = 'save/reward.png'
config['rewards_save_path'] = 'save/reward.dmp'

config['print_every'] = 1
