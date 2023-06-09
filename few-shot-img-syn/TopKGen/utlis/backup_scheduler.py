


class BackUpScheduler:
    def __init__(self, args, log_path):
        self.args = args 
        self.best_performance = 9999999
        self.log_path = log_path
        self.bad_behavior_num = 0

    def backup(self, new_progress_or_not, best_param_G):
        """if performance doesn't improve aross n evaludations, reset to the best G parameters to optimal and corrupt D parameters"""
        if self.bad_behavior_num >= self.args.backup_scheduler_patience:
            self.reset_G(netG, )
    
    def reset_G(self, netG, avg_param_G):
        
        load_params(netG, avg_param_G)
