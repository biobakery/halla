import pkg_resources  # part of setuptools to retrieve version

class HAllALogger(object):
    def __init__(self, name, verbose=True):
        self.verbose = verbose
        self.performance_txt = 'HAllA version:\t' + pkg_resources.require('HAllA')[0].version
        print(self.performance_txt)
    
    # def log_config(self, config):
    #     '''Log the configuration setting
    #     '''
    
    # def write_performance(self):