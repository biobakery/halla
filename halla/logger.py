import pkg_resources  # part of setuptools to retrieve version
import datetime

class HAllALogger(object):
    def __init__(self, name, config):
        self.name = name
        self.verbose = config.output['verbose']
        self.performance_txt = 'HAllA version:\t' + pkg_resources.require('HAllA')[0].version + '\n'
        self.durations = [] # details on durations for certain steps
        self.results = []   # details on results from certain steps
        self.log_config(config)
    
    def log_config(self, config, return_text=False):
        '''Log configuration parameters by iterating through config (Struct) variable
        if return_text is True, return text instead of printing anything
        '''
        def format_text(text):
            return(text.replace('_', ' '))
        log_txt = ''
        config_dict = config.__dict__
        for key_1 in config_dict:
            log_txt += '  %s:\n' % format_text(key_1)
            for key_2 in config_dict[key_1]:
                log_txt += '    %-*s: ' % (30, format_text(key_2)) + str(config_dict[key_1][key_2]) + '\n'
        if return_text:
            return('\n--Configuration parameters--\n' + log_txt)
        if not self.verbose: return
        print('Setting config parameters (irrelevant parameters will be ignored)...')
        print(log_txt)

    def log_step_start(self, message, sub=False):
        decorator = '--' if sub else '=='
        if self.verbose: print('%s %s %s' % (decorator, message, decorator))
    
    def log_step_end(self, label, dur_second, sub=False):
        decorator = '--' if sub else '=='
        dur_str = str(datetime.timedelta(seconds=dur_second))
        self.durations.append((label, dur_str))
        if self.verbose:
            print('%s Completed; total duration: %s %s\n' % (decorator, dur_str, decorator))

    def log_message(self, message):
        if self.verbose: print(message)
    
    def log_result(self, label, content):
        self.results.append((label, content))
        if self.verbose: print('  ', label, content)