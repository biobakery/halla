import pkg_resources  # part of setuptools to retrieve version
import datetime
from os.path import join

class HAllALogger(object):
    def __init__(self, name, config):
        self.name = name
        self.verbose = config.output['verbose']
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
            return(log_txt)
        if not self.verbose: return
        print('Setting config parameters (irrelevant parameters will be ignored)...')
        print(log_txt)

    def log_step_start(self, message, sub=False):
        '''Log the beginning of a step (or a sub-step):
        - a (main) step: == message ==
        - a (sub)  step: -- message --
        '''
        decorator = '--' if sub else '=='
        if self.verbose: print('%s %s %s' % (decorator, message, decorator))
    
    def log_step_end(self, label, dur_second, sub=False):
        '''Log the end of a step (or a sub-step):
        - a (main) step: == Completed; total duration: time ==
        - a (sub)  step: -- Completed; total duration: time --
        The (label, dur_second) is added to self.durations array to be printed in performance.txt
        '''
        decorator = '--' if sub else '=='
        dur_str = str(datetime.timedelta(seconds=dur_second))
        self.durations.append((label, dur_second))
        if self.verbose:
            print('%s Completed; total duration: %s %s\n' % (decorator, dur_str, decorator))

    def log_message(self, message):
        '''Log a message
        '''
        if self.verbose: print(message)
    
    def log_result(self, label, content):
        '''Log results with prefix '  '
        The (label, content) is added to self.results array to be printed in performance.txt
        '''
        self.results.append((label, content))
        if self.verbose: print('* ', label, content)
    
    def write_performance_log(self, dir_name, config, file_name='performance.txt'):
        '''Write performance.txt which contains:
        - halla version
        - configuration parameters
        - results details
        - durations details
        '''
        performance_txt = 'HAllA version:\t' + pkg_resources.require('HAllA')[0].version + '\n'
        # add config details
        performance_txt += '\n--Configuration parameters--\n' + self.log_config(config, return_text=True)
        # add result details
        performance_txt += '\n--Results--\n' + \
                           '\n'.join(['%-*s: ' % (60, item[0]) + str(item[1]) for item in self.results]) + '\n'
        # add duration details
        performance_txt += '\n--Durations--\n' + \
                           '\n'.join(['%-*s: ' % (60, item[0]) + str(datetime.timedelta(seconds=item[1])) \
                                      for item in self.durations]) + '\n'
        tot_dur = sum(item[1] for item in self.durations)
        performance_txt += '%-*s: ' % (60, 'Total execution time') + str(datetime.timedelta(seconds=tot_dur))
        file = open(join(dir_name, file_name), 'w')
        file.write(performance_txt)
        file.close() 