import json
import os


class Log(object):
    def __init__(self, hyper_parameters, variant=""):
        self.maximum_reward = []
        self.time_taken = []
        self.episodes = []
        self.weights = []
        self.hyper_parameters = hyper_parameters
        self.trained_policy = []
        self.total_steps = []
        self.data_identity = hyper_parameters['env_name'] + variant + '_' + hyper_parameters['algorithm'] + '_' + \
                             str(hyper_parameters['num_tasks']) + '_' + str(hyper_parameters['population'])

        self.avg_perf = []
        self.l2norm = []

        print("data identity is : {}".format(self.data_identity))

    def store_trial_stats(self, maximum_reward, time_taken, episodes, total_steps, **kwargs):
        self.maximum_reward.append(maximum_reward)
        self.time_taken.append(time_taken)
        self.episodes.append(episodes)
        self.total_steps.append(total_steps)
        if 'weight' in kwargs:
            self.weights.append(kwargs['weight'])
        if 'trained_policy' in kwargs:
            self.trained_policy.append(list(kwargs['trained_policy']))
        if 'avg_perf' in kwargs and 'l2norm' in kwargs:
            self.avg_perf.append(kwargs['avg_perf'])
            self.l2norm.append(kwargs['l2norm'])

    def clear_store_trial_stats(self):
        self.maximum_reward = []
        self.time_taken = []
        self.episodes = []
        self.weights = []
        self.trained_policy = []
        self.total_steps = []
        self.l2norm = []
        self.avg_perf = []

    def json_output(self):
        new_data = {self.data_identity: {'maximum_reward': self.maximum_reward,
                                         'time_taken': self.time_taken,
                                         'episodes': self.episodes,
                                         'total_steps': self.total_steps,
                                         'weights': self.weights,
                                         'hyper_parameters': self.hyper_parameters,
                                         'trained_policy': self.trained_policy,
                                         'avg_perf': self.avg_perf,
                                         'l2norm': self.l2norm}}

        if os.path.exists(self.hyper_parameters['file_name']):
            print('file exist')
            with open(self.hyper_parameters['file_name'], 'r') as f:
                data = json.load(f)
        else:
            print('file does not exist')
            data = {}

        data.update(new_data)

        with open(self.hyper_parameters['file_name'], 'w') as f:
            json.dump(data, f)

    def json_incremental_output(self, first_trial):
        new_data = {self.data_identity: {'maximum_reward': self.maximum_reward,
                                         'time_taken': self.time_taken,
                                         'episodes': self.episodes,
                                         'total_steps': self.total_steps,
                                         'weights': self.weights,
                                         'hyper_parameters': self.hyper_parameters,
                                         'trained_policy': self.trained_policy,
                                         'avg_perf': self.avg_perf,
                                         'l2norm': self.l2norm}}
        if not os.path.exists(self.hyper_parameters['file_name']):
            print('file does not exist')
            data = {}
            data.update(new_data)
        elif first_trial:
            print('first trial (clear previous data)')
            with open(self.hyper_parameters['file_name'], 'r') as f:
                data = json.load(f)
                # data[self.data_identity] = {}
                data.update(new_data)
        else:
            print('file exist')
            with open(self.hyper_parameters['file_name'], 'r') as f:
                data = json.load(f)
                data[self.data_identity]['maximum_reward'].extend(new_data[self.data_identity]['maximum_reward'])
                data[self.data_identity]['time_taken'].extend(new_data[self.data_identity]['time_taken'])
                data[self.data_identity]['episodes'].extend(new_data[self.data_identity]['episodes'])
                data[self.data_identity]['total_steps'].extend(new_data[self.data_identity]['total_steps'])
                data[self.data_identity]['weights'].extend(new_data[self.data_identity]['weights'])

                data[self.data_identity]['avg_perf'].extend(new_data[self.data_identity]['avg_perf'])
                data[self.data_identity]['l2norm'].extend(new_data[self.data_identity]['l2norm'])

        with open(self.hyper_parameters['file_name'], 'w') as f:
            json.dump(data, f)

        print("done saving!")


def log_tabular():
    raise NotImplementedError

