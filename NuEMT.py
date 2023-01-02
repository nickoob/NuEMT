# Code in this file is copied and adapted from https://github.com/modestyachts/ARS
import gym
import random
import argparse
import time
from copy import deepcopy
import ray
from shared_noise import *
from policies import *
import utils
import random
import mylog
from math import ceil, log, pow, pi
from numpy.linalg import norm
from decimal import Decimal
import decimal

np.set_printoptions(precision=5)

@ray.remote
class Worker(object):
    def __init__(self, env_seed,
                 env_name='',
                 env_config=None,
                 policy_params=None,
                 deltas=None,
                 rollout_type='rollout_mujoco',
                 delta_std=0.02):

        self.env = gym.make(env_name)
        self.env.seed(env_seed)
        self.rollout_type = rollout_type
        self.rollout = getattr(self, self.rollout_type)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        # print(self.deltas.get(1050, 7))
        self.policy_params = policy_params
        if policy_params['type'] == 'nonlinear':
            self.policy = NonLinearPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = delta_std

    def do_rollouts(self, delta_std, w_policies, mixture_weights, rollout_length, num_rollouts=1, shift=0, evaluate=False):
        rollout_rewards, deltas_idx = [], []
        deltas_sign = []
        total_steps = 0
        rollout_l2_decays = []
        policies_idx = []
        self.delta_std = delta_std

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policies[0])
                deltas_idx.append(-1)

                reward, r_steps = self.rollout(shift=0., rollout_length=self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
            else:
                policy_idx = random.choices(population=range(len(w_policies)), weights=mixture_weights, k=1)[0]
                policies_idx.extend([policy_idx, policy_idx])
                w_policy = w_policies[policy_idx]
                idx, delta = self.deltas.get_delta(w_policy.size)
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.extend([idx, idx])
                # positive
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps = self.rollout(shift=shift, rollout_length=rollout_length)
                rollout_rewards.append(pos_reward)
                deltas_sign.append(1)

                weight_decay = 0.01
                l2_decay = compute_weight_decay(weight_decay, self.policy.weights.flatten())
                rollout_l2_decays.append(l2_decay)
                # negative
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift=shift, rollout_length=rollout_length)
                rollout_rewards.append(neg_reward)
                deltas_sign.append(-1)

                l2_decay = compute_weight_decay(weight_decay, self.policy.weights.flatten())
                rollout_l2_decays.append(l2_decay)

                total_steps += pos_steps + neg_steps

        return {"rollout_rewards": rollout_rewards, "deltas_idx": deltas_idx, "policies_idx": policies_idx,
                "deltas_sign": deltas_sign, "total_steps": total_steps, "l2_decay": rollout_l2_decays}

    def rollout_mujoco(self, shift=0., rollout_length=None):
        """
           Performs one rollout of maximum length rollout_length.
           At each time-step it substracts shift from the reward.
       """
        total_reward = 0.
        steps = 0
        ob = self.env.reset()
        tmp_ob = deepcopy(ob)
        for i in range(rollout_length):
            # try:
            action = self.policy.act(ob)
            # time.sleep(.01)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            # self.env.render()
            if done:
                break

        return total_reward, steps

    def get_weights(self):
        return self.policy.get_weights()

    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def calculate_mixture_and_policy_step(self, solutions, projected_solutions, rewards, mod_rewards, w_policies, mixture_weights, task_id):
        mixture_update = np.zeros(len(w_policies))
        policy_update = np.zeros(solutions[0].shape)

        for reward, mod_reward, solution, projected_solution in list(zip(rewards, mod_rewards, solutions, projected_solutions)):
            likelihood_list = []
            projected_likelihood_list = []

            # print(w_policies)
            for w_policy in w_policies:
                likelihood_list.append(utils.factored_pdf(solution,
                                                          w_policy.flatten(),
                                                          np.ones(w_policy.size) * pow(self.delta_std, 2)))

                projected_likelihood_list.append(utils.factored_pdf(projected_solution,
                                                          w_policy.flatten(),
                                                          np.ones(w_policy.size) * pow(self.delta_std, 2)))

            epsilon = 1e-3
            mixture_weights_list = [Decimal(w+epsilon) for w in mixture_weights]
            likelihood_mixture_model = [a*b for a, b in zip(mixture_weights_list, likelihood_list)]
            projected_likelihood_mixture_model = [Decimal(a*b) for a, b in zip(mixture_weights_list, projected_likelihood_list)]
            for i in range(len(w_policies)):
                # this should be in Decimal type
                # print("np sum: {}".format(np.sum(likelihood_list)))
                likelihood_norm = likelihood_list[i] / sum(likelihood_mixture_model)
                # back to float
                mixture_update[i] += float(likelihood_norm) * reward
                # mixture_update[i] += float(likelihood_norm) * mod_reward

            try:
                p_likelihood_norm = float(projected_likelihood_list[task_id] / sum(projected_likelihood_mixture_model))
            except decimal.DivisionByZero:
                print("projected mixture model: {}, projected likelihood list: {}".format(projected_likelihood_mixture_model, projected_likelihood_list))
                exit()

            # if p_likelihood_norm == float("inf"):
            #     p_likelihood_norm = 1000.0
            policy_update += p_likelihood_norm * reward * mixture_weights[task_id] * (
                    projected_solution - w_policies[task_id].flatten())

        return mixture_update, policy_update


class ES(object):
    def __init__(self, env_name=None,
                 env_config=None,
                 gen=100,
                 policy_params=None,
                 num_workers=24,
                 pop_size=100,
                 delta_std=0.02,
                 delta_decay=None,
                 rollout_length=1000,
                 rollout_type='rollout_mujoco',
                 step_size=0.05,
                 shift='constant zero',
                 num_tasks=5,
                 params=None,
                 seed=123,
                 mixture_step_size=0.05,
                 mixture_step_decay=None,
                 timesteps_limit=900000000,
                 logger=None):

        self.timesteps_limit = timesteps_limit
        self.current_gen = 0
        self.env_config = env_config
        self.env_name = env_name
        self.total_steps = 0
        self.time_mixture = 0.0
        self.logger = logger
        self.gen = gen
        self.policy_params = policy_params
        self.num_workers = num_workers
        self.pop_size = int(pop_size / 2)
        self.delta_std = delta_std
        self.delta_decay = delta_decay
        self.rollout_length = rollout_length
        self.rollout_type = rollout_type
        self.step_size = step_size
        self.shift = shift
        self.params = params
        self.mixture_step_size = mixture_step_size
        self.mixture_step_decay = mixture_step_decay
        self.current_avg_reward_per_task = np.zeros(num_tasks)

        self.avg_performance = []
        self.l2norm = []

        self.current_best_sol = None
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=seed + 3)
        print('Created deltas table.')

        print('Initializing workers.')
        self.num_workers = num_workers
        self.num_tasks = num_tasks

        if policy_params['type'] == 'controller':
            self.rollout_length_per_task = [(i + 1) / self.num_tasks * rollout_length for i in
                                            range(self.num_tasks)]
        else:
            self.rollout_length_per_task = [int((i+1)/self.num_tasks * rollout_length) for i in range(self.num_tasks)]

        self.mixture_weights_per_task = [np.ones(i+1) / (i+1) for i in range(self.num_tasks)]

        for idx, mw in enumerate(self.mixture_weights_per_task):
            print("Mixture Weights for Task {}: {}".format(idx+1, mw))
        self.resource_allocation = self.mixture_weights_per_task[-1]
        print("Allocation Weight: {}".format(self.resource_allocation))
        self.pop_size_per_task = [int(ceil(self.pop_size * ra)) for ra in self.resource_allocation]
        print("Resource Allocation: {}".format([2*pop for pop in self.pop_size_per_task]))

        self.workers = [Worker.remote(seed+7*i,
                                      env_name=env_name,
                                      env_config=env_config,
                                      policy_params=policy_params,
                                      rollout_type=self.rollout_type,
                                      deltas=deltas_id,
                                      delta_std=delta_std) for i in range(num_workers)]

        # initialize policy
        if policy_params['type'] == 'nonlinear':
            self.policy_per_task = [NonLinearPolicy(policy_params) for _ in range(self.num_tasks)]
        else:
            raise NotImplementedError

        self.w_policy_per_task = [policy.get_weights() for policy in self.policy_per_task]

        print("Initialization of ES is completed")

    def train(self, trial_no):
        max_reward_list = [[0.0 for _ in range(self.num_tasks)]]
        num_episodes_list = [[0 for _ in range(self.num_tasks)]]
        time_taken_list = [0.0]
        total_steps_list = [0]
        weights_list = [[0.0 for _ in range(self.num_tasks)]]
        start_time = time.time()
        for g in range(1, self.gen+1):
            # step_start = time.time()
            self.current_gen = g
            print("======================== trial_no: {}, gen {} ========================= delta std: {}, mixture_step_size: {}"
                  .format(trial_no, g, self.delta_std, self.mixture_step_size))
            max_reward_per_task, num_episodes_per_task = self.train_step()
            weights_list.append(self.mixture_weights_per_task[-1].tolist())
            max_reward_list.append(max_reward_per_task)
            num_episodes_list.append(num_episodes_per_task)
            total_steps_list.append(self.total_steps)
            time_spent = time.time() - start_time
            time_taken_list.append(time_spent)
            print('total steps: {}'.format(self.total_steps))
            print("time spent: {}".format(time_spent))
            print('time spent on mixture: {}'.format(self.time_mixture))

            if self.delta_decay:
                if self.delta_std > 0.01:
                    self.delta_std *= self.delta_decay

            if self.mixture_step_decay:
                if self.mixture_step_decay > 0.01:
                    self.mixture_step_size *= self.mixture_step_decay

            t1 = time.time()

            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy_per_task[-1].observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy_per_task[-1].observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy_per_task[-1].observation_filter.clear_buffer()

            # sync all workers
            filter_id = ray.put(self.policy_per_task[-1].observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)

            if self.total_steps > self.timesteps_limit:
                break

        self.logger.store_trial_stats(max_reward_list, time_taken_list, num_episodes_list,
                                      total_steps_list, weight=weights_list,
                                      avg_perf=self.avg_performance, l2norm=self.l2norm)

    def train_step(self):
        step_per_task, mixture_weights_update_per_task, max_reward_per_task, num_episodes_per_task = self.parallel_rollouts()

        # stochastic gradient ascent
        self.w_policy_per_task = [w_policy + (self.step_size * step).reshape(w_policy.shape)
                                  for w_policy, step in list(zip(self.w_policy_per_task, step_per_task))]

        print("auxiliary task policy 1: {}".format(self.w_policy_per_task[0]))
        print("auxiliary task policy 2: {}".format(self.w_policy_per_task[1]))
        print("main task policy: {}".format(self.w_policy_per_task[-1]))

        self.l2norm.append([norm(policy) for policy in self.w_policy_per_task])

        start1 = time.time()
        self.mixture_weights_per_task = [orthogonal_projection(m_update * self.mixture_step_size, m_weight)
                                         for m_weight, m_update in list(zip(self.mixture_weights_per_task, mixture_weights_update_per_task))]

        self.time_mixture += time.time() - start1

        for idx, mw in enumerate(self.mixture_weights_per_task):
            print("Mixture Weights for Task {}: {}, sum: {}".format(idx+1, mw, np.sum(mw)))

        self.resource_allocation = self.mixture_weights_per_task[-1]
        print("Allocation Weight: {}".format(self.resource_allocation))
        self.pop_size_per_task = [max(1, int(ceil(self.pop_size * self.resource_allocation[idx])))
                                  if idx != self.num_tasks-1 else max(int(ceil(self.pop_size/self.num_tasks)), int(ceil(self.pop_size * self.resource_allocation[idx])))
                                  for idx in range(len(self.resource_allocation))]
        print("Resource Allocation: {}".format([2*pop for pop in self.pop_size_per_task]))

        return max_reward_per_task, num_episodes_per_task

    def parallel_rollouts(self, num_rollouts=None, evaluate=False):
        results_per_task = []
        num_episodes_per_task = []
        max_reward_per_task = []
        for task_id, pop in enumerate(self.pop_size_per_task):
            policies_id = ray.put(self.w_policy_per_task[:task_id+1])
            mixture_weights_id = ray.put(self.mixture_weights_per_task[task_id])
            num_rollouts_ids_list = []
            results = []

            tasktime = time.time()
            if pop >= self.num_workers:
                num_rollouts = int(pop / self.num_workers)
                num_rollouts_ids_list.append([worker.do_rollouts.remote(self.delta_std,
                                                                        policies_id,
                                                                        mixture_weights_id,
                                                                        self.rollout_length_per_task[task_id],
                                                                        num_rollouts=num_rollouts,
                                                                        shift=self.shift,
                                                                        evaluate=evaluate) for worker in self.workers])
                num_rollouts_ids_list.append([worker.do_rollouts.remote(self.delta_std,
                                                                        policies_id,
                                                                        mixture_weights_id,
                                                                        self.rollout_length_per_task[task_id],
                                                                        num_rollouts=1,
                                                                        shift=self.shift,
                                                                        evaluate=evaluate) for worker in
                                              self.workers[:(pop % self.num_workers)]])
            else:
                num_rollouts_ids_list.append([worker.do_rollouts.remote(self.delta_std,
                                                                        policies_id,
                                                                        mixture_weights_id,
                                                                        self.rollout_length_per_task[task_id],
                                                                        num_rollouts=1,
                                                                        shift=self.shift,
                                                                        evaluate=evaluate) for worker in
                                              self.workers[:(pop % self.num_workers)]])

            for rollout_ids in num_rollouts_ids_list:
                results.extend(ray.get(rollout_ids))

            print("task {}'s time on eval: {}".format(task_id+1, time.time()-tasktime))
            results_per_task.append(results)
            num_episodes_per_task.append(len(results))

        mixture_weights_step_per_task = []
        policy_step_per_task = []
        for task_id, results in enumerate(results_per_task):
            rollout_rewards, deltas_idx = [], []
            policies_idx = []
            deltas_sign = []
            l2_decay = []
            for result in results:
                deltas_idx += result['deltas_idx']
                policies_idx += result['policies_idx']
                rollout_rewards += result['rollout_rewards']
                deltas_sign += result['deltas_sign']
                l2_decay += result['l2_decay']
                self.total_steps += result['total_steps']

            rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
            max_reward = np.amax(rollout_rewards)
            max_reward_idx = np.argmax(rollout_rewards)

            if task_id == self.num_tasks-1:
                avg_perf = np.zeros(self.num_tasks)
                avg_cnt = np.zeros(self.num_tasks)
                for r, p in list(zip(rollout_rewards, policies_idx)):
                    avg_perf[p] += r
                    avg_cnt[p] += 1

                avg_cnt[avg_cnt == 0] = 1
                self.avg_performance.append(list(avg_perf / avg_cnt))

                print("avg_performance: {}".format(self.avg_performance[-1]))

            print('Task {}: Total episode ran: {}, Maximum reward of collected rollouts: {}, Average Reward: {}'
                  .format(task_id+1, rollout_rewards.size, max_reward, np.mean(rollout_rewards)))
            max_reward_per_task.append(max_reward)

            rollout_rewards += np.array(l2_decay, dtype=np.float64)
            rollout_rewards = fitness_shaping(rollout_rewards)
            mod_rollout_rewards = deepcopy(rollout_rewards)

            solutions = []
            projected_solutions = []
            print("mixture weight length: {}".format(len(self.mixture_weights_per_task[task_id])))
            for d_idx, p_idx, sign in list(zip(deltas_idx, policies_idx, deltas_sign)):
                solution = self.deltas.get(d_idx, self.w_policy_per_task[p_idx].size) * sign * self.delta_std + self.w_policy_per_task[p_idx].flatten()
                if p_idx != task_id:
                    projected_solution = projected_solutions_on_target(self.w_policy_per_task[task_id].flatten(), solution, self.delta_std)
                else:
                    projected_solution = solution

                solutions.append(solution)
                projected_solutions.append(projected_solution)

            time1 = time.time()

            policies_id = ray.put(self.w_policy_per_task[:task_id+1])
            mixture_weights_id = ray.put(self.mixture_weights_per_task[task_id])
            p_sum_ids_list = []
            results = []

            if len(solutions) >= self.num_workers:
                p_sum = int(len(solutions) / self.num_workers)
                p_sum_ids_list.append([worker.calculate_mixture_and_policy_step.remote(solutions[idx*p_sum:(idx+1)*p_sum],
                                                                                       projected_solutions[idx*p_sum:(idx+1)*p_sum],
                                                                                       rollout_rewards[idx*p_sum:(idx+1)*p_sum],
                                                                                       mod_rollout_rewards[idx*p_sum:(idx+1)*p_sum],
                                                                                       policies_id,
                                                                                       mixture_weights_id,
                                                                                       task_id=task_id)
                                       for idx, worker in enumerate(self.workers)])
                p_sum_ids_list.append([worker.calculate_mixture_and_policy_step.remote([solutions[p_sum*self.num_workers+idx]],
                                                                                       [projected_solutions[p_sum * self.num_workers + idx]],
                                                                                       [rollout_rewards[p_sum*self.num_workers+idx]],
                                                                                       [mod_rollout_rewards[p_sum*self.num_workers+idx]],
                                                                                       policies_id,
                                                                                       mixture_weights_id,
                                                                                       task_id=task_id)
                                       for idx, worker in enumerate(self.workers[:(len(solutions) % self.num_workers)])])
            else:
                p_sum_ids_list.append([worker.calculate_mixture_and_policy_step.remote([solutions[idx]],
                                                                                       [projected_solutions[idx]],
                                                                                       [rollout_rewards[idx]],
                                                                                       [mod_rollout_rewards[idx]],
                                                                                       policies_id,
                                                                                       mixture_weights_id,
                                                                                       task_id=task_id)
                                       for idx, worker in enumerate(self.workers[:(len(solutions) % self.num_workers)])])

            for p_sum_ids in p_sum_ids_list:
                results.extend(ray.get(p_sum_ids))

            mixture_weights_step = np.zeros(self.mixture_weights_per_task[task_id].shape)
            policy_step = np.zeros(self.w_policy_per_task[task_id].size)
            for m_step, p_step in results:
                mixture_weights_step += m_step
                policy_step += p_step

            mixture_weights_step_per_task.append(mixture_weights_step / len(rollout_rewards))
            policy_step_per_task.append(policy_step / (len(rollout_rewards) * pow(self.delta_std, 2)))

            self.time_mixture += time.time() - time1

        return policy_step_per_task, mixture_weights_step_per_task, max_reward_per_task, num_episodes_per_task


def projected_solutions_on_target(mean, vec, sigma):
    mahalanobis_dist = np.sqrt(np.sum(((vec - mean) / sigma) ** 2))
    r = 1
    if mahalanobis_dist > r:
        return mean + (vec - mean) * min(1, r / mahalanobis_dist)

    return vec


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    # print(x.argsort())
    ranks[x.argsort()] = np.arange(len(x))
    # print(ranks)
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def fitness_shaping(x):
    y = -x.ravel()
    ranks = np.empty(len(y), dtype=int)
    ranks[y.argsort()] = np.arange(len(y)) + 1
    l = len(y)
    a = np.log(np.ones(l) * (l / 2) + np.ones(l)) - np.log(ranks)
    a[a < 0] = 0
    b = np.sum(a)
    a = a / b
    return a


def normalise_mixture_weights(list_of_weights):
    for i in range(len(list_of_weights)):
        list_of_weights[i] = list_of_weights[i] / np.sum(list_of_weights[i])

    return list_of_weights


def orthogonal_projection(x, m_weight):
    projected_vector = (x - np.ones(len(x)) * np.sum(x)/(1.0*len(x)))
    projected_weights = m_weight + projected_vector
    if (projected_weights >= 0.0).all():
        return projected_weights
    else:
        min_idx = np.argmin(projected_weights)
        scaling = m_weight[min_idx] / (m_weight[min_idx] - projected_weights[min_idx])
        return m_weight + scaling * projected_vector


def compute_weight_decay(weight_decay, model_param):
    # model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param * model_param, axis=0)


def run_evolutionary_search(params):
    variant = ""
    params['rollout_type'] = 'rollout_mujoco'
    params['env_config'] = None
    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print("ob dim: {}, ac_dim: {}".format(ob_dim, ac_dim))
    # ac_high = env.action_space.high
    # ac_low = env.action_space.low
    policy_params = {'type': params['policy_type'],
                     'base_ea': params['algorithm'],
                     'ob_dim': ob_dim,
                     'ac_dim': ac_dim,
                     # 'ac_high': ac_high,
                     # 'ac_low': ac_low,
                     'h1_dim': 64,
                     'h2_dim': 64,
                     'ob_filter': 'MeanStdFilter'}

    params['h1_dim'] = policy_params['h1_dim']
    params['h2_dim'] = policy_params['h2_dim']

    seeds = [467, 667, 11, 120, 534, 888, 524, 672, 913, 682,
             641, 91, 460, 596, 580, 107, 249, 398, 44, 496,
             812, 742, 884, 405, 134, 850, 8, 703, 743, 318]
    if params['algorithm'] == 'es':
        logger = mylog.Log(params, variant)
        for i in range(1, params['num_trials']+1):
            ray.init(include_dashboard=False, num_cpus=params['num_workers'])
            print("====================================== Trial No.{} ==================================="
                  .format(i))
            es = ES(gen=params['generation'],
                    num_workers=params['num_workers'],
                    pop_size=params['population'],
                    policy_params=policy_params,
                    mixture_step_size=params['mixture_step_size'],
                    step_size=params['step_size'],
                    delta_std=params['delta_std'],
                    delta_decay=params['delta_decay'],
                    env_name=params['env_name'],
                    env_config=params['env_config'],
                    num_tasks=params['num_tasks'],
                    rollout_length=params['rollout_length'],
                    rollout_type=params['rollout_type'],
                    shift=params['shift'],
                    seed=seeds[i-1],
                    logger=logger,
                    timesteps_limit=params['timesteps_limit'])
            es.train(i)
            ray.shutdown()
            logger.json_incremental_output(i == 1)
            logger.clear_store_trial_stats()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env_type', type=str, default='mujoco')
    parser.add_argument('--timesteps_limit', '-tl', type=int, default=80000000)
    parser.add_argument('--generation', '-g', type=int, default=1000000)
    parser.add_argument('--population', '-p', type=int, default=128)
    parser.add_argument('--step_size', type=float, default=0.05)
    parser.add_argument('--mixture_step_size', type=float, default=0.05)
    parser.add_argument('--delta_std', type=float, default=0.02)
    parser.add_argument('--delta_decay', type=float, default=None)
    parser.add_argument('--num_tasks', '-t', type=int, default=2)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--algorithm', type=str, default="es")
    parser.add_argument('--num_trials', '-nt', type=int, default=20)
    parser.add_argument('--file_name', type=str, default="result_nuemt.json")
    parser.add_argument('--num_workers', type=int, default=24)

    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='nonlinear')
    args = parser.parse_args()
    params = vars(args)
    run_evolutionary_search(params)
