import numpy as np
from math import exp, pi, sqrt, pow
from decimal import Decimal
from copy import deepcopy


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]

    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0

    counter = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):

        assert len(batch_weights) == len(batch_vecs) <= batch_size
        counter+=1
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)

    return total, num_items_summed


def calculate_resource_allocation(mixture_weights_task):
    mixture_weights_per_task = deepcopy(mixture_weights_task)
    for i in range(1, len(mixture_weights_per_task)):
        tmp = mixture_weights_per_task[i][i-1] * mixture_weights_per_task[i-1]
        mixture_weights_per_task[i][i-1] = 0.0
        mixture_weights_per_task[i][:i] += tmp

    tmp = mixture_weights_per_task[-1][-1] * mixture_weights_per_task[-1]
    mixture_weights_per_task[-1][-1] = 0.0
    mixture_weights_per_task[-1] += tmp
    return mixture_weights_per_task[-1]


def factored_pdf(x, mean, cov):
    pdf = Decimal(1.0)
    for i in range(x.size):
        pdf *= Decimal(exp(-0.5 * (pow((x[i] - mean[i]), 2) / cov[i])) / sqrt(cov[i]*2*pi))

    return pdf


def mixture_and_policy_update(rewards, solutions, policies, mixture_weights, task_id, delta_std):
    mixture_update = [0.0 for _ in range(len(policies))]
    policy_update = np.zeros(solutions[0].shape)
    for reward, solution in list(zip(rewards, solutions)):
        likelihood_list = []
        for w_policy in policies:
            likelihood_list.append(factored_pdf(solution,
                                                w_policy.flatten(),
                                                np.ones(w_policy.size) * pow(delta_std, 2)))
        for i in range(len(policies)):
            # this should be in Decimal type
            likelihood_norm = likelihood_list[i] / sum(likelihood_list)
            # back to float
            mixture_update[i] += float(likelihood_norm) * reward

        p_likelihood_norm = likelihood_list[task_id] / sum(likelihood_list)
        policy_update += float(p_likelihood_norm) * reward * mixture_weights[task_id] * (solution - policies[task_id].flatten())

    return np.array(mixture_update) / len(solutions), policy_update / (len(solutions) * pow(delta_std, 2))
