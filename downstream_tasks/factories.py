import logging

import torch

from testing.EulerHeunSampler import EulerHeunSampler_doa_cond

logger = logging.getLogger(__name__)


def network_from_checkpoint(network, device, checkpoint_path, eval=True, return_checkpoint_index=False):
    if eval:
        network = network.eval()

    network.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f'Loaded checkpoint from train iteration {state_dict["it"]}')
    network.load_state_dict(state_dict['ema'])  # EMA weights

    if return_checkpoint_index:
        return state_dict['it'], network
    else:
        return network


def make_sampler(network, sampling_parameters, device, checkpoint_path):
    logger.info('making EulerHeunSampler object')

    checkpoint_i, network = network_from_checkpoint(network=network, device=device, checkpoint_path=checkpoint_path,
                                                    eval=True, return_checkpoint_index=True)

    return checkpoint_i, EulerHeunSampler_doa_cond(model=network,
                                                   diff_params=sampling_parameters['diff_params'],
                                                   T=sampling_parameters['T'],
                                                   Schurn=sampling_parameters['Schurn'],
                                                   Snoise=sampling_parameters['Snoise'],
                                                   Stmin=sampling_parameters['Stmin'],
                                                   Stmax=sampling_parameters['Stmax'],
                                                   order=sampling_parameters['order'],
                                                   schedule_mode=sampling_parameters['schedule_mode'])
