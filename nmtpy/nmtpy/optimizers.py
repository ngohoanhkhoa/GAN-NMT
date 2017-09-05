# -*- coding: utf-8 -*-
import numpy as np

import theano
import theano.tensor as tensor

from .defaults import FLOAT

def sgd(tparams, grads, inp, cost, lr0):
    """Stochastic Gradient Descent optimizer."""
    # define the update step rule
    updates = []
    for p, g in zip(tparams.values(), grads):
        updates.append((p, p - lr0 * g))

    return updates

def rmsprop(tparams, grads, inp, cost, lr0=0.01, rho=0.95, eps=1e-6):
    """RMSProp optimizer."""
    # define the update step rule
    updates = []
    one = tensor.constant(1.)
    for p, g in zip(tparams.values(), grads):
        # Accumulate gradient squares
        v = theano.shared(np.zeros(p.get_value().shape).astype(FLOAT))
        # rho * past + (1 - rho) * current
        v_new = (rho * v) + (one - rho) * g**2
        updates.append((v, v_new))
        updates.append((p, p - (lr0 * g / tensor.sqrt(v_new + eps))))

    return updates

def adadelta(tparams, grads, inp, cost, lr0=1., rho=0.95, eps=1e-6):
    """Adadelta optimizer."""
    # define the update step rule
    updates = []
    one = tensor.constant(1.)
    for p, g in zip(tparams.values(), grads):
        v = theano.shared(np.zeros(p.get_value().shape).astype(FLOAT))
        u = theano.shared(np.zeros(p.get_value().shape).astype(FLOAT))

        # Accumulate gradient squares
        # rho * past + (1 - rho) * current
        v_new = (rho * v) + (one - rho) * g**2
        updates.append((v, v_new))

        # Update rule
        up = (g * tensor.sqrt(u + eps) / tensor.sqrt(v_new + eps))
        updates.append((p, p - lr0 * up))

        # Accumulate update magnitudes
        updates.append((u, rho * u + (one - rho) * up**2))

    return updates

def adam(tparams, grads, inp, cost, lr0=0.0001, b1=0.9, b2=0.999, eps=1e-8):
    """ADAM optimizer."""
    i = theano.shared(np.float64(0.).astype(FLOAT))
    i_t = i + 1.

    # Running learning-rate
    lr_t = lr0 * (tensor.sqrt(1. - b2**i_t) / (1. - b1**i_t))

    updates = []

    for p, g in zip(tparams.values(), grads):
        m = theano.shared(np.zeros(p.get_value().shape).astype(FLOAT), p.name + '_mu')
        v = theano.shared(np.zeros(p.get_value().shape).astype(FLOAT), p.name + '_var')

        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        p_t = p - (lr_t * (m_t / (tensor.sqrt(v_t) + eps)))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates
