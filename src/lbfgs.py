import numpy

from chainer import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.stack_size = 100
_default_hyperparam.min_ro = 1e-5


class LBFGSRule(optimizer.UpdateRule):

    """Update rule of L-BFGS.
    """

    def __init__(self, parent_hyperparam=None,
                 lr=None, stack_size=None, min_ro=None):
        super(LBFGSRule, self).__init__(parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if stack_size is not None:
            self.hyperparam.stack_size = stack_size
        if min_ro is not None:
            self.hyperparam.min_ro = min_ro

    def init_state(self, param):
        with cuda.get_device(param.data):
            self.state['s'] = []

    def update_core(self, param):
        xp = cuda.get_array_module(param.data)
        data = xp.ravel(param.data)
        grad = xp.ravel(param.grad)
        if not 'x' in self.state:
            s = xp.zeros_like(data)
            y = xp.zeros_like(grad)
            self.state['x'] = data.copy()
            self.state['g'] = grad.copy()
            h0 = 1
        else:
            s = data - self.state['x']
            y = grad - self.state['g']
            self.state['x'][...] = data
            self.state['g'][...] = grad
            ys = xp.dot(y.T, s)
            if ys > 1e-10:
                self.state['s'].append((s, y))
                if len(self.state['s']) > self.hyperparam.stack_size:
                    self.state['s'] = self.state['s'][1:]
                h0 = xp.dot(y.T, s) / xp.dot(y.T, y)
            else:
                h0 = self.state['h0']
        self.state['h0'] = h0

        q = grad.copy()
        stack = []
        for s, y in self.state['s'][::-1]:
            ro_inv = xp.dot(y.T, s)
            if xp.abs(ro_inv) < self.hyperparam.min_ro:
                if ro_inv >= 0:
                    ro_inv = self.hyperparam.min_ro
                else:
                    ro_inv = -self.hyperparam.min_ro
            ro = 1 / ro_inv
            a = ro * xp.dot(s.T, q)
            q -= a * y
            stack.append((ro, a))
        q *= h0
        for s, y in self.state['s']:
            ro, a = stack.pop()
            b = ro * xp.dot(y.T, q)
            q += (a - b) * s
        param.data -= self.hyperparam.lr * xp.reshape(q, param.data.shape)

class LBFGS(optimizer.GradientMethod):
    """L-BFGS.
    """

    def __init__(self,
                 lr=_default_hyperparam.lr,
                 stack_size=_default_hyperparam.stack_size,
                 min_ro=_default_hyperparam.min_ro):
        super(LBFGS, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.stack_size = stack_size
        self.hyperparam.min_ro = min_ro

    lr = optimizer.HyperparameterProxy('lr')
    stack_size = optimizer.HyperparameterProxy('stack_size')
    min_ro = optimizer.HyperparameterProxy('min_ro')

    def create_update_rule(self):
        return LBFGSRule(self.hyperparam)
