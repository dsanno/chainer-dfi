import numpy

from chainer import cuda
from chainer import optimizer

class LBFGS(optimizer.GradientMethod):

    "L-BFGS optimization algorithm."
    def __init__(self, lr=0.01, size=100):
        self.lr = lr
        self._size = size
        self._min_ro = 1e-5

    def init_state(self, param, state):
        with cuda.get_device(param.data):
            state['s'] = []

    def update_one(self, param, state):
        xp = cuda.get_array_module(param.data)
        data = xp.ravel(param.data)
        grad = xp.ravel(param.grad)
        if not 'x' in state:
            s = xp.zeros_like(data)
            y = xp.zeros_like(grad)
            state['x'] = data.copy()
            state['g'] = grad.copy()
            h0 = 1
        else:
            s = data - state['x']
            y = grad - state['g']
            state['x'][...] = data
            state['g'][...] = grad
            ys = xp.dot(y.T, s)
            if ys > 1e-10:
                state['s'].append((s, y))
                if len(state['s']) > self._size:
                    state['s'] = state['s'][1:]
                h0 = xp.dot(y.T, s) / xp.dot(y.T, y)
            else:
                h0 = state['h0']
        state['h0'] = h0

        q = grad.copy()
        stack = []
        for s, y in state['s'][::-1]:
            ro_inv = xp.dot(y.T, s)
            if xp.abs(ro_inv) < self._min_ro:
                if ro_inv >= 0:
                    ro_inv = self._min_ro
                else:
                    ro_inv = -self._min_ro
            ro = 1 / ro_inv
            a = ro * xp.dot(s.T, q)
            q -= a * y
            stack.append((ro, a))
        q *= h0
        for s, y in state['s']:
            ro, a = stack.pop()
            b = ro * xp.dot(y.T, q)
            q += (a - b) * s
        param.data -= self.lr * xp.reshape(q, param.data.shape)
