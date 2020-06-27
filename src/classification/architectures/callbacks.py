from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1.,
                 reduce_on_plateau=0, monitor='val_loss', reduce_factor=2,
                 max_momentum=0.95, min_momentum=0.85, verbose=1):
        """
        References:
            Original Paper: https://arxiv.org/abs/1803.09820
            Blog Post: https://sgugger.github.io/the-1cycle-policy.html
            Code Reference:
                https://github.com/bckenstler/CLR
                https://github.com/amaiya/ktrain/blob/master/ktrain/lroptimize/triangular.py
        """
        super(Callback, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** x
            self.scale_mode = 'iterations'
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self.orig_base_lr = None
        self.wait = 0

        # restoring weights due to CRF bug
        self.best_weights = None

        # LR reduction
        self.verbose = verbose
        self.patience = reduce_on_plateau
        self.factor = 1. / reduce_factor
        self.monitor = monitor
        if 'acc' not in self.monitor:
            self.monitor_op = lambda a, b: np.less(a, b)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b)
            self.best = -np.Inf

        # annihalting LR
        self.overhump = False

        # cyclical momentum
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        if self.min_momentum is None and self.max_momentum:
            self.min_momentum = self.max_momentum
        elif self.min_momentum and self.max_momentum is None:
            self.max_momentum = self.min_momentum
        self.cycle_momentum = True if self.max_momentum is not None else False

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
        self.orig_base_lr = self.base_lr

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

        # annihilate learning rate
        prev_overhump = self.overhump
        iterations = (self.clr_iterations + 1) % (self.step_size * 2)
        if iterations / self.step_size > 1:
            self.overhump = True
        else:
            self.overhump = False
        if not prev_overhump and self.overhump:
            self.base_lr = self.max_lr / 1000
        elif prev_overhump and not self.overhump:
            self.base_lr = self.orig_base_lr

        # set momentum
        if self.cycle_momentum:
            if self.overhump:
                current_percentage = 1. - ((iterations - self.step_size) / float(
                    self.step_size))
                new_momentum = self.max_momentum - current_percentage * (
                        self.max_momentum - self.min_momentum)
            else:
                current_percentage = iterations / float(self.step_size)
                new_momentum = self.max_momentum - current_percentage * (
                        self.max_momentum - self.min_momentum)
            K.set_value(self.model.optimizer.beta_1, new_momentum)
            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.beta_1))

    def on_epoch_end(self, epoch, logs=None):
        if self.patience:
            current = logs.get(self.monitor)
            if current is None:
                raise Exception('cannot monitor %s' % self.monitor)
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    min_lr = 1e-7
                    current_lr = float(K.get_value(self.model.optimizer.lr))
                    if self.max_lr > min_lr:
                        self.base_lr = self.base_lr * self.factor
                        self.max_lr = self.max_lr * self.factor
                        new_lr = current_lr * self.factor
                        new_lr = max(new_lr, min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose:
                            print('\nEpoch %05d: Reducing Max LR on Plateau: '
                                  'new max lr will be %s (if not early_stopping).' % (epoch + 1, self.max_lr))
                        self.wait = 0
