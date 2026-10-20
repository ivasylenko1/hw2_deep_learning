import torch

# Implementation very largely inspired by https://medium.com/@yehyafarhat_3213/how-to-build-your-own-custom-optimizer-in-pytorch-step-by-step-58fbe987cfae
class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.):
        assert lr > 0., "learning rate must be positive"
        assert betas[0] >= 0. and betas[0] < 1., "beta1 must be inside [0, 1)"
        assert betas[1] >= 0. and betas[1] < 1., "beta2 must be inside [0, 1)"
        assert eps > 0., "epsilon must be greater than 0"
        assert weight_decay >= 0., "weight_decay must be greater or equal to 0"

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                if group['weight_decay'] != 0:    # AdamW weight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(group['beta1']).add_(p.grad, alpha=1 - group['beta1'])   # exp_avg = exp_avg * group['beta1'] + p.grad * (1-group['beta1'])
                exp_avg_sq.mul_(group['beta2']).addcmul_(p.grad, p.grad, value=1 - group['beta2'])   # exp_avg_sq = exp_avg_sq * group['beta2'] + p.grad * p.grad * (1-group['beta2'])

                m_hat = exp_avg / (1 - group['beta1'] ** t)   # m_hat = exp_avg / (1 - group['beta1'] ** t)
                v_hat = exp_avg_sq / (1 - group['beta2'] ** t)   # v_hat = exp_avg_sq / (1 - group['beta2'] ** t)

                p.sub_(group['lr'] * m_hat / (v_hat.sqrt() + group['eps']))   # p = p - group["lr"] * m_hat / (torch.sqrt(v_hat) + group["eps"])

