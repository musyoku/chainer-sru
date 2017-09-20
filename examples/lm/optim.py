from chainer import optimizers

def get_current_learning_rate(opt):
	if isinstance(opt, optimizers.NesterovAG):
		return opt.lr
	if isinstance(opt, optimizers.MomentumSGD):
		return opt.lr
	if isinstance(opt, optimizers.SGD):
		return opt.lr
	if isinstance(opt, optimizers.Adam):
		return opt.alpha
	raise NotImplementedError()

def get_optimizer(name, lr, momentum):
	if name == "sgd":
		return optimizers.SGD(lr=lr)
	if name == "msgd":
		return optimizers.MomentumSGD(lr=lr, momentum=momentum)
	if name == "nesterov":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	raise NotImplementedError()

def decay_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.SGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.MomentumSGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return final_value
		opt.alpha *= factor
		return
	raise NotImplementedError()