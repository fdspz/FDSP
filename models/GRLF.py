from torch.autograd import Function

# class ReverseLayerF(Function):

#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha

#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha

#         return output, None

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return ReverseLayerF.apply(x, lambda_)