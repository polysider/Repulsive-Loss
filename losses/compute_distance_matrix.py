import torch


def compute_distance_matrix(x1, x2):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:

        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

        Args:
            x1: first input tensor NxD
            x2: second input tensor MxD
            p: the norm degree. Default: 2

        Shape:
            - Input: :math:`(N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)`

        >>> input1 = Variable(torch.randn(100, 128))
        >>> input2 = Variable(torch.randn(20, 128))
        >>> output = compute_distance_matrix(input1, input2)
        >>> output.backward()
    """
    assert x1.size()[1] == x2.size()[1], "Input should have same feature dimensions"
    assert x1.dim() == 2, "Input must be a 2D matrix."

    N = x1.size()[0]
    M = x2.size()[0]

    x1_sum_of_squares = torch.sum(x1.pow(2), 1, keepdim=True)
    x2_sum_of_squares = torch.sum(x2.pow(2), 1, keepdim=True)

    # XX = torch.matmul(x1_sum_of_squares, Variable(torch.ones([1, M])))
    # YY = torch.matmul(Variable(torch.ones([N, 1])), torch.t(x2_sum_of_squares))

    XY = torch.matmul(x1, x2.t())

    # D = XX + YY - 2*XY

    D = x1_sum_of_squares + torch.t(x2_sum_of_squares) - 2*XY

    return D.pow(0.5)