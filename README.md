# Local monotone operator learning using non-monotone operators: MnM-MOL


## Getting started

The recovery of magnetic resonance (MR) images from undersampled measurements is a key problem that has been the subject of extensive research in recent years. Unrolled approaches, which rely on end-to-end training of convolutional neural network (CNN) blocks within iterative reconstruction algorithms, offer state-of-the-art performance. These algorithms require a large amount of memory during training, making them difficult to employ in high-dimensional applications. Deep equilibrium (DEQ) models and the recent monotone operator learning (MOL) approach were introduced to eliminate the need for unrolling, thus reducing the memory demand during training. Both approaches require a Lipschitz constraint on the network to ensure that the forward and backpropagation iterations converge. Unfortunately, the constraint often results in reduced performance compared to the unrolled methods. The main focus of this work is to relax the constraint on the CNN block in two different ways. Inspired by convex-non-convex regularization strategies, we now impose the monotone constraint on the sum of the gradient of the data term and the CNN block, rather than constrain the CNN itself to be a monotone operator. This approach enables the CNN to learn possibly non-monotone score functions, which can translate to improved performance.  In addition, we only restrict the operator to be monotone in a local neighborhood around the image manifold. Our theoretical results show that the proposed algorithm is guaranteed to converge to the fixed point and that the solution is robust to input perturbations, provided that it is initialized close to the true solution. Our empirical results show that the relaxed constraints translate to improved performance and that the approach enjoys robustness to input perturbations similar to MOL.


## Relevant paper

[Local monotone operator learning using non-monotone operators: MnM-MOL](https://arxiv.org/abs/2312.00386)
