# Introduction

Context is hospital and care home environements where it is often difficult to obtain data of critical situations such as falls due to the infrequency of it happening (tail of distribution) as well as privacy considerations.
Overall goal is improve performance of e.g. video classification tasks by training on synthetic data.
Synthetic data should capture the underlying distribution of the data, but be steerable to sample a subset of situations of interest.

We simplify the problem by explicitly modelling the 3D environment including human-to-environment and human-to-human interactions as a strong inductive bias for our generative diffusion model.

Public available datasets HumanML3D and InterHuman containing individual as well as two-person interactive human motion captures with associated text descriptions is used in combination with 3D scene reconstructions of hospital and carehome video captures as training data.

In order to recover the 3D geometry of the captured scenes, we make use of off-the-shelf models ProHMR and Depth Anything to generate per frame human pose and scene depth labels which is used as input, along with 2D keypoint annotations, to a joint optimization process unifying the coordinate systems and ensuring smooth trajectories yielding the final 3D scene reconstructions.

Important factors of success:

- Easily controllable to generate specific situations of interest
- Generative model is able to generalize and generate novel situations while still capturing the underlying data distribution, necessary for improving performance of e.g. down stream classification tasks.

# Background

# Denoising Diffusion Probabilistic Models

In recent years, several types of generative models such as Variational Autoencoders (VAEs), Generative Adverserial Networks (GANs), autoregressive models and flow-based models have shown remarkable results in data generation of varying data modalities, such as images, audio, videos and text. Most recently, Denoising Diffusion Probabilistic Models (DDPMs) have gained large popularity especially within the field of image generation due to several reasons such as high-quality data generation, versitility in several data domains as well as controllability, allowing one to steer the generation towards desired outputs.

A DDPM is a parametarized ($\theta$) Markov chain trained using variational inference to reverse a (forward) diffusion process, $q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)$ wherein the signal of the data, $\mathbf{x}_0$, is gradually destroyed by adding gaussian noise according to predefined noise schedule $\{\beta_t  \in (0,1) \}_{t=1}^T$ giving rise to increasingly noisy samples, $\mathbf{x}_1 \ldots \mathbf{x}_T$:

$$
    q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1 - \beta_t}\mathbf{x}_t,\beta_t \mathbf{I} ), \quad  q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0}) = \prod_{t=1}^T q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1})
$$

with $T$ being the discritized number of diffusion steps before all original information is completely discarded. The goal of the inverse or backwards process then becomes to iteratively remove the noise, in order to arrivate at the original data. More formally, the process is defined as:

$$
    p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert x_{t}), \quad p_\theta(\mathbf{x}_{t-1} \vert x_{t}) = \mathcal{N}(\mathbf{x}_{t-1}; \mathbf{\mu}_\theta(\mathbf{x}_t, t), \mathbf{\Sigma}_\theta(\mathbf{x}_t, t))
$$

taking starting point in pure noise $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$, incrementally removing the noise through the learned functions, $\mathbf{\mu}_\theta(\mathbf{x}_t, t)$ and $\mathbf{\Sigma}_\theta(\mathbf{x}_t, t)$ commonly parameterized by a deep neural network.

Using the reparameterization trick, we are able to sample any noisy version of our data, $\mathbf{x}_t$, at time step $t$ given our original data $\mathbf{x}_0$. Recall our forward transition probability function, $q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$. Letting $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ and using the reparameterization trick the expression can be rewritten as:

$$
    \mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}
$$

where $\epsilon_{t-1} \sim \mathcal{N}(0,1)$. Expanding the recursive definition then gives:

$$
\begin{align*}
    \mathbf{x}_t    & = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
                    & = \sqrt{\alpha_t} \left(
    \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2}
    \right) + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
                    & = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\epsilon_{t-2} + \sqrt{1 - \alpha_{t}} \epsilon_{t-1} \\
                    & = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t}\alpha_{t-1}}\bar{\epsilon}_{t-2}
\end{align*}
$$

where $\bar{\epsilon}_{t-2}$ merges the two independent Gaussians $\epsilon_{t-1}$ and $\epsilon_{t-2}$ into a single Gaussian with new variance as the sum of variances $\alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_{t}) = 1 - \alpha_t \alpha_{t-1}$. Recursively applying the definition of $\mathbf{x}_t$ and merging the gaussian noise terms results in the simplified expression referred to as the _neat property_:

$$
    \mathbf{x}_t = \sqrt{\bar{\alpha}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}} \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

or by undoing the reparameterization:

$$
    q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

The model is trained by minimizing the Evidence Lower Bound (ELBO) on the negative likelihood:

$$
    \mathcal{L}_{VLB}
        = \mathbb{E}_q\left[
            - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)}
        \right]
        = \mathbb{E}_q\left[
            - \log p(\mathbf{x_T}) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_{t})}{q(\mathbf{x}_t \vert \mathbf{x}_{t-1})}
        \right]
        \leq \mathbb{E}\left[- \log p_\theta(\mathbf{x}_0)\right]
$$

which can be rewritten as the sum of KL-divergences

## Classifier Free Guidance

<!--

    Writ stuff here
!>

The denoising network is trained variational inference, by minimizing the _simple objective_, $\mathcal{L}_\text{simple} = \Vert \mathbf{\epsilon}_t - \mathbf{\epsilon}_\theta(\sqrt{\bar{\alpha}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}} \mathbf{\epsilon}_t, t ) \Vert^2$.

approximation of the variational inference in order to maximize the Evidence Lower Bound (ELBO), shown emperically to achieve good results.

## Forward process

# SMPL model

#
$$
