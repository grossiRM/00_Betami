#!/usr/bin/env python
# coding: utf-8
# %%

# # Exploration of adding noise to observations (or not) when using the Iterative Ensemble Smoother (iES)

# ### We can start out with an objective function.

# Let's make a couple definitions:
# $\boldsymbol{y}$ are the observation values
# $\boldsymbol{X}$ is the action of the model and $\boldsymbol{\beta}$ is the current set of model parameters  
# Then, $\mathbf{y}\mathbf{-}\mathbf{X}\boldsymbol{\beta}$ is the discrepancy (e.g. residual or error) between model outputs and observations.
# 
# We want to minimize those, and for a variety of reasons, we actually want to minimize the square of those errors, and that set of squared errors is $\Phi$, called the "objective function"
# 
# $$\Phi\left(\boldsymbol{\beta}\right) =  \left(\mathbf{y}\mathbf{-}\mathbf{X}\boldsymbol{\beta}\right)^{T}\left(\mathbf{y}\mathbf{\mathbf{-}}\mathbf{X}\boldsymbol{\beta}\right) = \sum_{i=1}^{N_{obs}}\left( y_i - \sum_{j=1}^{N_{par}} X_{ij}\beta_{j}\right)^2$$
# 
# But....we know that not each observation is of equal value/information/importance/certainty, so we can use weights to amplify or suppress the contributions from specific observations. We can assign weights as:  
# $$\omega=\frac{1}{\sigma}$$
# or, in matrix form, the weights are defined as $\boldsymbol{Q}$
# $$\Phi\left(\boldsymbol{\beta}\right)=\left(\mathbf{y}\mathbf{-}\mathbf{X}\boldsymbol{\beta}\right)^{T}\mathbf{Q}\left(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\right) = \sum_{i=1}^{N_{obs}}\left(\omega_i \left(y_i - \sum_{j=1}^{N_{par}} X_{ij}\beta_{j}\right)\right)^2$$  
# 

# %%


from wvn_helper import plot_mod_obs


# Without observation noise, we hope that the modeled output distribution will overlap (or "cover") the single observation value.

# %%


plot_mod_obs(6)


# If the modeled output distribution does _not_ overlap the observed value, that observation is in "prior data conflict".

# %%


plot_mod_obs(11,pdc=True)


# But what if we set an observation weight for the obs? 
# 
# Based on general background, we assume 
# $\omega \approx \frac{1}{\sigma}$
# 
# So, sampling the noise around the observation results in 
# $\sigma = \frac{1}{\omega}$
# 
# For example, if this is a head observation, 
# $\sigma=0.5m$ is a reasonable starting point might be $\omega=\frac{1}{.5}=2.0$
# 

# %%


plot_mod_obs(6, noisy=True, std=.5)


# %%


plot_mod_obs(11, noisy=True, std=.5, pdc=True)


# That's all good, but what if we adjust weights to balance the objective function? This is a common approach and perfectly defensible, but say we decrease the weight on the heads quite a bit. is $\omega \approx \frac{1}{\sigma}$ still valid?
# 
# If we adjust $\omega$ to be $0.2$ then $\sigma = \frac{1}{\omega} = \frac{1}{0.2} = 5.0$

# %%


plot_mod_obs(11, noisy=True, std=[.5,5] )


# So, there are pretty important implications in how we choose observation noise. This can make the difference between a realistic representation of the noise and an unrealistic (and really, uninformative) level of spread in the observation values. Further, this decision can make the difference between concluding that an observation is in prior-data-conflict or not. With many observations, the implications of this are very important.
