# sde-hjb-solver

This Python repository contains the implementation of the finite difference method for solving the Hamilton-Jacobi-Bellman (HJB) equation associated to the importance sampling (IS) problem of diffusion processes.

## Setting

### Importance sampling problem
Consider the stochastic process in $\mathbb{R}^d$ following the controlled dynamics for a given potential landscape $U_\text{pot}$:

$$\mathrm{d} X_s^u = (-\nabla U_\text{pot}(X_s^u) + \sigma(X_s^u) u(X_s^u))\mathrm{d}s + \sigma(X_s^u) \mathrm{d}W_s, \qquad X_0^u = x.$$

We aim to estimate the following expectation value by doing importance sampling

$$\Psi(x) = \mathbb{E}^x \bigl[I(X) \bigr] = \mathbb{E}^x \bigl[I(X^u) M^u \bigr], \quad I(X) = \exp \Bigl( -g(x_{\tau}) - \int\limits_0^{\tau} f(X_t) \mathrm{d}t \Bigl),$$

where $\tau$ is the first hitting time of the set $D \subset \mathbb{R}^d$ and $M^u$ is the corresponding exponential Martingale provided by the Girsanov formula.

Every control $u \in \mathcal{U}$ provides us with an unbiased estimator of $\Psi(x)$. We want to find the $u^* \in \mathcal{U}$ which minimizes the variance of the importance sampling estimator

$$u^* \in \text{argmin}_u \{ \text{Var}^x \bigl(I(X^u) M^u \bigr) \}.$$

### HJB equation
It is well known that the quantity that we want to estimate satisfies the following BVP on the domain $\mathcal{O} \coloneqq \mathcal{D} \cap D^c$ 

$$(\mathcal{L} -f(x)) \Psi(x) = 0 \quad \forall x \in \mathcal{O}, \quad \Psi(x) = \exp(-g(x)) \quad \forall x \in \partial{\mathcal{O}},$$

where $\mathcal{L}$ denotes the infinitesimal generator of the original not controlled process i.e. case $u=0$.

## Contains
Finite difference method for the 1d and 2d cases where the original stochastic dynamics follow the overdamped langevin equation for different potential landscapes. The following settings are considered
* Moment generation function (MGF),
* Committor function.

See Examples for details.

## Install

1) clone the repo 
```
git clone https://github.com/riberaborrell/sde-hjb-solver.git
```

2) move inside the directory, create virtual environment and install required packages
```
cd sde-hjb-solver
make venv
```

3) activate venv
```
source venv/bin/activate
```

4) create .env file containing the path of your data folder
(not even need if the data folder is just data/ inside the project) 
```
SDE_HJB_DATA_DIR=/data/abs/path
```


## Developement

in step 2) also install developement packages
```
make develop
```

## Examples
#### 1d double well and mgf setting
Overdamped langevin dynamics, i.e. $\sigma = \sqrt{2 \beta^{-1}}$, with the 1-dimensional double well potential and the momgent generating function (MGF) setting 

$$\Psi_\lambda(x) = \mathbb{E}^x[\exp(\lambda \tau)], \quad (f=-\lambda, \quad g=0, \quad \tau=\tau_C),$$

for $\lambda=-1$, target set $C = [1, \infty)$ and $\beta=1$.


```
python src/sde_hjb_solver/compute_hjb_solution_1d_st.py --setting mgf --alpha 1 --beta 1 --h 0.001 --plot
```
<table>
  <tr>
    <td><img src="examples/img/doublewell_1d_potential.png" width=400 height=360></td>
    <td><img src="examples/img/doublewell_1d_mgf_psi.png" width=400 height=360></td>
  </tr>
  <tr>
    <td><img src="examples/img/doublewell_1d_mgf_perturbed-potential.png" width=400 height=360></td>
    <td><img src="examples/img/doublewell_1d_mgf_control.png" width=400 height=360></td>
 </table>

#### 2d asymmetric double well and mgf setting
Overdamped langevin dynamics with the asymmetric 2-dimensional double well potential, $\alpha=(1, 2)$, MGF setting, $\beta=1$ and target set

$$C = \{x \in \mathbb{R}^2 \mid U_\text{pot} < 0.25, \quad x_1 >0, \quad x_2 > 0 \}.$$

```
python src/sde_hjb_solver/compute_hjb_solution_2d_st.py --setting mgf --alpha 1 2 --beta 1 --h 0.05 --plot
```
<table>
  <tr>
    <td><img src="examples/img/doublewell_2d_potential.png" width=400 height=360></td>
    <td><img src="examples/img/doublewell_2d_mgf_psi.png" width=400 height=360></td>
  </tr>
  <tr>
    <td><img src="examples/img/doublewell_2d_mgf_perturbed-potential.png" width=400 height=360></td>
    <td><img src="examples/img/doublewell_2d_mgf_control.png" width=400 height=360></td>
  </tr>
 </table>

 #### 2d triple well and committor setting
Overdamped langevin dynamics with the 2-dimensional triple well potential and the committor setting

$$\Psi(x) = q_\text{AB}(x) = \mathbb{P}^x[\tau_B < \tau_A], \quad (f=0, \quad g=-log 1_B, \quad \tau=\tau_A \wedge \tau_B).$$

with source set $A$ and hitting set $B$ given by
$$A = \{x \in \mathbb{R}^2 \mid U_\text{pot} < -3.5, \quad x_1 <0 \},$$
$$B = \{x \in \mathbb{R}^2 \mid U_\text{pot} < -3.5, \quad x_1 >0 \}.$$

```
python src/sde_hjb_solver/compute_hjb_solution_2d_st.py --problem triplewell --setting committor --beta 1 --h 0.05 --plot
```

<table>
  <tr>
    <td><img src="examples/img/triplewell_2d_potential.png" width=400 height=360></td>
    <td><img src="examples/img/triplewell_2d_committor_psi.png" width=400 height=360></td>
  </tr>
  <tr>
    <td><img src="examples/img/triplewell_2d_committor_perturbed-potential.png" width=400 height=360></td>
  </tr>
 </table>

