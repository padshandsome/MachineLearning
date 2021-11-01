EM Algorithm

$\underline{Motivating\ Example}$

This problem is inspired by a signal channel disruption problem:

We observe $y = [y_1,y_2,...,y_N]^T$, generated i.i.d from $w=[w_1,w_2,...,w_N]^T$

Goal: $(\delta^{\star},\epsilon^{\star}) = \underset{\delta,\epsilon}{argmax} P_Y(y;\epsilon,\delta)$

Using total probability formula, for every single observed $y_i$ , we have

$P_Y(y_i,\epsilon,\delta) = \sum_{w_i}P_W(w_i,\delta)P_{Y|W}(y_i|w_i;\epsilon) =P_W(0;\delta)P_{Y|0}(y_i|0;\epsilon) + P_W(1;\delta)P_{Y|0}(y_i|1;\epsilon) $

$=(1-\delta)\epsilon^{y_i}(1-\epsilon)^{1-y_i}+\delta(1-\epsilon)^{y_i}\epsilon^{1-y_i}$

Then the likelihood is 

$P_Y(y;\epsilon,\delta) = \prod_{i=1}^{N}P_Y(y_i;\epsilon,\delta) = \prod_{i=1}^N(1-\delta)\epsilon^{y_i}(1-\epsilon)^{1-y_i}+\delta(1-\epsilon)^{y_i}\epsilon^{1-y_i}$

Then we try to take the deravative of the loglikelihood

$logP_Y(y;\epsilon,\delta) = \sum_{i=1}^Nlog((1-\delta)\epsilon^{y_i}(1-\epsilon)^{1-y_i}+\delta(1-\epsilon)^{y_i}\epsilon^{1-y_i})$

Because it has summation in between, so it can not be factorized, thus it is hard to compute brutely.



$\underline{Problem\ Setup}$

$\underline{y}$: Observed Data $\Rightarrow$ y above

$\underline{x}$: Unknown parameter (Deterministic) $\Rightarrow$ $\epsilon,\delta$ above

$\underline{z}$: Latent/ Hidden RVs $\Rightarrow$ w above

Goal: $\hat{x} = \underset{x}{argmax}\ P_Y(y;x) = \underset{x}{argmax}\ log P_y(y;x) \triangleq \underset{x}{argmax}\ l(y;x)$

From above motivating problem, it's hard to solve the optimization problem directly. But we can consider to construct the lower bound of loglikelihood, find the maximum of the lower bound.

However, generally finding the maxium of the lower bound cannot guarantee to finding the maxium of any functions above the bound. But we can prove with this setup that the bound is tight, thus leads to the problem solved.



$\underline{Construct\ Lower\ Bound}$

$l(y;x) = logP_Y(y;x) = log(\sum_ZP_{Y|Z}(y|z;x)P_Z(z;x)) = log(\sum_ZP_{Y,Z}(y,z;x))$

$= log(\sum_Zq(z|y)\frac{P_{Y,Z}(y,z;x)}{q(z|y)})$, where $q(z|y)$ is non-zero, arbitrary function over z, depending on y.

We can notice that $q(z|y)\frac{P_{Y,Z}(y,z;x)}{q(z|y)}$ is convex combination, and log function is concave function. Then we can use Jensen Inequality:

$log(\lambda z_1+(1-\lambda)z_2) \geq \lambda log(z_1) +(1-\lambda)log(z_2)$

$\Rightarrow l(y;x) \geq \sum_Zq(z|y)log\frac{P_{Y,Z}(y,z;x)}{q(z|y)} \triangleq L(q;x) \ \ \ \forall q(\cdot),x$



$\underline{EM\ Algorithm}$

(E-Step)         $q^{(t+1)} = \underset{q(\cdot)}{argmax}\ L(q,x^{(t)})$

(M-Step)        $x^{(t+1)} = \underset{x}{argmax}\ L(q^{(t+1)},x)$



We start with the M-step (order doesn't matter)

$L(q^{(t+1)},x) = \sum_Zq^{(t+1)}(z|y)(logP_{Y,Z}(y,z;x)-logq^{(t+1)}(z|y))$ 

The last term has nothing to do with x, so we can just ignore it.

$x^{(t+1)} = \underset{x}{argmax}\ L(q^{(t+1)},x) = \underset{x}{argmax}\ \sum_Zq^{(t+1)}(z|y)logP_{Y,Z}(y,z;x) = \underset{x}{argmax}\ E_{q^{(t+1)}(z|y)}[logP_{Y,Z}(y,z;x)]$ 



Then compute the M-step:

$L(q,x^{(t)}) = \sum_Zq(z|y)log\frac{P_{Y,Z}(y,z;x^{(t)})}{q(z|y)} = \sum_Zq(z|y)log\frac{P_{Z|Y}(z|y;x^{(t)})P_Y(y;x^{(t)})}{q(z|y)}$ 

=$\sum_Zq(z|y)(log\frac{P_{Z|Y}(z|y;x^{(t)})}{q(z|y)}+logP_Y(y;x^{(t)})) = \sum_Zq(z|y)log\frac{P_{Z|Y}(z|y;x^{(t)})}{q(z|y)}+\sum_Zq(z|y)logP_Y(y;x^{(t)})$ 

$=\sum_Zq(z|y)log\frac{P_{Z|Y}(z|y;x^{(t)})}{q(z|y)}+logP_Y(y;x^{(t)}) $

For the first part, use Gibbs' Inequality:

$E_P[logp(z)]\geq E_P[logq(z)]$ with equality if and only if $p(z) = q(z)$

Then the first part be like

$\sum_Zq(z|y)log\frac{P_{Z|Y}(z|y;x^{(t)})}{q(z|y)}\leq 0$

Remember our goal is to maximize the $L(q,x^{(t)})$, so we need to reaches the 0. We just let 

$q^{(t+1)}(z|y) = P_{z|y}(z|y;x^{(t)})$ , q function be the posterior.  Under this assumption, we substitute the new q into $L$.

$L(q^{(t+1)},x^{(t)}) = logP_Y(y;x^{(t)})=l(y;x^{(t)})$

Thus we prove that with every iteration of the two steps, we can always reach the maximum of $l(y;x^{(t)})$, which exactly is our goal.

$\star $ However, in practice, generally it is hard to compute the posterior because its normalization term. May need some approximation on posterior.



$\underline{Reiterate\ Motivating\ Example}$

$\underline{K\ Means}$

$\underline{K\ Neareast\ Neighbor }$ 



