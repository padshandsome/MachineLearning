Newton Raphson's Method and Logistic Regression

Newton Raphson's Method

Goal: to find the root of $f(x) = 0$ .

We have Taylor expansion of $f(x)$ at $x_0$:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \epsilon$$, where $\epsilon$ is the error.

Let the Taylor expansion equals to 0, the root can be approximated by 

$$x = x_0 - \frac{f(x_0)}{f'(x_0)}$$



Logistic Regression

Set ups:

$X \in R^{n\times m}$ :n samples, m-1 features datasets; By convention, we will add 1 dummy feature to the datasets.

$Y\in R^{n\times 1}$: n binary true labels;

$\beta\in R^{m\times 1}$: m-1 feature parameters; By convention, we will add 1 dummy feature to the datasets.

The likelihood for logistic regression is:

$ p(y=1|\beta;X) = \frac{1}{1+e^{-X\beta}}$

and also $p(y=0|\beta;X) = 1- \frac{1}{1+e^{-X\beta}} = \frac{e^{-X\beta}}{1+e^{-X\beta}}$







