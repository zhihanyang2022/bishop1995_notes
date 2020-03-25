# Least-square closed-form solution

<u>Author</u>: Zhihan Yang @ Carleton College

<u>Date</u>: Sunday, March 22, 2020 (during spring break)

The least-square method is a way to solve for the weights of a *generalized linear discriminant*, which is a form of single-layer networks.

## Generalized linear discriminant

The generalized linear discriminant is given by:
$$
y_{o}(\bold{x})=\sum_{i=0}^I w_{o, i} \phi_i(\bold{x})
$$

- $\phi_i(\bold{x})$: a feature function that takes a vector $\bold{x}$ as input and outputs a scalar.

## Loss function

The loss function $E$ of a weight vector $\bold{w}$ is defined as follows:
$$
E(\bold{w})=\frac{1}{2} \sum_{n=1}^N \sum_{o=1}^O \{t_{n, o} - \sum_{i=0}^I w_{o,i} \phi_{n, i} \}^2
$$

- $N$: the number of training examples
- $O$: the number of output nodes / the number of dependent variables to model
- $I$: the number of input notes / the number of independent variables to model
- $t_{n, o}$: the label for the $o$-th output of the $n$-th training example

- $w_{o, i}$: the weight connecting from the $i$-th input node to the $o$-th output node
- $\phi_{n, i}$: the value of $i$-th feature of the $n$-th training example

## Derivative of $E(\bold{w})$ with respect to $w_{o, i}$

Expand the summation over $O$:
$$
\begin{align}
E(\bold{w})&=\frac{1}{2} \sum_{n=1}^N \sum_{o=1}^O \{t_{n, o} - \sum_{i=0}^I w_{o,i} \phi_{n, i} \}^2 \\
&=\frac{1}{2} \sum_{n=1}^N 
[
(t_{n,1}-\sum_{i=0}^Iw_{1, i} \phi_{n,i})^2
+\cdots+\newline&
\color{red}{(t_{n,o}-\sum_{i=0}^Iw_{o, i} \phi_{n,i})^2}
+\cdots+\newline&
(t_{n,O}-\sum_{i=0}^Iw_{O, i} \phi_{n,i})^2]
\end{align}
$$
We see that only the term highlighted with red involves the term $w_{o, i}$. For now, let us only consider the derivative of this part with respect to $w_{o, i}$, since the derivative of a sum (over $N$) is equivalent to a sum of the derivatives of its consisting elements. Before we derive its derivative, let us further expand it:
$$
\begin{align}
E_{n, o}(\bold{w})
&=(t_{n,o}-\sum_{i=0}^Iw_{o, i} \phi_{n,i})^2 \\
&=(\sum_{i=0}^Iw_{o, i} \phi_{n,i}-t_{n,o})^2 \\

&=(w_{o, 0} \phi_{n, 0}
+\cdots+
\color{orange}{w_{o, i}} \color{blue}{\phi_{n, i}}
+\cdots+
w_{o, I} \phi_{n, I} - t_{n,o})^2 \\
\end{align}
$$
The terms that won’t be zero after differentiation are those that directly involve $w_{o, i}$:
$$
2w_{o, 0} \color{orange}{w_{o, i}} \phi_{n, 0} \color{blue}{\phi_{n, i}}
+\cdots+
w_{o, i}^\color{orange}{2} \phi_{n, i}^\color{blue}{2}
+\cdots+
2w_{o, I} \color{orange}{w_{o, i}}  \phi_{n, I} \color{blue}{\phi_{n, i}}
- 
2\color{orange}{w_{o, i}}  \color{blue}{\phi_{n, i}}t_{n,o}
$$
Taking the derivative of the long expression above with respect to $w_{o, i}$, obtain:
$$
\begin{align}
\frac{\partial E_{n, o}(\bold{w})}{\partial w_{o, i}} = 
2w_{o, 0} \phi_{n, 0} \color{blue}{\phi_{n, i}}
+\cdots+
2w_{o, i} \phi_{n, i}^\color{blue}{2}
+\cdots+
2w_{o, I} \phi_{n, I} \color{blue}{\phi_{n, i}}
- 
2\color{blue}{\phi_{n, i}}t_{n,o} 

\end{align}
$$
Factor, where $i’$ and $o’$ serve the same purpose as $i$ and $o$:
$$
=2[(\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'}) - t_{n, o}]\phi_{n, i}
$$
Therefore, 
$$
\frac{\partial E(\bold{w})}{\partial w_{o, i}} = \sum_{n=1}^N \{[(\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'}) - t_{n, o}]\phi_{n, i}\}
$$
We set the RHS to zero the solve for the value of $w_{o, i}$ that leads to zero loss:
$$
\sum_{n=1}^N \{[(\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'}) - t_{n, o}]\phi_{n, i}\}=0
$$
In order to find a solution to the equation above it is convenient to write it in a matrix notation. The challenge is how?

## Matrix notation

A matrix multiplication between two matrices can be denoted by:
$$
(AB)_{i,j}=\sum_{n} A_{i,n} B_{n,j}
$$
A matrix multiplication between three matrices can be denoted by:
$$
\begin{align}
(ABC)_{i,j}
&=\sum_{n} (AB)_{i,n} C_{n,j} \\
&=\sum_{n} ( \sum_{n'} A_{i,n'} B_{n,'j} ) C_{n,j}  
\end{align}
$$
Notice the similarity between $\frac{\partial E(\bold{w})}{\partial w_{o, i}}$ and the expansion of a triple-matrix multiplication as summations. Therefore, we can convert $\frac{\partial E(\bold{w})}{\partial w_{o, i}}$ into matrix notation involving a triple-matrix multiplication:
$$
\begin{align}
\sum_{n=1}^N \{[(\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'}) - t_{n, o}]\phi_{n, i}\} &=0 \\
\sum_{n=1}^N \{(\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'})\phi_{n, i} - t_{n, o}\phi_{n, i}\} &=0 \\
\sum_{n=1}^N (\sum_{i'=0}^Iw_{o', i'} \phi_{n, i'})\phi_{n, i} -\sum_{n=1}^N t_{n, o}\phi_{n, i} &= 0 \\
\end{align}
$$
To get the dimensions of the matrices right, align the axis alphabets:
$$
\begin{align}
\sum_{n=1}^N (\sum_{i'=0}^Iw_{o', i'} \phi_{i', n})\phi_{n, i} &= \sum_{n=1}^N t_{o, n}\phi_{n, i} \\
W^T \boldsymbol{\phi}^T \boldsymbol{\phi}&=T^T \boldsymbol{\phi}
\end{align}
$$
where the following shapes have been assumed:

- $W$: I, O
- $\boldsymbol{\phi}$: N, I
- $T$: N, O  
- so that $\boldsymbol{\phi} W$ would have the same shape as $T$.

To have the equation in terms of $W$ instead of $W_T$, transpose both sides:
$$
\begin{align}
(\boldsymbol{\phi}^T \boldsymbol{\phi}) W &= \boldsymbol{\phi}^T T \\
W &= (\boldsymbol{\phi}^T \boldsymbol{\phi})^{-1} \boldsymbol{\phi}^T T \\
&= \boldsymbol{\phi}^{\star} T
\end{align}
$$
where $\boldsymbol{\phi}^{*}=(\boldsymbol{\phi}^T \boldsymbol{\phi})^{-1}\boldsymbol{\phi}^T$ and is called the pseudo-inverse of $\boldsymbol{\phi}$.