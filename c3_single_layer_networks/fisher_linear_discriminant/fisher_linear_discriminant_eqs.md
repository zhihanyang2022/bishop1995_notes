# Fisher linear discriminant equations

$$
J(\vec{w})=\frac{\vec{w} S_{B} \vec{w}}{\vec{w} S_{W} \vec{w}}
$$

$$
\frac{\partial J(\vec{w})}{\partial \vec{w}}=\frac{u^{\prime} v-u v^{\prime}}{v^{2}}
$$

---

$$
u=\vec{w}^{\top} S_{B} \vec{w}
$$

$$
u^{\prime}=\left(S_{B}+S_{B}^{\top}\right) \vec{w}=2 S_{B} \vec{w}
$$

Derivation:
$$
\begin{aligned} S_{B} &=\left(\vec{m}_{2}-\vec{m}_{1}\right)\left(\vec{m}_{2}-\vec{m}_{1}\right)^{T} \\ 

S_{B}^{\top} &= \left( \left(\vec{m}_2-\vec{m}_{1}\right)\left(\vec{m}_{2}-\vec{m}_{1}\right)^{T} \right)^T \\ 

&=\left(\vec{m}_{2}-\vec{m}_{1}\right)^{T T}\left(\vec{m}_2-\vec{m}_{1}\right)^{T} \\ &=\left(\vec{m}_{2}-\vec{m}_{1}\right)\left(\vec{m}_{2}-\vec{m}_{1}\right)^{T} \\ 

\end{aligned}
$$
Therefore, $S_B^T = S_B$.

---

$$
v=\vec{w}^{\top} S_{W} \vec{w}
$$

$$
v^{\prime}=\left(S_{W}+S_{W}^{T}\right) \vec{w} = 2 S_{W} \vec{w}
$$

Derivation:
$$
S_{W} = \sum_{n \in C_{1}}\left(\vec{x}^{n}-\vec{m}_{1}\right)\left(\vec{x}^{n}-\vec{m}_{1}\right)^{T}+

\sum_{n \in C_{2}}\left(\vec{x}^{n}-\vec{m}_{2}\right)\left(\vec{x}^{n}-\vec{m}_{2}\right)^{T} \\
$$
Since a transpose of sums is a sum of transposes,
$$
\begin{aligned} 

S_{W}^{T} 

&=\sum_{n \in C_{1}}\left[\left(\vec{x}^{n}-\vec{m}_{1}\right)\left(\vec{x}^{n}-\vec{m}_{1}\right)^{T}\right]^{T}+

\sum_{n \in C_{2}}\left[\left(\vec{x}^{n}-m_{2}\right)\left(\vec{x}^{n}-\vec{m}_{2}\right)^{T}\right]^{T} \\ 

&= \sum_{n \in C_{1}}\left(\vec{x}^{n}-\vec{m}_{1}\right)\left(\vec{x}^{n}-\vec{m}_{1}\right)^{T}+

\sum_{n \in C_{2}}\left(\vec{x}^{n}-\vec{m}_{2}\right)\left(\vec{x}^{n}-\vec{m}_{2}\right)^{T} \\

\end{aligned}
$$
Therefore, $S_{W}^T = S_{W}$.

---

$$
\begin{align}

\frac{\partial J(\vec{w})}{\partial \vec{w}}

&=\frac{u^{\prime} v-u v^{\prime}}{v^{2}} \\ 

&=\frac{\left(2 S_{B} \vec{w}\right)\left(\vec{w}^{\top} S_W \vec{w}\right)-\left(\vec{w}^{\top} S_{B} \vec{w}\right)\left(2 S_{w} \vec{w}\right)}{\left(\vec{w}^{\top} S_{W} \vec{w}\right)\left(\vec{w}^{\top} S_{W} \vec{w}\right)} \\ 

&=\frac{2 S_{B} \vec{w}}{\vec{w}^{\top} S_{W} \vec{w}}-\frac{\left(\vec{w}^{\top} S_{B} \vec{w}\right)\left(2 S_{W} \vec{w}\right)}{\left(\vec{w}^{\top} S_{W} \vec{w}\right)\left(\vec{w}^{\top} S_{W} \vec{w}\right)}

\end{align}
$$

---

$$
\begin{align}

\frac{\partial J(\vec{w})}{\partial \vec{w}}

&=0 \\

\frac{2 S_{B} \vec{w}}{\vec{w}^{T} S_{w} \vec{w}}

&=\frac{\left(\vec{w}^{T} S_{B} \vec{w}\right)\left(2 S_{w} \vec{w}\right)}{\left(\vec{w}^{T} S_{w} \vec{w}\right)\left(\vec{w}^{T} S_{w} \vec{w}\right)} \\


\underbrace{2\left(\vec{w}^{\top} S_{w} \vec{w}\right)}_{\text {scalar }} \left( S_{B} \vec{w}\right)

&=\underbrace{2 \left(\vec{w}^{\top} S_{B} \vec{w}\right)}_{\text {scalar }} \left(S_{w} \vec{w}\right)

\end{align}
$$

Since we don’t care about the magnitude of $\vec{w}$, only its direction, we remove the scalars:
$$
S_{B} \vec{w} \propto S_{W} \vec{w}
$$
Since $S_B = (\vec{m}_2 - \vec{m}_1)(\vec{m}_2 - \vec{m}_1)^T$, $S_B \vec{w}$ is always in the direction of $\vec{m}_2-\vec{m}_1$ because $(\vec{m}_2 - \vec{m}_1)^T \vec{w}$ is a scalar:
$$
(\vec{m}_2 - \vec{m}_1)\propto S_{W} \vec{w}
$$

$$
\vec{w} \propto S_{W}^{-1}(\vec{m}_2 - \vec{m}_1) 
$$

This choice of $\vec{w}$ is known as Fisher’s linear discriminant, although it is strictly not a discrminant but rather a specific choice of direction for projection of data down to one dimension.

