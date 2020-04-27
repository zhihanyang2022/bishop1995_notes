# bishop1995_c6_cde_estimation

## Section 1

$$
\begin{aligned} 

s^{2}(x) 

&=\left\langle\|t-\langle t | x\rangle\|^{2} | x\right\rangle \\ 

&=\int\|t-\langle t | x\rangle\|^{2} p(t | x) d t \\ 

&=\int\|t-\langle t | x\rangle\|^{2}
\underbrace{
	\left(\sum_{j} \alpha_{j}(x) \phi_{j}(t | x)\right)
}_{\text{Definition of } \langle t|x \rangle}
d t \\ 

&=\sum_{j} \alpha_{j}(x) \int\|t-\langle t | x\rangle\|^{2} \phi_{j}(t | x) d t \\

&=\sum_{j} \alpha_{j}(x) \int\left[\left(t-\mu_{j}(x)\right)+\left(\mu_{j}(x)-\langle t | x\rangle\right)\right]^{2} \phi_{j}(t | x) d t \\

&=\sum_{j} \alpha_{j}(x) \int
\left[
	\left(t-\mu_{j}(x)\right)^{2}+2\left(t-\mu_{j}(x)\right)\left(\mu_{j}(x)-\langle t | x \rangle \right)+\left(\mu_{j}(x)-\langle t | x \rangle \right)^{2}
\right]
\phi_{j}(t | x) d t \\

&=\sum_{j} \alpha_{j}(x)
(
\underbrace{\int\left(t-\mu_{j}(x)\right)^{2} \phi_{j}(t | x) d t}_{\phi_{j} \text { is a Gaussian;} \text { this is } \sigma_{j}^{2}(x).}
\\
&+2
\underbrace{\int \left(t-\mu_{j}(x)\right)\left(\mu_{j}(x)-\langle t | x\rangle\right) \phi_{j}(t | x) d t}_{\text{ Equals 0. See section 2. }}
\\
&+
\underbrace{\int\left(\mu_{j}(x)-\langle t | x\rangle\right)^{2}\phi_{j}(t | x) d t}_{\text{Equals }\left(\mu_{j}(x)-\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right)^{2}. \text{ See section 3.}}
\\

&=\sum_{j} \alpha_{j}(x)\left( \sigma_j(x)^2 + \left(\mu_{j}(x)-\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right)^{2} \right)

\end{aligned}
$$

## Section 2



## Section 3







For a specific $j$:

hoho
$$
\begin{aligned} 
& \int t\langle t | x\rangle \phi_{j}(t | x) d t \\

=& \int t
	\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right)
\phi_{j}(t | x) d t \\

=&\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right) 
\underbrace{\int t \phi_{j}(t | x) d t}_{\phi_j \text{ is a Gaussian; this equals } \mu_j.} \\

=&\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right) \mu_{j}(x) 
\end{aligned}
$$
hoho
$$
\begin{aligned} & 
\int \mu_{j}(x)\langle t | x\rangle \phi_{j}(t | x) d t \\
=& \int \mu_{j}(x)\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right) \phi_{j}(t | x) d t \\
=& \mu_{j}(x)\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right) \underbrace{\int\phi_{j}(t | x) d t}_{\text{Gaussian integrates to 1.}} \\
=&\left(\sum_{j} \alpha_{j}(x) \mu_{j}(x)\right) \mu_{j}(x) 
\end{aligned}
$$
hoho

