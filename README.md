# Spatio-Temporal-Graph-Convolutional-Networks-A-Deep-Learning-Framework-for-Traffic-Forecasting

# reference
- https://www.zhihu.com/question/54504471
- https://en.wikipedia.org/wiki/Laplacian_matrix
- https://tkipf.github.io/graph-convolutional-networks/
- https://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/
- http://cs229.stanford.edu/section/cs229-moregaussians.pdf

# abstract
- Spatio-Temporal Graph Convolutional Network
- tackle the time series prediction problem in traffic domain
- complete convolutional structures.

# introduction
- linear regression perform well on short interval forecast instead of long terms
- this is a data-driven and using spotio-temporal information method.
- fully utilize spatio-information instead of treating it as discrete units
- $$\hat v_{t+1},...,\hat v_{t+H}  = argmax log_{10} P(v_{t+1},...,v_{t+H}|v_{t-M},...,v_{t})$$
- ![function one](Selection_001.png)
- where $$v_t \in R^n$$, n is an observation vector of n road segments at time step t

## Convolutions on Graphs
![convolution on graphs](Selection_002.png)

- normalized Laplacian
	- Random walk normalized Laplacian
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6dd08f5980b0d529e9e9413b41093a0195bb92e2)
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/33096ab8c4e5ada54c3429767b9275f96629a934)
	- analogy to The Multivariate Gaussian Distribution 
![function one](Selection_003.png)
	- Symmetric normalized Laplacian  L:
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f9007674eecb50de92fe6aadceee5df23c834b66)
![laplacian](https://wikimedia.org/api/rest_v1/media/math/render/svg/4ab36f74a92195f5be3814f444442270977b1f11)


- first generation of GNC
![](https://www.zhihu.com/equation?tex=+y_%7Boutput%7D%3D%5Csigma+%5Cleft%28U%5Cleft%28%5Cbegin%7Bmatrix%7D%5Ctheta_1+%26%5C%5C%26%5Cddots+%5C%5C+%26%26%5Ctheta_n+%5Cend%7Bmatrix%7D%5Cright%29+U%5ET+x+%5Cright%29+%5Cqquad%283%29)

- second generation of GNC
![](https://www.zhihu.com/equation?tex=+y_%7Boutput%7D%3D%5Csigma+%5Cleft%28U%5Cleft%28%5Cbegin%7Bmatrix%7D%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+%5Clambda%5Ej_1+%26%5C%5C%26%5Cddots+%5C%5C+%26%26+%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+%5Clambda%5Ej_n+%5Cend%7Bmatrix%7D%5Cright%29+U%5ET+x+%5Cright%29+%5Cqquad%284%29)
![](https://www.zhihu.com/equation?tex=U+%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+%5CLambda%5Ej+U%5ET+%3D%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+U%5CLambda%5Ej+U%5ET+%3D+%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+L%5Ej)
![](https://www.zhihu.com/equation?tex=+y_%7Boutput%7D%3D%5Csigma+%5Cleft%28+%5Csum_%7Bj%3D0%7D%5EK+%5Calpha_j+L%5Ej+x+%5Cright%29+%5Cqquad%285%29)
![](https://pic4.zhimg.com/80/v2-5f756da1ce39f38d408bd771a15c8ad3_hd.jpg)
![](https://pic4.zhimg.com/80/v2-a13b82907a364c3707a18bb8572b3a63_hd.jpg)
	- if k == n, receptive field is n hop 
- third generation of GNC
![](Selection_004.png)
	- where $$c_1$$, $$c_2$$ and $$c_3$$ are fixed
	- The only trainable parameters are $$\theta_0$$ and $$\theta_1$$
	- in the final version the authors even further fix $$\theta_0 = -\theta_1$$
![](Selection_005.png)

# Network Architecture
- main architecture
![](Selection_006.png)
- GLU architecture
![](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2016-september-later/GCN/2.png)
- main equation
![](Selection_007.png)
- final equation
![](Selection_008.png)

# Experiments
- linear interpolation method for missing values
- normalized by standard score method((x-mean)/std)
- adjacency matrix
![](Selection_009.png)
10,0.5

# result
![](Selection_010.png)
![](Selection_011.png)