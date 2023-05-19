# spark-recommendation-system

# Matrix Factorization and Collaborative filtering

Matrix factorization using collaborative filtering is widely used in recommender systems to provide personalized and relevant recommendations to users based on their previous interactions and behaviour. Collaborative filtering assumes that if two users have similar preferences on certain items, they are likely to have similar preferences on other items as well.

## Matrix factorization:
It discovers latent features of items and users. It represents user-item interaction in the form of a  matrix, where the rows represent various users and the columns represents ratings, and it decomposes this matrix into two lower rank matrices - the *user* matrix and the *item* matrix.

Thus, the decomposition can help identify the latent factors/features that represent the user preferences by capturing the patterns and correlations in the data that can be used to make recommendations.

The factorized matrices are multiplied together to reconstruct the original matrix and predict missing ratings, thereby making new recommendations to users. 

Mathematically, the rating, in terms of the latent factors can be represented by

$$\hat r_{u,i} = q_{i}^{T}p_{u}$$

where $\hat r_{u,i}$ is the predicted ratings for user $u$ and item $i$, and $q_{i}^{T}$ and $p_{u}$ are latent factors for item and user, respectively.

The challenge to the matrix factorization problem is to find $q_{i}^{T}$ and $p_{u}$. This is achieved by methods such as matrix decomposition. A learning approach is therefore developed to converge the decomposition results close to the observed ratings as much as possible. Furthermore, to avoid overfitting issue, the learning process is regularized. For example, a basic form of such matrix factorization algorithm is represented as below.

$$\min\sum(r_{u,i} - q_{i}^{T}p_{u})^2 + \lambda(||q_{i}||^2 + ||p_{u}||^2)$$

where $\lambda$ is a the regularization parameter. 

## Alternating Least Square (ALS)

Owing to the term of $q_{i}^{T}p_{u}$ the loss function is non-convex. Gradient descent method can be applied but this will incur expensive computations. An Alternating Least Square (ALS) algorithm was therefore developed to overcome this issue. 

The basic idea of ALS is to learn one of $q$ and $p$ at a time for optimization while keeping the other as constant. This makes the objective at each iteration convex and solvable. The alternating between $q$ and $p$ stops whenoptimal convergence is obtained. This iterative computation can be parallelised and/or distributed, which makes the algorithm desirable for use cases where the dataset is large and thus the user-item rating matrix is super sparse (as is typical in recommendation scenarios).

## PySpark ALS module for recommender systems

- PySpark is a distributed computing framework that provides scalability to handle large user-item interactions efficiently.
- PySpark ALS also takes advantage of parallel processing of data, therefore resulting in faster training and inference times compared to traditional single-node implementations. 
- PySpark ALS also supports various configurations and parameters, allowing easy fine-tuning of the model. 
- PySpark ALS provides functionalities and strategies to handle cold-start recommendations.

## Implementation

This study presents a self-contained notebook that implements a recommender system on MovieLens 100k dataset using Spark ALS algorithm. 

### Packages

- pandas
- numpy
- seaborn
- [pyspark](https://spark.apache.org/docs/latest/api/python/)
- [recommenders](https://github.com/microsoft/recommenders)


