import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X


def insertionSort(eigenvalues, eigenvectors):
    for index in range(1,len(eigenvalues)):

        currenteigenvalue = eigenvalues[index]
        currenteigenvector = eigenvectors[index]
        position = index

        while position>0 and eigenvalues[position-1]<currenteigenvalue:
            eigenvalues[position]=eigenvalues[position-1]
            eigenvectors[position] = eigenvectors[position-1]
            position = position-1

        eigenvalues[position]=currenteigenvalue
        eigenvectors[position]=currenteigenvector

    return eigenvalues, eigenvectors

def POV_arr(eigenvalues):
    arr = []
    sum = 0
    for i in range(0, len(eigenvalues)):
        sum += eigenvalues[i]

    cumulate = 0
    for i in range(0, len(eigenvalues)):
        cumulate += eigenvalues[i]
        arr.append(cumulate / sum)
    return arr

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    
    # Sum all the elements column by column
    sum_X  = [[ sum(x) for x in zip(*X) ]]
    sum_X = np.asarray(sum_X)
    # the following lines do the almost the exact same thing
    # sum_X_np = np.sum(X, axis = 0)
    '''
    print 'sum_X'
    print sum_X
    print sum_X.shape
    '''

    # Calculate the mean vector
    mean_vector = sum_X / 1000.0
    '''
    print 'mean_vector'
    print mean_vector
    print mean_vector.shape
    '''

    # Normalize the whole thing
    thousand_ones = np.ones((1000, 1))
    X_normalized = np.dot(thousand_ones, mean_vector)
    X_normalized = np.subtract(X, X_normalized)
    '''
    print 'X_normalized'
    print X_normalized
    print X_normalized.shape
    '''

    S = np.cov(np.transpose(X_normalized))
    '''
    print 'Convariance Matrix I calculated'
    print S
    print S.shape
    print '\n'
    '''

    [eig_vals, eig_vecs] = np.linalg.eig(S)

    # Make a list of (eigenvalue, eigenvector) tuples

    '''
    And about the negative eigenvalues, it is just a matter of eigh. 
    As eigenvalues shows the variance in a direction, 
    we care about absolute value but if we change a sign, 
    we also have to change the "direcction" (eigenvector). 
    You can make this multiplying negative eigenvalues and their corresponding eigenvectors with -1.0
    
    Refer to : http://stackoverflow.com/questions/22885100/principal-component-analysis-in-python-analytical-mistake
    '''
    s = np.where(eig_vals < 0)
    eig_vals[s] = eig_vals[s] * -1.0
    eig_vecs[:,s] = eig_vecs[:,s] * -1.0

    D = eig_vals
    V = eig_vecs
    '''
    print 'Eigenvalues D'
    print D.shape
    print D
    print 'Eigenvectors V'
    print V.shape
    print V
    '''

    [D, V] = insertionSort(D, V)
    '''
    print 'Sorted Eigenvalues D'
    print D.shape
    print D
    print 'Sorted Eigenvectors V'
    print V.shape
    print V
    '''    

    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.

    X = []
    for i in range(0, len(D)):
        X.append(i + 1)

    plt.plot(X, D, color= 'b')
    '''
    for i in range(0, len(D)):
        print i
        plt.plot(i+1, D[i])
        plt.scatter(i+1, D[i], color = 'b')
    '''
    plt.title('Task A: POV: Eigenvalues')
    plt.ylabel('Eigenvalues')
    plt.xlabel('k')
    plt.show()


    POV = POV_arr(D)
    print 'POV'
    print POV
    print '\n'
    plt.plot(X, POV, marker = 'o', color='b', linestyle='-')
    plt.title('Task A: Proportion of variance(POV)')
    plt.ylabel('Prop. of var')
    plt.xlabel('k')
    plt.show()


    return [V, D]


def scikit_pca(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 9)
    pca.fit(X)
    '''
    print 'Scikit learn covariance matrix', pca.get_covariance(), '\n'
    '''

    covariance_matrix = pca.get_covariance()
    [eig_vals, eig_vecs] = np.linalg.eig(covariance_matrix)

    X = []
    for i in range(0, len(eig_vals)):
        X.append(i + 1)

    plt.plot(X, eig_vals, color= 'r')
    plt.title('Task A: Eigenvalues using sklearn')
    plt.ylabel('Eigenvalues')
    plt.xlabel('k')
    plt.show()


    POV = POV_arr(eig_vals)
    print 'POV using sklearn'
    print POV
    print '\n'
    plt.plot(X, POV, marker = 'o', color='r', linestyle='-')
    plt.title('Task A: Proportyion of variance (POV) using sklearn')
    plt.ylabel('Prop. of var')
    plt.xlabel('k')
    plt.show()



    return pca

def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!

    ####################################################################
    pca(X)
    scikit_pca(X)


if __name__ == '__main__':
    main()

