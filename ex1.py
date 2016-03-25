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

        while position>0 and eigenvalues[position-1]>currenteigenvalue:
            eigenvalues[position]=eigenvalues[position-1]
            eigenvectors[position] = eigenvectors[position-1]
            position = position-1

        eigenvalues[position]=currenteigenvalue
        eigenvectors[position]=currenteigenvector

    return eigenvalues, eigenvectors

'''
eigenvalues = [54,26,93,17,77,31,44,55,20]
insertionSort(eigenvalues)
print(eigenvalues)
'''

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
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i[0])


    print 'eig_pairs'
    print eig_pairs
    '''

    '''
    And about the negative eigenvalues, it is just a matter of eigh. 
    As eigenvalues shows the variance in a direction, 
    we care about absolute value but if we change a sign, 
    we also have to change the "direcction" (eigenvector). 
    You can make this multiplying negative eigenvalues and their corresponding eigenvectors with -1.0
    '''
    # http://stackoverflow.com/questions/22885100/principal-component-analysis-in-python-analytical-mistake
    s = np.where(eig_vals < 0)
    eig_vals[s] = eig_vals[s] * -1.0
    eig_vecs[:,s] = eig_vecs[:,s] * -1.0

    D = eig_vals
    V = eig_vecs
    print 'Eigenvalues D'
    print D
    print D.shape
    print 'Eigenvectors V'
    print V
    print V.shape

    [D, V] = insertionSort(D, V)
    print 'Sorted Eigenvalues D'
    print D
    print D.shape
    print 'Sorted Eigenvectors V'
    print V
    print V.shape    

    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    return [V, D]


def s_pca(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 9)
    pca.fit(X)
    '''
    print 'Scikit learn convariance matrix', pca.get_covariance(), '\n'
    '''
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
    result = s_pca(X)
    
    print result.explained_variance_ratio_
    first_pc = result.components_[0]
    second_pc = result.components_[1]
    third_pc = result.components_[2]
    fourth_pc = result.components_[3]
    fifth_pc = result.components_[4]
    sixth_pc = result.components_[5]
    seventh_pc = result.components_[6]
    eighth_pc = result.components_[7]
    ninth_pc = result.components_[8]
    print first_pc
    print second_pc
    print third_pc
    print fourth_pc
    print fifth_pc
    print sixth_pc
    print seventh_pc
    print eighth_pc
    print ninth_pc
    '''
    transformed_data = result.transform(X)
    for ii, jj in zip(transformed_data, X):
        plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color = 'r')
        plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], color = 'c')
        plt.scatter(jj[0],jj[1], color='b')
    plt.show()    
    '''



if __name__ == '__main__':
    main()

