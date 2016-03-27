import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from time import time

def load_data(digits = [0], num = 200):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 200 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        totalsize += min([len(os.walk('train%d' % digit).next()[2]), num])
    print 'We will load %d images' % totalsize
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print '\nReading images of digit %d' % digit,
        for i in xrange(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = misc.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print '\n'
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()

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

def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 500 images
    # each row of matrix X represents an image
    X = load_data(digits, 500)
    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)

    ####################################################################
    # plot the eigen images, eigenvalue v.s. the order of eigenvalue, POV
    # v.s. the order of eigenvalue
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``digit_pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.9 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``digit_pov.jpg'',
    #   ``description.txt'' and ``ex2.py''.
    # YOUR CODE HERE!

    ####################################################################
    """ 1. do the PCA on matrix X """
    n_components = 150
    print "Extracting the top", n_components, "eigenfaces from ",  X.shape[0], " faces"
    t0 = time()
    pca = PCA(n_components=n_components).fit(X)
    print "done in %0.3fs" % (time() - t0)

    eigenfaces = pca.components_.reshape(n_components, 28, 28)
    print "number of eigenfaces: ", len(eigenfaces)
    n_row = 1
    n_col = 9
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenfaces[i].reshape(28,28), cmap=plt.cm.gray)
        title_text = 'Eigenvalue ' + str(i + 1)
        plt.title(title_text, size=12)
        plt.xticks(())
        plt.yticks(())

    plt.show()

    print pca.explained_variance_ratio_
    print pca.explained_variance_ratio_.cumsum()

    
    covariance_matrix = pca.get_covariance()
    print covariance_matrix.shape
    [eig_vals, eig_vecs] = np.linalg.eig(covariance_matrix)
    print eig_vals.shape
    #print eig_vals
    print eig_vecs.shape

    real_eig_vals = []
    imag_eig_vals = []
    for i in range(len(eig_vals)):
        real_eig_vals.append(eig_vals[i].real)
        imag_eig_vals.append(eig_vals[i].imag)
    real_eig_vals.sort(reverse = True)

    #print 'real_eig_val'
    #print real_eig_vals

    #POV = POV_arr(eig_vals)
    POV = POV_arr(real_eig_vals)
    print 'Pov'
    print POV
    X = []
    for i in range(0, len(POV)):
        X.append(i+1)
    #plt.plot(X, POV, marker='o',color='b',linestyle='-')
    plt.plot(X, POV, color='b', linestyle='-')
    plt.title('Task B: Proportion of variance(POV)')
    plt.ylabel('Prop. of var')
    plt.xlabel('k')
    plt.show()
    for i in range(0, len(POV)):
        if POV[i] > 0.9:
            dimen_idx = i + 1
            break
    print 'No. of dimensions need to be reserved: ', dimen_idx
    





if __name__ == '__main__':
    main()
