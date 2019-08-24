
# coding: utf-8

# In[16]:


import numpy as np


# In[11]:


def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - np.dot(Vj, (Yij - np.dot(Ui, Vj))))


# In[12]:


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - np.dot(Ui, (Yij - np.dot(Ui, Vj))))


# In[13]:


def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    error = 0
    for row in Y:
        i = row[0]
        j = row[1]
        yij = row[2]
        error += (yij - np.dot(U[i-1], V[j-1]))**2        
    return (reg / 2) * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2) + .5 * (error / len(Y))


# In[14]:


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.uniform(low=-.5, high=.5, size=(M, K))
    V = np.random.uniform(low=-.5, high=.5, size=(N, K))
    t0 = get_err(U, V, Y)
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    for index in idx:
            i = Y[index][0]
            j = Y[index][1]
            y = Y[index][2]
            U[i-1] -= grad_U(U[i-1], y, V[j-1], reg, eta)
            V[j-1] -= grad_V(V[j-1], y, U[i-1], reg, eta)
    t1 = get_err(U, V, Y)
    delt = t0 - t1
    for i in range(max_epochs-1):
        old_e = get_err(U, V, Y)
        idx = np.arange(len(Y))
        np.random.shuffle(idx)
        for index in idx:
            i = Y[index][0]
            j = Y[index][1]
            y = Y[index][2]
            U[i-1] -= grad_U(U[i-1], y, V[j-1], reg, eta)
            V[j-1] -= grad_V(V[j-1], y, U[i-1], reg, eta) 
        new_e = get_err(U, V, Y)   
        if ((old_e - new_e) / delt) <= eps: 
            break       
    return (U, V, get_err(U, V, Y, 0))

