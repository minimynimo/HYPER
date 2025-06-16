#!python3
"""
2019/8/29
This is an numpy implementation of echo state network (reservoir computing).
"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
#from progressbar import ProgressBar

class ESN():
    """
    The Echo State Network.

    args:
     - input_size           : size of the observed model space.
     - output_size          : size of the whole model space. Identical to input_size, if not designated.
     - reservoir_size       : size of the reservoir. Default 1000.
     - adjacency_density    : The density of the adjacency matrix for reservoir evolution. Default 0.5
     - spectral_radius      : Maximum magnitude of eigenvalues of adjacency matrix. Default 1.0
     - input_scale          : scale of the input-reservoir mapping matrix. Default 1.0

    Parameters:
     - A
     - W_in
     - W_out
     - reservoir_size
     - input_size
     - output_size
    """

    def __init__(self, input_size, output_size=None, model_size = 1, reservoir_size=1000, adjacency_density=0.5, spectral_radius=1.0, input_scale=1.0):
        #print('model init')
        self.A = None
        self.W_in = None
        self.W_out = None

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        if output_size == None:
            self.output_size = input_size
        else:
            self.output_size = output_size

        self.A = sparse.rand(reservoir_size, reservoir_size, density=adjacency_density)
        self.A = np.array(self.A.todense())
        max_eigval = np.abs(LA.eigvals(self.A)).max()
        if max_eigval > 0:
            self.A = self.A * spectral_radius / max_eigval

        #self.A = self.A * spectral_radius / (np.abs(LA.eigvals(self.A)).max())

        self.W_in = np.zeros((reservoir_size, input_size+model_size))
        q = int(reservoir_size / (input_size+model_size))
        for i in range((input_size+model_size)):
            np.random.seed(seed=i)
            self.W_in[i*q:(i+1)*q, i] = input_scale * (np.random.rand(q) * 2 - 1)

    def forward(self, input_vector, knwbsd_sim, previous_state=None):
        """
        Compute the reservoir states of next timestep.

        args:
            - input_vector      : model state at current timestep.
            - previous_state    : reservoir state at current timestep.
        """
        if previous_state is None:
            previous_state = np.zeros((self.reservoir_size, 1))

        # reshape vectors to be column vectors.
        input_vector = input_vector.copy().reshape(-1, 1)       # (4,1)
        knwbsd_sim = np.array([knwbsd_sim]).reshape(-1, 1) if np.isscalar(knwbsd_sim) else knwbsd_sim.reshape(-1, 1)
        previous_state = previous_state.copy().reshape(-1, 1)   # (R,1)

        mergedlist = np.concatenate([knwbsd_sim, input_vector], axis=0) # 5,1
        # compute the reservoir state at next timestep.
        r = np.tanh(np.dot(self.A, previous_state) + np.dot(self.W_in, mergedlist))
        #           (R*R) @ (R,1)             + (R*48) @ (48,1)

        # returns the reservoir state vector as a row vector.
        return r.reshape(-1).copy()


    def train(self, observed_data, knwbsd_sim, target_data=None, washout=0, ridge_param=0.01, spin = False, outlier = False):
        """
        Train the network with input data.
        args:
        - observed_data     : input matrix (npoints * length). Each columns stands for the state at each time.
        - target_data       : output matrix. Designate if input data (observed data) is imperfect.
        - washout           : timesteps to drop.
        - ridge_param       : parameter for ridge regression
        """
        # observed_data <- input_cal    # (4,n)
        # target_data <- target_cal     # (n, )

        # spin settings
        if spin == False:
            spin = 1
        observed_data = np.tile(observed_data, (1, spin))
        knwbsd_sim = np.tile(knwbsd_sim,spin)
        target_data = np.tile(target_data, spin)

        if type(target_data) == type(None):
            target_data = observed_data[:, 1:]
            observed_data = observed_data[:, :-1]   

        # arguments assertion
        if observed_data.shape[0] != self.input_size:
            raise ValueError('Observed data size is set to {}, not {}'.format(self.input_size, observed_data.shape[0]))
        #if target_data.shape[0] != self.output_size:
        #    raise ValueError('Target data size is set to {}, not {}'.format(self.output_size, target_data.shape[0]))
        if observed_data.shape[1] != len(target_data):
            raise ValueError('Observed and target data duration should be the same, got {} and {}'.format(observed_data.shape[1], len(target_data)))

        # Outlier
        if outlier != False:
            outliers = np.average(target_data)+outlier * np.std(target_data)
            for i in range(len(target_data)):
                if target_data[i] > outliers:
                    target_data[i] = outliers

        target_data = target_data[1:]

        # calculate reservoir state
        train_length = observed_data.shape[1]-1 # n <- true n -1 (for knwbsd t-dt)
        #p = ProgressBar(0, train_length)

        reservoir = np.zeros((self.reservoir_size, train_length))   # (R,n)
        reservoir[:, 0] = self.forward(observed_data[:, 1],knwbsd_sim[1])         # (R, )

        for i in range(1, train_length):
            #p.update(i) # progressbar
            reservoir[:, i] = self.forward(observed_data[:, i+1], knwbsd_sim[i+1], reservoir[:, i-1])
        #p.update(train_length)
        #p.finish()
        
        #reservoir_last = reservoir[:, -1].copy()    # (R, )
        reservoir_last = reservoir.copy()    # (R, n)


        ##For non-hybrid RC
        # u~R(t) = U R^t
        # r*(t) = R R^T + beta*I
        # u~R(t)= Wout r*(t)
        # so
        # Wout = U R^T (R R^T + beta*I)^-1


        ##For hybrid RC
        # z(t) = (K[u(t-dt)] /up/bottom/ r*(t)) = np.vstack(K[u(t-dt),r*(t)])=np.vstack(K[u(t-dt)],R R^T + beta*I)
        # u~H(t) = Wout (K[u(t-dt)] /up/bottom/ r*(t))
        # so 
        # Wout = U Z^t (Z Z^T + beta*I))^-1

        #########
        # nonlinear transformation
        reservoir_star = self.transform_reservoir_state(reservoir)  # (R,n)

        # washout and target data setting
        reservoir = reservoir[:, washout:] #(R,n)
        target = target_data[washout:]      # (n, )
        target = target.reshape(-1, 1).T    # (1,n)

        
        #r_star = self.transform_reservoir_state(reservoir) #(R,n)
        knwbsd_sim = knwbsd_sim[:-1].reshape(1,-1) # (n,)

        # ridge regression
        Z = np.concatenate((knwbsd_sim, reservoir_star), axis=0) #(R+1,n)

        U = Z @ Z.T + ridge_param * np.eye(len(Z)) # (R+1,R+1)
        # (R+1,n) @ (n,R+1) + (R+1,R+1)
        Uinv = LA.inv(U)            # (R+1,R+1)

        self.W_out = (np.dot( np.dot(target, Z.T) ,Uinv))  # ((1,n)@ (n,R+1) )   @ (R+1,R+1) = (1,R+1) #1,501
        return self.W_out, reservoir_last

    def predict(self, current_reservoir, test_observed, knwbsd_sim, ptb_func=None, ptb_scale=0.0, nexttime=False, extended_interval=0, fix = False):
        """
        Generate the predict time series.
        args:
        - current_reservoir     : last reservoir state in training phase
        - test_observed         : whole test data # 4inputs  by obs duration
        - nexttime              : whether to give true state every step
        - extended_interval     : interval to give the true state
        """
        # current_reservoir <- reservoir    #(R,n)
        # test_observed <- input_eva        #(4,m)

        # pertubation
        if ptb_func == 'normal':
            ptb = lambda: np.random.normal(loc=0, scale=ptb_scale, size=self.input_size)
        elif type(ptb_func) != type(None):
            ptb = ptb_func
        else:
            ptb = lambda: np.zeros((self.input_size))

        test_length = test_observed.shape[1]  # m
    
        predict = np.zeros((self.output_size, test_length)) # (1,m)

        #p = ProgressBar(0, test_length)
        current_reservoir = current_reservoir[:,-1] ## last of the reservoir (R,1)

        # initial predict
        current_reservoir = self.forward(test_observed[:,0] + ptb(), knwbsd_sim[0], current_reservoir)    # (R, )
        #reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False)          # (R, )
        reservoir_star = self.transform_reservoir_state(current_reservoir, inplace=False)          # (R, )

        predict[:,0] = (self.W_out @ np.concatenate((knwbsd_sim[0].reshape(-1),reservoir_star)).reshape(-1,1)).reshape(-1)            # (1, )
        # (1,R+1) @ (R+1,1) .reshape(-1,1)

        # predict iteration
        for i in range(1, test_length):  # 1,1090
            #p.update(i) # progressbar
            if nexttime:
                current_reservoir = self.forward(test_observed[:, i] + ptb(), knwbsd_sim[i], current_reservoir) # (R,1)
            elif extended_interval > 0 and i % extended_interval == 0:
                current_reservoir = self.forward(test_observed[:, i-100],knwbsd_sim[i-100])
                for j in range(99, 0, -1):
                    current_reservoir = self.forward(test_observed[:, i-j], knwbsd_sim[i-j], current_reservoir)
                current_reservoir = self.forward(test_observed[:, i] + ptb(), knwbsd_sim[i], current_reservoir)
            else:
                current_reservoir = self.forward(predict[:,i-1], knwbsd_sim[i-1], current_reservoir) # should be (R,1)
            #reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False) #should be (R,1)
            reservoir_star = self.transform_reservoir_state(current_reservoir, inplace=False)          # (R, )
            predict[:,i] = (self.W_out @ np.concatenate((knwbsd_sim[i].reshape(-1),reservoir_star)).reshape(-1,1)).reshape(-1)
    
            if predict[:,i] < 0:
                predict[:,i] = 0
                #   { (1,R) @ (R,1) }.reshape(-1) should be
        #p.update(test_length)
        #p.finish()


        # predict (1,m)
        return predict

    def nonlinear_transf(self, matrix, inplace=False):
        """
        Apply nonlinear row transformation to input matrix.
        """
        if not inplace:
            matrix = matrix.copy()
        row_pre = matrix[0].copy()
        for i in range(2, matrix.shape[0], 2):
            row_tmp = matrix[i].copy()
            matrix[i] = (matrix[i-1] * row_pre).copy()
            row_pre = row_tmp.copy()
        return matrix
    
    def transform_reservoir_state(self,r, inplace = False):
        # Initialize the vector r* with the same shape as r 
        # r as vector
        r_star = np.zeros_like(r)

        # Iterate over the elements of r and apply the transformation
        for j in range(len(r)):
            if (j + 1) % 2 == 1:  # odd index (1-based, so (j+1) is used)
                r_star[j] = r[j]
            else:  # even index
                r_star[j] = r[j] ** 2
        return r_star
