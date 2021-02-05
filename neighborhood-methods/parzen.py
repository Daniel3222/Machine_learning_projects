import numpy as np
banknote = np.genfromtxt('data_banknote_authentication.txt', delimiter=",")


######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

# --- distance calculator ----
def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)


class Q1:

    def feature_means(self, banknote):
        return banknote[:, :-1].mean(axis=0)

    def covariance_matrix(self, banknote):
        cov_matrix = np.cov(banknote[:, :-1].transpose(), bias=True)
        return cov_matrix

    def feature_means_class_1(self, banknote):
        class_1 =  banknote[np.where(banknote[:, 4] == 1)]
        class_1 = class_1[:, :-1]
        return class_1.mean(axis=0)

    def covariance_matrix_class_1(self, banknote):
        class_1 = banknote[np.where(banknote[:, 4] == 1)]
        class_1 = class_1[:, :-1]
        cov_matrix = np.cov(class_1.transpose(), bias=True)
        return cov_matrix


class HardParzen:
    def __init__(self, h):
        self.h = h

    # The train function for knn / Parzen windows is really only storing the dataset
    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    # The prediction function takes as input test_data and returns an array containing the predicted classes.
    def compute_predictions(self, test_data):
        # Initialization of the count matrix and the predicted classes array
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        # For each test datapoint
        for (i, ex) in enumerate(test_data):

            # i is the row index
            # ex is the i'th row

            # Find the distances to each training set point using dist_func
            distances = minkowski_mat(ex, self.train_inputs)

            # Go through the training set to find the neighbors of the current point (ex)

            neighbour_idx = []  # initialise la liste
            h = self.h  # initialise le radius

            neighbour_idx = np.array([j for j in range(len(distances)) if distances[
                j] <= h])  # for j, de (0 à 49), si la distance du training point j est plus petite que le radius
            # alors, mets dans la liste neighbors_idx, l'indice du training point qui satisfait
            # cette condition

            h *= 2  # le diamètre d'un cercle, c'est 2 fois le radius

            if len(neighbour_idx) == 0:  # aucun training point tombe dans le radius du test point
                classes_pred[i] = draw_rand_label(ex, np.unique(self.train_labels))
            # Calculate the number of neighbors belonging to each class and write them in counts[i, :]
            for k in neighbour_idx:  # ici k représente l'indice du/(des) training point(s) qui satisfait la condition du test point
                counts[i, self.train_labels[k]] += 1 # count iteratively for each test point, the number of votes for each class

            classes_pred[i] = np.argmax(counts[i, :]) # predict for each test point, the class that holds the most votes

        return classes_pred



class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    # The prediction function takes as input test_data and returns an array containing the predicted classes.
    def compute_predictions(self, test_data):
        # Initialization of the count matrix and the predicted classes array
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            sigma = self.sigma  # initialise le sigma
            d = (self.train_inputs.shape[1]) # find the number of dimensions (parameter d of the kernel K)
            k = 1/((2 * np.pi)**(d/2) * sigma**d) * np.e**(-0.5 * minkowski_mat(ex, self.train_inputs)/sigma**2) # find the weights between each point and

            for idx, weight_kernel in enumerate(k):
              counts[i, self.train_labels[idx]] += weight_kernel

            classes_pred[i] = np.argmax(counts[i, :])

        return classes_pred

def split_dataset(banknote):
    training_set = []
    for i in range(banknote.shape[0]):
        if i % 5 == 0:
            training_set.append(i)
        elif i % 5 == 1:
            training_set.append(i)
        elif i % 5 == 2:
            training_set.append(i)

    validation_set = []
    for i in range(banknote.shape[0]):
        if i % 5 == 3:
            validation_set.append(i)


    test_set = []
    for i in range(banknote.shape[0]):
        if i % 5 == 4:
            test_set.append(i)

    return banknote[training_set], banknote[validation_set], banknote[test_set]

class ErrorRate:

    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train.astype('int32')
        self.x_val = x_val
        self.y_val = y_val.astype('int32')

    def hard_parzen(self, h):
        parzenh = HardParzen(h=h)
        parzenh.train(self.x_train,  self.y_train)
        classes_pred_parzenh = parzenh.compute_predictions(self.x_val)
        sum_diff = np.subtract(classes_pred_parzenh, self.y_val)
        sum_diffs_2 = np.count_nonzero(sum_diff==0)
        return (abs(sum_diffs_2 - sum_diff.size)/self.y_val.shape[0])


    def soft_parzen(self, sigma):
        parzens = SoftRBFParzen(sigma=sigma)
        parzens.train(self.x_train,  self.y_train)
        classes_pred_parzens = parzens.compute_predictions(self.x_val)
        sum_diff = np.subtract(classes_pred_parzens, self.y_val)
        sum_diffs_2 = np.count_nonzero(sum_diff == 0)
        return (abs(sum_diffs_2 - sum_diff.size) / self.y_val.shape[0])


def get_test_errors(banknote):

    # parameters for h and sigma
    h_star = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_star = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    # Splitting of the dataset into train, validation and test
    data = split_dataset(banknote)
    train = data[0]
    x_train = train[:, :-1]
    y_train = train[:, -1]

    val = data[1]
    x_val = val[:, :-1]
    y_val = val[:, -1]

    test = data[2]
    x_test = test[:, :-1]
    y_test = test[:, -1]

    # We now instantiate an empty list that will contain all variants of h, and their associated error rate, as a tuple
    test_h_star = list()
    test_sigma_star = list()

    for h in h_star:
        error = ErrorRate(x_train, y_train, x_val, y_val)
        test_h_star.append((h, error.hard_parzen(h)))
    for sigma in sigma_star:
        error = ErrorRate(x_train, y_train, x_val, y_val)
        test_sigma_star.append((sigma, error.soft_parzen(sigma)))

    # find the best "h" for the minimum error
    arr_error_h = np.array(test_h_star)
    sort_array = np.argsort(arr_error_h[:, 1])
    min_error_h = arr_error_h[sort_array[0], :][0]

    # find the best sigma for the minimum error
    arr_error_sigma = np.array(test_sigma_star)
    sort_array = np.argsort(arr_error_sigma[:, 1]) # sort on the error column
    min_error_sigma = arr_error_sigma[sort_array[0], :][0] # get the first tuple (the parameter) for the lowest value of error

    # Instantiate an ErrorRate object as error
    error = ErrorRate(x_train, y_train, x_test, y_test)

    # use our best parameters to train and evaluate our model
    hard_parzen_error = error.hard_parzen(min_error_h)
    soft_parzen_error = error.soft_parzen(min_error_sigma)

    return [hard_parzen_error, soft_parzen_error]


def random_projections(X, A):
    return (np.dot(X,A))*(1/(2)**(1/2)) # X is a matrix, so is A, in this case A will be a randomly created matrix
                                        # we don't need to specify the nature of the matrices because we want the function
                                        #  to take as many different types of arrays, as long as they satisfy condition for
                                        # multiplying them
