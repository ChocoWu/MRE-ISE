import torch
from torch import nn
from torch.nn import functional as F

from cores.lamo.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):


    def __init__(self, text_input_size, visual_input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0):
        """
        Initialize InferenceNetwork.

        Args
            text_input_size : int, dimension of text input
            visual_input_size : int, dimension of visual input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(DecoderNetwork, self).__init__()
        assert isinstance(text_input_size, int), "text input_size must by type int."
        assert isinstance(visual_input_size, int), "visual input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.text_input_size = text_input_size
        self.visual_input_size = visual_input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_text_word_matrix = None
        self.topic_visual_word_matrix = None


        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                text_input_size, visual_input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                text_input_size, visual_input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, text_input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(text_input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

        self.alpha = torch.Tensor(n_components, visual_input_size)
        if torch.cuda.is_available():
            self.alpha = self.alpha.cuda()
        self.alpha = nn.Parameter(self.alpha)
        nn.init.xavier_uniform_(self.alpha)

        self.alpha_batchnorm = nn.BatchNorm1d(visual_input_size, affine=False)

        # dropout on theta
        self.drop_alpha = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            text_word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            # word_dist: batch_size x input_size
            self.topic_text_word_matrix = self.beta

            # in: batch_size x input_size x n_components
            visual_word_dist = F.softmax(
                self.alpha_batchnorm(torch.matmul(theta, self.alpha)), dim=1)
            # visual_word_dist: batch_size x input_size
            self.topic_visual_word_matrix = self.alpha

        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_text_word_matrix = beta
            text_word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size

            # simplex constrain on Beta
            alpha = F.softmax(self.alpha_batchnorm(self.alpha), dim=1)
            self.topic_visual_word_matrix = alpha
            visual_word_dist = torch.matmul(theta, alpha)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")

        # classify labels

        estimated_labels = None

        if labels is not None:
            estimated_labels = self.label_classification(theta)

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, text_word_dist, visual_word_dist, estimated_labels

    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
