import numpy as np
import math

# import gym.envs.gym_foo.job_distribution
from gym.envs.gym_rs.job_distribution import *
# import job_distribution

class Parameters:
    def __init__(self, randstate):

        # self.output_filename = 'data/tmp'


        self.episode_max_length = 10000 #0  # enforcing an artificial terminal # the max small time steps that 1 episode has # consistent with steps in gym _init_.py
        # self.num_epochs = 10000         # number of training epochs
        self.simu_len = 50 #200 #10 #self.episode_max_length #10             # length of the busy cycle that repeats itself
        self.num_ex = 2 #1                # number of sequences

        # self.output_freq = 10          # interval for output and store parameters

        # self.num_seq_per_batch = 10    # number of sequences to compute baseline
        # self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_clu = 4               # number of clusters !!!!!!

        self.num_res = 1               # number of resources in the system
        self.num_nw = 5 #10 #5                # maximum allowed number of work in the queue

        self.time_horizon = 100 #200 #30 # 20         # number of time steps in the graph
        self.max_job_len = 9 #3  #10 # 15          # maximum duration of new jobs
        self.res_slot = 15 #5 #10             # maximum number of available resource slots
        self.max_job_size = 9 #5 # 10         # maximum resource request of new work

        self.backlog_size = 100 #300 #500 #60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        # self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 1 #5       # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        self.max_transmiss_rate = 5

        self.trans_ratio = 0.5

        self.trans_rate = self.gener_transimi_rate(randstate)
        # print(self.trans_rate)
        # distribution for new job arrival
        # self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)
        self.dist = Dist(self.num_res,self.num_clu,self.max_job_size, self.max_job_len, randstate, self.trans_rate)


        

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_transmiss_rate * self.num_clu + 
             self.max_job_size * self.num_nw) * self.num_clu + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # # compact representation
        # self.network_compact_dim = (self.num_clu + 1) * \
        #     (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full
        self.drop_penalty = -1        # penalty for those jobs that are dropped (because backlog is full)

        # self.num_frames = 1           # number of frames to combine and process
        # self.lr_rate = 0.001          # learning rate
        # self.rms_rho = 0.9            # for rms prop
        # self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # # supervised learning mimic policy
        # self.batch_size = 10
        # self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_transmiss_rate * self.num_clu + 
             self.max_job_size * self.num_nw) * self.num_clu + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # # compact representation
        # self.network_compact_dim = (self.num_clu + 1) * \
        #     (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action


    def gener_transimi_rate(self, randstate):

        f1 = open("/home/veronica/.local/lib/python3.5/site-packages/gym/envs/gym_rs/clu_trans.txt", 'r')
        lines1=f1.readlines()
        nw_clu_t = []
        #print(len(lines1))
        len_len = len(lines1)
        # print(len_len)
        # print(len_len/3)

        numl = 0 
        tem_list = []
        for line1 in lines1:
            tem_l1 = line1.replace('[','').replace(']','')  #.replace(' ',',')
            row = tem_l1.split()
            if numl % self.num_clu == 0 and numl != 0:
                nw_clu_t.append(tem_list)
                tem_list = []
            tem_save = []
            for item in row:
                tem_save.append(int(item))

            # nw_len.append(tem_list)
            tem_list.append(tem_save)

            if numl == len(lines1)/self.num_clu - 1:
                nw_clu_t.append(tem_list)

            numl +=1 
            # print(len(nw_len))

            f1.close()

        tem_clu_t = np.array(nw_clu_t)

        # print(tem_clu_t[0])

        # print(len(tem_clu_t[0]))
        # print(tem_clu_t.size())


        return tem_clu_t[0]

        # tem_a = np.zeros((self.num_clu, self.num_clu), dtype = int)
        # for i in range(len(tem_a)):
        #     for j in range(i, len(tem_a[i])):
        #         if j != i:
        #             tem_r = randstate.randint(3, self.max_transmiss_rate+1)
        #             tem_a[i][j] = tem_r
        #             tem_a[j][i] = tem_r
        #         else:
        #             tem_a[i][j] = 0

        # return tem_a

