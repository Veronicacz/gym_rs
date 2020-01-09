import gym
from gym import spaces, utils #errors
from gym.utils import seeding
import numpy as np
# import gym.envs.gym_foo.job_distribution 
# import gym.envs.gym_foo.parameters as parameters
from gym.envs.gym_rs.job_distribution import *
from gym.envs.gym_rs.parameters import *
# import job_distribution
# import parameters
import math
import matplotlib.pyplot as plt


from collections import deque


class RsEnv(gym.Env):
	"""docstring for FooEnv"""

	metadata = {'render.modes': ['human']}

	def __init__(self, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42,render=False, repre='image',end='all_done'):  # 'no_new_job'
		# self.pa = parameters.Parameters()
		self.ransta = np.random.RandomState(123)
		self.pa = Parameters()
		self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
		# self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
		self.action_space = spaces.Discrete(self.pa.num_nw+1) #, shape=(1,), dtype = np.int16)
		# self.action_space.low = np.array([0])
		# self.action_space.high = np.array([self.action_space.n])
		self.nw_dist = self.pa.dist.bi_model_dist
		self.curr_time = 0
		self.end = end


		# print("network_input_height")
		# print(self.pa.network_input_height)  # 20
		# print("network_input_width")
		# print(self.pa.network_input_width)  #124

		#set up random seed
		# if self.pa.unseen:
		# 	np.random.seed(314159)
		# else:
		# 	np.random.seed(seed)

		if nw_len_seqs is None or nw_size_seqs is None:
			# generate new work
			self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = \
            	self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
			# self.workload = np.zeros(self.pa.num_res)
			# for i in range(self.pa.num_res):
				# print(self.nw_size_seqs[:, i])
				# print(self.nw_len_seqs)
				# exit()
				# self.workload[i] = \
    #                 np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
    #                 float(self.pa.res_slot) / \
    #                 float(len(self.nw_len_seqs))
				# print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
			self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu]) 
                                           #[self.pa.num_ex, self.pa.simu_len])
			self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
			self.nw_sub_seqs = np.reshape(self.nw_sub_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
			self.nw_trans_seqs = np.reshape(self.nw_trans_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
		else:
			self.nw_len_seqs = nw_len_seqs
			self.nw_size_seqs = nw_size_seqs
			self.nw_sub_seqs = nw_sub_seqs
			self.nw_trans_seqs = nw_trans_seqs
		self.seq_no = 0  # which example sequence
		self.seq_idx = -1 # 0  # index in that sequence
		# initialize system
		self.machine = Machine(self.pa, self.ransta)
		self.job_slot = JobSlot(self.pa)
		self.job_backlog = JobBacklog(self.pa)
		self.job_record = JobRecord()
		self.extra_info = ExtraInfo(self.pa)

		self.disslog = Dismisslog(self.pa)
		self.usedcolormap = []
		self.usedcolormap.append(0)
		self.state = self.observe()
		# self.reward = self.get_reward()


	def generate_sequence_work(self, joblist_len):
		nw_len_seq = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
		nw_size_seq = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
		nw_sub_seqs = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
		nw_trans_seqs = np.zeros((joblist_len, self.pa.num_clu), dtype=int)

		for i in range(joblist_len):

			if np.random.rand() < self.pa.new_job_rate:  # a new job comes
				nw_len_seq[i, :], nw_size_seq[i, :], nw_sub_seqs[i,:], nw_trans_seqs[i, :] = self.nw_dist()
		return nw_len_seq, nw_size_seq, nw_sub_seqs, nw_trans_seqs

	def get_new_job_from_seq(self, seq_no, seq_idx):
		new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx, :],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time,
                      job_order=self.nw_sub_seqs[seq_no, seq_idx, :],
                      job_trans=self.nw_sub_seqs[seq_no, seq_idx, :])
		return new_job

	def observe(self):
		backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
		image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

		ir_pt = 0

		for i in range(self.pa.num_clu):
			image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
			ir_pt += self.pa.res_slot

			for k in range(self.pa.num_clu):
				image_repr[:,ir_pt:ir_pt + self.pa.max_transmiss_rate] = self.machine.trans_canvas[i, k, :, :]
				ir_pt += self.pa.max_transmiss_rate
			
			for j in range(self.pa.num_nw):
				if self.job_slot.slot[j] is not None:  # fill in a block of work
					if self.job_slot.slot[j].res_vec[i] != 0:
						image_repr[: self.job_slot.slot[j].len[i], ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1
				ir_pt += self.pa.max_job_size

		###########modified by zc###############
		if self.job_backlog.curr_size == 0:
			image_repr[0,ir_pt: ir_pt + backlog_width] = 1

		else:
			image_repr[0: self.job_backlog.curr_size / backlog_width,ir_pt: ir_pt + backlog_width] = 1
		########################################

		if self.job_backlog.curr_size % backlog_width > 0:
			image_repr[self.job_backlog.curr_size / backlog_width,
                       ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
		ir_pt += backlog_width

		# image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
  #                                             float(self.extra_info.max_tracking_time_since_last_job)

  		image_repr[:, ir_pt: ir_pt + 1] = self.disslog.size

		ir_pt += 1

		assert ir_pt == image_repr.shape[1]

		return image_repr

	
	# def plot_state(self):
	# 	plt.figure("screen", figsize=(20, 5))

	# 	skip_row = 0

	# 	for i in range(self.pa.num_res):

	# 		plt.subplot(self.pa.num_res,
 #                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
			
	# 		plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

	# 		for j in range(self.pa.num_nw):

	# 			job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))

	# 			if self.job_slot.slot[j] is not None:  # fill in a block of work

	# 				job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

	# 			plt.subplot(self.pa.num_res,
 #                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

	# 			plt.imshow(job_slot, interpolation='nearest', vmax=1)

	# 			if j == self.pa.num_nw - 1:
	# 				skip_row += 1

	# 	skip_row -= 1
	# 	backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
	# 	backlog = np.zeros((self.pa.time_horizon, backlog_width))

	# 	# backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
	# 	backlog[: int(self.job_backlog.curr_size / backlog_width), : backlog_width] = 1

	# 	# backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1
	# 	tem_ra = (self.job_backlog.curr_size % backlog_width)
	# 	backlog[int(self.job_backlog.curr_size / backlog_width), : tem_ra] = 1

	# 	plt.subplot(self.pa.num_res,
 #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                    self.pa.num_nw + 1 + 1)

	# 	plt.imshow(backlog, interpolation='nearest', vmax=1)

	# 	plt.subplot(self.pa.num_res,
 #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

	# 	extra_info = np.ones((self.pa.time_horizon, 1)) * \
 #                     self.extra_info.time_since_last_new_job / \
 #                     float(self.extra_info.max_tracking_time_since_last_job)

	# 	plt.imshow(extra_info, interpolation='nearest', vmax=1)

	# 	plt.show()     # manual
 #        # plt.pause(0.01)  # automatic

	def get_reward(self):

		reward = 0
		for j in self.machine.running_job:
			length = 0
			for clu in range(self.pa.num_clu):
				if job.res_vec[clu] != 0:
					length += job.len[clu] 
					length += job.trans[clu]
			reward += self.pa.delay_penalty / float(length)

		for j in self.job_slot.slot:
			if j is not None:
				# reward += self.pa.hold_penalty / float(j.len)
				length = 0
				for clu in range(self.pa.num_clu):
					if job.res_vec[clu] != 0:
						length += job.len[clu] 
						length += job.trans[clu]
				reward += self.pa.delay_penalty / float(length)

		for j in self.job_backlog.backlog:
			if j is not None:
				# reward += self.pa.dismiss_penalty / float(j.len)
				length = 0
				for clu in range(self.pa.num_clu):
					if job.res_vec[clu] != 0:
						length += job.len[clu] 
						length += job.trans[clu]
				reward += self.pa.delay_penalty / float(length)

		return reward

	def step(self, a, repeat=False):
		status = None

		done = False
		reward = 0
		info = None

		if a == self.pa.num_nw:  # explicit void action
			status = 'MoveOn'
		elif self.job_slot.slot[a] is None:  # implicit void action
			status = 'MoveOn'
		else:
			allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
			if not allocated:  # implicit void action
				status = 'MoveOn'
			else:
				status = 'Allocate'

		if status == 'MoveOn':
			self.curr_time += 1
			self.machine.time_proceed(self.curr_time)
			self.extra_info.time_proceed()

            # add new jobs
			self.seq_idx += 1

			if self.end == "no_new_job":  # end of new job sequence
				if self.seq_idx >= self.pa.simu_len:
					done = True
			elif self.end == "all_done":  # everything has to be finished
				if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
					done = True
				elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
					done = True

				if not done:

					for i in range(self.pa.num_nw):
						if self.job_slot.slot[i] is None and self.job_backlog.curr_size > 0:
							tem_job = self.job_backlog.backlog.popleft()
							self.job_slot.slot[i] = tem_job

					if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
						new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

						if new_job.len > 0:  # a new job comes
							
							to_backlog = True
							for i in range(self.pa.num_nw):
								if self.job_slot.slot[i] is None:  # put in new visible job slots
									self.job_slot.slot[i] = new_job
									self.job_record.record[new_job.id] = new_job
									to_backlog = False
									break
							if to_backlog:
								if self.job_backlog.curr_size < self.pa.backlog_size:
									self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
									self.job_backlog.curr_size += 1
									self.job_record.record[new_job.id] = new_job
								else:  # abort, backlog full
									print("Backlog is full.")
									self.disslog.log[self.disslog.size] = new_job
									self.disslog.size += 1
                                	# exit(1)
							self.extra_info.new_job_comes()
			reward = self.get_reward()

		elif status == 'Allocate':
			self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
			self.job_slot.slot[a] = None

            # dequeue backlog
			if self.job_backlog.curr_size > 0:
				# self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
				# self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
				# self.job_backlog.backlog[-1] = None
				# self.job_backlog.curr_size -= 1
				self.job_slot.slot[a] = self.job_backlog.backlog.popleft()
				self.job_backlog.backlog.append(None)
				self.job_backlog.curr_size -= 1



		ob = self.observe()

		info = self.job_record

		# if done:
		# 	self.seq_idx = 0

		# 	if not repeat:
		# 		self.seq_no = (self.seq_no + 1) % self.pa.num_ex

		# 	self.reset()

		#if self.render:
		#	self.plot_state()

		return ob, reward, done, info



	def reset(self):

		self.pa = Parameters()
		self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
		self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
		# self.action_space = spaces.Box(0, self.pa.num_nw-1, shape=(1,), dtype = np.int16)
		self.nw_dist = self.pa.dist.bi_model_dist

		self.seq_idx = 0
		self.curr_time = 0

        # initialize system
		self.machine = Machine(self.pa)
		self.job_slot = JobSlot(self.pa)
		self.job_backlog = JobBacklog(self.pa)
		self.job_record = JobRecord()
		self.extra_info = ExtraInfo(self.pa)


		self.seq_no = 0  # which example sequence
		self.seq_idx = 0  # index in that sequence

		self.machine = Machine(self.pa)
		self.job_slot = JobSlot(self.pa)
		self.job_backlog = JobBacklog(self.pa)
		self.job_record = JobRecord()
		self.state = self.observe()
		self.reward = self.get_reward()


		return self.state

	# def render(self, mode='human', close=False):
	# 	if not close:
 #            raise NotImplementedError(
 #                "This environment does not support rendering")





class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time, job_order, job_trans):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.order = job_order
        self.trans = job_trans


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        # self.backlog = [None] * pa.backlog_size
        self.backlog = deque()
        for i in range(pa.backlog_size):
        	self.backlog.append(None)
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}

class Dismisslog:
	def __init__(self, pa):
		self.log = [None] * pa.episode_max_length
		self.size = 0

class Transqueue_clus:
	def __init__(self, pa):
		self.queue = [None] * pa.num_clu
		# self.queue = deque()
		for i in range(pa.num_clu):
			self.queue[i] = deque()
			for j in range(pa.episode_max_length):
				self.queue[i].append(None)

		self.size = [0] * pa.num_clu

		# for i in range(pa.num_clu):
		# 	self.queue[i] = [None] * pa.episode_max_length

class Machine:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.num_clu = pa.num_clu
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_clu)) * self.res_slot

        self.running_job = []

        # colormap for graphical representation
        # self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        self.colormap = np.arange(5, 201, 2)
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((self.num_clu, self.time_horizon, self.res_slot))

        # self.trans_queue = [None] * self.num_clu

        # for i in range(pa.num_clu):
        # 	self.trans_queue[i] = [None] * self.num_clu

        self.trans_canvas = np.zeros((self.num_clu, self.num_clu,self.time_horizon, pa.max_transmiss_rate))

        # for i in range(pa.num_clu):
        # 	for j in range(pa.num_clu):
        # 		for k in range(pa.time_horizon):
        # 			tem_num = pa.trans_rate[i][j]
        # 			self.trans_canvas[i][j][k][0:tem_num] = 1

    def allocate_job(self, job, curr_time, pa):

        allocated = False

        new_avbl_res = self.avbl_slot

        leng = 0
        for i in range(self.num_clu):
        	if job.res_vec[i] != 0: leng += job.len[i]

        trans = 0
        for i in range(self.num_clu):
        	trans += job.trans[i]

        total_t = leng + trans

        for i in range(1, self.num_clu+1):

        	tem_f = np.where(job.order == i)
        	if j.res_vec[tem_f] != 0: break


        for t in range(0, self.time_horizon -total_t):
        	new_avbl_res = self.avbl_slot[t:t+job.len[tem_f], :] - job.res_vec[tem_f]
        	tem_t = t+job.len[tem_f]+job.trans[tem_f]
        	for j in range(1, self.num_clu-tem_f+1):
        		tem_in = np.where(job.order == tem_f+j)
        		if job.res_vec[tem_in] != 0:
        			new_avbl_res = self.avbl_slot[tem_t:tem_t+job.len[tem_in], :] - job.res_vec[tem_f]
        			if job.trans[tem_in] != 0:
        				tem_t = tem_t+job.len[tem_in]+job.trans[tem_in]

        	if np.all(new_avbl_res[:] >= 0):

        		allocated = True

        		self.avbl_slot[t: t + total_t, :] = new_avbl_res
        		job.start_time = curr_time + t
        		job.finish_time = job.start_time + total_t

        		self.running_job.append(job)

        		# update graphical representation

        		used_color = np.unique(self.canvas[:])
        		# WARNING: there should be enough colors in the color map
        		for color in self.colormap:
        			if color not in used_color:
        				new_color = color
        				break

        		assert job.start_time != -1
        		assert job.finish_time != -1
        		assert job.finish_time > job.start_time
        		canvas_start_time = job.start_time - curr_time
        		canvas_end_time = job.finish_time - curr_time

        		tem_time = job.start_time - curr_time
        		tem_prev = None
        		for i in range(tem_f, self.num_clu):
        			tem_index = np.where(job.order == i)
        			if tem_prev != None and job.res_vec[tem_index] != 0 and job.trans[tem_prev][tem_index] != 0:
        				trans_end_time = job.trans[tem_prev][tem_index]
        				for t in range(tem_time, tem_time+trans_end_time):
        					avbl_slot = np.where(self.trans_canvas[tem_prev,tem_index, t, :] == 0)[0]
        					self.trans_canvas[tem_prev,tem_index, t, avbl_slot[: pa.trans_rate[tem_prev][tem_index]]] = new_color
        				tem_time += trans_end_time

        			if job.res_vec[tem_index] != 0:
        				sub_end_time = job.len[tem_index]
        				for t in range(tem_time, tem_time+sub_end_time):
        					avbl_slot = np.where(self.canvas[tem_index, t, :] == 0)[0]
        					self.canvas[tem_index, t, avbl_slot[: job.res_vec[tem_index]]] = new_color
        				tem_time += sub_end_time
        				tem_prev = tem_index

        		break

        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

        self.trans_canvas[:,:,:-1,:] = self.trans_canvas[:,:, 1:,:]
        self.trans_canvas[:,:,-1,:] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1

