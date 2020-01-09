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
		self.pa = Parameters(self.ransta)
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
		# 	# generate new work
			# self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = \
   #          	self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
		# 	# self.workload = np.zeros(self.pa.num_res)
		# 	# for i in range(self.pa.num_res):
		# 		# print(self.nw_size_seqs[:, i])
		# 		# print(self.nw_len_seqs)
		# 		# exit()
		# 		# self.workload[i] = \
  #   #                 np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
  #   #                 float(self.pa.res_slot) / \
  #   #                 float(len(self.nw_len_seqs))
		# 		# print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))


			# self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = \
   #          	self.read_work_files()

			self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = self.read_work_files()

			# print("test:")
			# print(self.nw_len_seqs.shape)

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

	def running_num(self):
		num = len(self.machine.running_job)

		return num

	def read_work_files(self):
		f1 = open("/home/veronica/.local/lib/python3.5/site-packages/gym/envs/gym_rs/nwlen.txt", 'r')
		lines1=f1.readlines()
		nw_len = []
		len_len = len(lines1)

		pro_len = self.pa.simu_len*self.pa.num_ex

		numl = 0 
		tem_list = []
		for line1 in lines1: 
			tem_l1 = line1.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l1.split()
			if numl % pro_len == 0 and numl != 0:
				nw_len.append(tem_list)
				tem_list = []
			tem_save = []
			for item in row:
				tem_save.append(int(item))

			tem_list.append(tem_save)

			if numl == len(lines1)/pro_len - 1:
				nw_len.append(tem_list)

			numl +=1

		f1.close()

		f2 = open("/home/veronica/.local/lib/python3.5/site-packages/gym/envs/gym_rs/nwsize.txt", 'r')
		lines2=f2.readlines()
		nw_size = []
		count = 0
		epos_list = []

		len_size = len(lines2)
		for line2 in lines2:
			tem_l2 = line2.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l2.split()
			if count % pro_len == 0 and count != 0: 
				nw_size.append(epos_list)
				epos_list = []
			tem_list = []

			for item in row:
				tem_list.append(int(item))

			epos_list.append(tem_list)

			if count == len(lines1)/pro_len -1:

				nw_size.append(epos_list)


			count += 1
		f2.close()

		f3 = open("/home/veronica/.local/lib/python3.5/site-packages/gym/envs/gym_rs/nworder.txt", 'r')
		lines3=f3.readlines()
		nw_order = []
		count2 = 0
		epos_order = []
		#print(len(lines2))
		len_order = len(lines3)
		# print(len_size)
		for line3 in lines3:
			tem_l3 = line3.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l3.split()
			if count2 % pro_len == 0 and count2 != 0: 
				nw_order.append(epos_order)
				epos_order = []
			tem_order = []
			for item in row:
				tem_order.append(int(item))

			epos_order.append(tem_order)

			if count2 == len(lines1)/pro_len -1:
				nw_order.append(epos_order)

			count2 += 1
		f3.close()

		f4 = open("/home/veronica/.local/lib/python3.5/site-packages/gym/envs/gym_rs/nwtrans.txt", 'r')
		lines4=f4.readlines()
		nw_trans = []
		count4 = 0
		epos_trans = []
		#print(len(lines2))
		len_trans = len(lines4)
		# print(len_size)
		for line4 in lines4:
			tem_l4 = line4.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l4.split()
			if count4 % pro_len == 0 and count4 != 0: 
				nw_trans.append(epos_trans)
				epos_trans = []
			tem_trans = []

			for item in row:
				tem_trans.append(int(item))

			epos_trans.append(tem_trans)

			if count4 == len(lines1)/pro_len -1:
				nw_trans.append(epos_trans)

			count4 += 1
		f4.close()

		# print(len(nw_len))
		# print(len(nw_size))
		# print(len(nw_order))
		# print(len(nw_trans))
		# print(len(nw_size[100]))
		# print(nw_len[0])
		# print(nw_size[0])
		# print(nw_len[2507])
		# print(nw_size[2507])

		# print(len_len)
		# print(len_size)
		# print(len_trans)
		# print(len_order)

		# exit()

		assert len_len == len_size
		assert len_len == len_order
		assert len_trans == len_len

		# epsi = 100
		# run_size = nw_size[:-1]
		tem_size = np.array(nw_size[0])
		# print(len(tem_size))
		# run_len = nw_len[:-1]
		tem_len = np.array(nw_len[0])
		tem_order = np.array(nw_order[0])
		tem_trans = np.array(nw_trans[0])
		# print(len(tem_trans))
		# print(tem_trans)
		# print(len(tem_trans[0]))

		return tem_len, tem_size, tem_order, tem_trans

	def read_files2(self):

		f1 = open("env/nwlen2.txt", 'r')
		lines1=f1.readlines()
		nw_len = []
		len_len = len(lines1)
		pro_len = self.pa.simu_len

		numl = 0 
		tem_list = []
		for line1 in lines1: 
			tem_l1 = line1.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l1.split()
			if numl % pro_len == 0 and numl != 0:
				nw_len.append(tem_list)
				tem_list = []
			tem_save = []
			for item in row:
				tem_save.append(int(item))

			tem_list.append(tem_save)

			if numl == len(lines1)/pro_len - 1:
				nw_len.append(tem_list)

			numl +=1

		f1.close()

		f2 = open("env/nwsize2.txt", 'r')
		lines2=f2.readlines()
		nw_size = []
		count = 0
		epos_list = []

		len_size = len(lines2)
		for line2 in lines2:
			tem_l2 = line2.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l2.split()
			if count % pro_len == 0 and count != 0: 
				nw_size.append(epos_list)
				epos_list = []
			tem_list = []

			for item in row:
				tem_list.append(int(item))

			epos_list.append(tem_list)

			if count == len(lines1)/pro_len -1:

				nw_size.append(epos_list)


			count += 1
		f2.close()

		f3 = open("env/nworder2.txt", 'r')
		lines3=f3.readlines()
		nw_order = []
		count2 = 0
		epos_order = []
		#print(len(lines2))
		len_order = len(lines3)
		# print(len_size)
		for line3 in lines3:
			tem_l3 = line3.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l3.split()
			if count2 % pro_len == 0 and count2 != 0: 
				nw_order.append(epos_order)
				epos_order = []
			tem_order = []
			for item in row:
				tem_order.append(int(item))

			epos_order.append(tem_order)

			if count2 == len(lines1)/pro_len -1:
				nw_order.append(epos_order)

			count2 += 1
		f3.close()

		f4 = open("env/nwtrans2.txt", 'r')
		lines4=f4.readlines()
		nw_trans = []
		count4 = 0
		epos_trans = []
		#print(len(lines2))
		len_trans = len(lines4)
		# print(len_size)
		for line4 in lines4:
			tem_l4 = line4.replace('[','').replace(']','')  #.replace(' ',',')
			row = tem_l4.split()
			if count4 % pro_len == 0 and count4 != 0: 
				nw_trans.append(epos_trans)
				epos_trans = []
			tem_trans = []

			for item in row:
				tem_trans.append(int(item))

			epos_trans.append(tem_trans)

			if count4 == len(lines1)/pro_len -1:
				nw_trans.append(epos_trans)

			count4 += 1
		f4.close()

		# print(len(nw_len))
		# print(len(nw_size))
		# print(len(nw_order))
		# print(len(nw_trans))
		# print(len(nw_size[100]))
		# print(nw_len[0])
		# print(nw_size[0])
		# print(nw_len[2507])
		# print(nw_size[2507])

		assert len_len == len_size
		assert len_len == len_order
		assert len_trans == len_len

		# epsi = 100
		# run_size = nw_size[:-1]
		tem_size = np.array(nw_size[0])
		# print(len(tem_size))
		# run_len = nw_len[:-1]
		tem_len = np.array(nw_len[0])
		tem_order = np.array(nw_order[0])
		tem_trans = np.array(nw_trans[0])
		# print(len(tem_trans))
		# # print(tem_trans)
		# print(len(tem_trans[0]))

		return tem_len, tem_size, tem_order, tem_trans


	def com_job_file(self):
		nw_len_seqs1, nw_size_seqs1, nw_sub_seqs1, nw_trans_seqs1 = self.read_work_files()
		nw_len_seqs2, nw_size_seqs2, nw_sub_seqs2, nw_trans_seqs2 = self.read_files2()

		nw_len_seqs = np.zeros((2, 50, 4))
		nw_size_seqs = np.zeros((2, 50, 4))
		nw_sub_seqs = np.zeros((2, 50, 4))
		nw_trans_seqs = np.zeros((2, 50, 4))
		
		nw_len_seqs[0, :] = nw_len_seqs1
		nw_len_seqs[1, :] = nw_len_seqs2

		nw_size_seqs[0, :] = nw_size_seqs1
		nw_size_seqs[1, :] = nw_size_seqs2

		nw_sub_seqs[0, :] = nw_sub_seqs1
		nw_sub_seqs[1, :] = nw_sub_seqs2

		nw_trans_seqs[0, :] = nw_trans_seqs1
		nw_trans_seqs[1, :] = nw_trans_seqs2

		return nw_len_seqs, nw_size_seqs, nw_sub_seqs, nw_trans_seqs


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
                      job_trans=self.nw_trans_seqs[seq_no, seq_idx, :])
		return new_job

	def get_sjf_len_action(self):
		action = np.zeros(1)

		length = np.zeros(self.pa.num_nw)

		for i in range(self.pa.num_nw):
			tem_length = 0
			if self.job_slot.slot[i] is not None:
				for j in range(self.pa.num_clu):
					if self.job_slot.slot[i].res_vec[j] != 0:
						tem_length += self.job_slot.slot[i].len[j]
						tem_length += self.job_slot.slot[i].trans[j]


			length[i] = tem_length


		# tem_action = np.argmax(length) #length.index(min(length))
		tem = []
		if np.any(length[:] > 0):

			for i in range(self.pa.num_nw):
				if length[i] != 0:
					tem.append(length[i])

			tem_np = np.array(tem)
			tem_inde = np.argmin(tem_np)
			tem_value = tem_np[tem_inde]

			true_index = np.where(length == tem_value)[0]


			action[0] = true_index+1


		return action


	def get_job_num(self):

		tem = 0
		for i in range(self.pa.num_nw):
			if self.job_slot.slot[i] != None:
				tem += 1

		return tem 


	def get_sjf_workload_action(self):
		action = np.zeros(1)

		wo_load = np.zeros(self.pa.num_nw)
		
		for i in range(self.pa.num_nw):
			if self.job_slot.slot[i] is not None:
				tem_load = self.get_job_workload(self.job_slot.slot[i])
				wo_load[i] = tem_load

		tem = []
		if np.any(wo_load[:] > 0):
			for i in range(self.pa.num_nw):
				if wo_load[i] != 0:
					tem.append(wo_load[i])

			tem_np = np.array(tem)
			tem_inde = np.argmin(tem_np)
			tem_value = tem_np[tem_inde]

			true_index = np.where(wo_load == tem_value)[0][0]

			action[0] = true_index + 1
		return action

	def get_random_action(self):
		action = np.zeros(1)

		action[0] = self.ransta.randint(0, self.pa.num_nw+1)

		return action

	def get_job_workload(self, job):
		load = 0
		for i in range(self.pa.num_clu):
			if job.res_vec[i] != 0:
				load += job.res_vec[i] * job.len[i]
			if job.trans[i] != 0:
				load += job.res_vec[i] * job.len[i] * self.pa.trans_ratio #10 #job.res_vec[i] * job.len[i] * self.pa.trans_ratio

		return load


	def get_job_len(self, job):
		leng = 0
		for i in range(self.pa.num_clu):
			if job.res_vec[i] != 0:
				leng += job.len[i]
			if job.trans[i] != 0:
				leng += job.trans[i]

		return leng

	def get_backlog_num(self):
		return self.job_backlog.curr_size

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
						image_repr[: self.job_slot.slot[j].len[i], ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = self.job_slot.slot[j].order[i] #1
				ir_pt += self.pa.max_job_size

		###########modified by zc###############
		# if self.job_backlog.curr_size == 0:
		# 	image_repr[0,ir_pt: ir_pt + backlog_width] = 1

		# else:
		# 	image_repr[0: self.job_backlog.curr_size / backlog_width,ir_pt: ir_pt + backlog_width] = 1
		########################################
		image_repr[: int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + backlog_width] = 255 #1 # 250


		# if self.job_backlog.curr_size % backlog_width > 0:
		# 	image_repr[self.job_backlog.curr_size / backlog_width,
  #                      ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1

		if self.job_backlog.curr_size % backlog_width > 0: # the last line of backlog is not complete
			image_repr[int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 255 #1 # 250
                         
		ir_pt += backlog_width

		# image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
  #                                             float(self.extra_info.max_tracking_time_since_last_job)

		image_repr[:, ir_pt: ir_pt + 1] = self.disslog.size

		ir_pt += 1

		# print(ir_pt)
		# print(image_repr.shape[1])

		assert ir_pt == image_repr.shape[1]
		
#		print("observation:")
#		print(image_repr)

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
				if j.res_vec[clu] != 0:
					length += j.len[clu] 
					length += j.trans[clu]
			reward += self.pa.delay_penalty / float(length)

		for j in self.job_slot.slot:
			if j is not None:
				# reward += self.pa.hold_penalty / float(j.len)
				length = 0
				for clu in range(self.pa.num_clu):
					if j.res_vec[clu] != 0:
						length += j.len[clu] 
						length += j.trans[clu]
				reward += self.pa.hold_penalty / float(length)

		for j in self.job_backlog.backlog:
			if j is not None:
				# reward += self.pa.dismiss_penalty / float(j.len)
				length = 0
				for clu in range(self.pa.num_clu):
					if j.res_vec[clu] != 0:
						length += j.len[clu] 
						length += j.trans[clu]
				reward += self.pa.dismiss_penalty / float(length)

		for j in self.disslog.log:
			if j is not None:
				# reward += self.pa.dismiss_penalty / float(j.len)
				length = 0
				for clu in range(self.pa.num_clu):
					if j.res_vec[clu] != 0:
						length += j.len[clu] 
						length += j.trans[clu]
				reward += self.pa.drop_penalty / float(length)
		return reward

	def step(self, a, repeat=False):
		status = None

		done = False
		reward = 0
		info = False #None

		if a == 0:  # explicit void action
			status = 'MoveOn'
		elif self.job_slot.slot[a-1] is None:  # implicit void action
			status = 'MoveOn'
		else:
			allocated = self.machine.allocate_job(self.job_slot.slot[a-1], self.curr_time, self.pa)
			print("allocated")
			print(allocated)
			if not allocated:  # implicit void action
				status = 'MoveOn'
			else:
				status = 'Allocate'

		if status == 'MoveOn':
			self.curr_time += 1
			self.machine.time_proceed(self.curr_time, self.pa)
			self.extra_info.time_proceed()

            # add new jobs
			self.seq_idx += 1

			info = True

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

					#for i in range(self.pa.num_nw):
					#	if self.job_slot.slot[i] is None and self.job_backlog.curr_size > 0:
					#		tem_job = self.job_backlog.backlog.popleft()
					#		self.job_slot.slot[i] = tem_job

					if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
						new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

						if np.any(new_job.res_vec > 0):  # a new job comes

							print("new job comes")

							# print(new_job.trans)
							
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


						# double job coming

						new_job = self.get_new_job_from_seq(self.seq_no+1, self.seq_idx)

						if np.any(new_job.res_vec > 0):  # a new job comes

							print("new job comes")

							# print(new_job.trans)
							
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


			tem_ac = []
			tem_ac.append(self.curr_time)
			act_job = self.job_slot.slot[a-1]
			fin_t = act_job.finish_time
			sta_t = act_job.start_time
			com_t = fin_t - sta_t

			ent_t = act_job.enter_time
			slow_com = fin_t - ent_t
			job_l = self.get_job_len(act_job)
			slowdown = float(slow_com/job_l)
			tem_ac.append(com_t)
			tem_ac.append(slow_com)
			tem_ac.append(slowdown)

			f=open("/home/veronica/Desktop/env/job_profile/0726/normal/act_job_info.txt", "a")
			f.write("%s\n" % tem_ac)
			f.close()

			# exit()
			self.job_record.record[self.job_slot.slot[a-1].id] = self.job_slot.slot[a-1]
			self.job_slot.slot[a-1] = None

            # dequeue backlog
			if self.job_backlog.curr_size > 0:
				# self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
				# self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
				# self.job_backlog.backlog[-1] = None
				# self.job_backlog.curr_size -= 1
				self.job_slot.slot[a-1] = self.job_backlog.backlog.popleft()
				self.job_backlog.backlog.append(None)
				self.job_backlog.curr_size -= 1



		ob = self.observe()

		# info = self.job_record

		# if done:
		# 	self.seq_idx = 0

		# 	if not repeat:
		# 		self.seq_no = (self.seq_no + 1) % self.pa.num_ex

		# 	self.reset()

		#if self.render:
		#	self.plot_state()
		print("status:")
		print(status)

		# print(done)
		# print(info)
		print("reward:")
		print(reward)
		# count = 0
		# for i in range(self.pa.num_nw):
		# 	if self.job_slot.slot[i] != None:
		# 		print(self.job_slot.slot[i].len)
		# 		print(self.job_slot.slot[i].res_vec)
		# 		count+=1

		# print("job num:")
		# print(count)
#		print("ob:")
#		print(ob)
		

# 		count_nw = 0
# 		for i in range(self.pa.num_nw):
# 			if self.job_slot.slot[i] != None:
# 				count_nw+=1


# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/job_queue_num.txt", "a")
# #		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
# 		# for item in self.nw_len_seqs:
# 		f.write("%s\n" % count_nw)
# 		f.close()


# 		tem_l = []
# 		bl_num = self.job_backlog.curr_size
# 		run_num = len(self.machine.running_job)
# 		tem_l.append(a)
# 		tem_l.append(count_nw)
# 		tem_l.append(bl_num)
# 		tem_l.append(run_num)
# 		tem_l.append(done)


# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/job_info.txt", "a")
# #		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
# 		# for item in self.nw_len_seqs:
# 		f.write("%s\n" % tem_l)
# 		f.close()
		



		return ob, reward, done, info



	def reset(self):


		# tem_seed = np.random.randint(9999)
		# self.ransta = np.random.RandomState(tem_seed)
		# self.pa = Parameters(self.ransta)


		# self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
		# # self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
		# # self.action_space = spaces.Box(0, self.pa.num_nw-1, shape=(1,), dtype = np.int16)
		# self.action_space = spaces.Discrete(self.pa.num_nw+1)
		# self.nw_dist = self.pa.dist.bi_model_dist

		# nw_len_seqs = None
		# nw_size_seqs = None

		# if nw_len_seqs is None or nw_size_seqs is None:
		# 	# generate new work
		# 	self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = \
  #           	self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
		# 	# self.workload = np.zeros(self.pa.num_res)
		# 	# for i in range(self.pa.num_res):
		# 		# print(self.nw_size_seqs[:, i])
		# 		# print(self.nw_len_seqs)
		# 		# exit()
		# 		# self.workload[i] = \
  #   #                 np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
  #   #                 float(self.pa.res_slot) / \
  #   #                 float(len(self.nw_len_seqs))
		# 		# print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
		# 	self.nw_len_seqs = np.reshape(self.nw_len_seqs,
  #                                          [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu]) 
  #                                          #[self.pa.num_ex, self.pa.simu_len])
		# 	self.nw_size_seqs = np.reshape(self.nw_size_seqs,
  #                                          [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
		# 	self.nw_sub_seqs = np.reshape(self.nw_sub_seqs,
  #                                          [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
		# 	self.nw_trans_seqs = np.reshape(self.nw_trans_seqs,
  #                                          [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
		# else:
		# 	self.nw_len_seqs = nw_len_seqs
		# 	self.nw_size_seqs = nw_size_seqs
		# 	self.nw_sub_seqs = nw_sub_seqs
		# 	self.nw_trans_seqs = nw_trans_seqs


# 		f1=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwlen.txt", "a")
# #		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
# 		for item in self.nw_len_seqs:
# 			f1.write("%s\n" % item)
# 		f1.close()
# #		f1.write(tem_str1)
# #		f1.close()

# 		f2=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwsize.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_size_seqs:
# 			f2.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f2.close()


# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/nworder.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_sub_seqs:
# 			f.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f.close()

# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwtrans.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_trans_seqs:
# 			f.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f.close()


# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/clu_trans.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		# print(self.pa.trans_rate)
# 		for item in self.pa.trans_rate:
# 			f.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f.close()

# 		total_load = [0] * self.pa.num_clu
		
# 		for seqt in range(self.pa.simu_len):
# 			for i in range(self.pa.num_clu):
# 				tem_load = self.nw_len_seqs[self.pa.num_ex-1][seqt][i] * self.nw_size_seqs[self.pa.num_ex-1][seqt][i]
# 				# tem_trans = self.nw_trans_seqs[self.pa.num_ex-1][seqt][i] 
# 				tem_load = 1.5 * tem_load
# 				total_load[i] += tem_load

# 		nord = (self.pa.simu_len + self.pa.time_horizon) * self.pa.res_slot
# 		workload = []
# 		for i in range(self.pa.num_clu):
# 			workload.append(total_load[i] / float(nord))

# 		f3=open("/home/veronica/Desktop/env/job_profile/0726/normal/workload.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		f3.write("%s\n" % workload)
# #		f2.write(tem_str2)
# 		f3.close()		
		self.seq_no = 0  # which example sequence
		self.seq_idx = -1  # index in that sequence
		self.curr_time = 0

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
    def __init__(self, pa, ranstate):
        self.num_res = pa.num_res
        self.num_clu = pa.num_clu
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.ranstate = ranstate

        self.avbl_slot = np.ones((self.time_horizon, self.num_clu)) * self.res_slot
        self.avbl_trans = np.ones((self.time_horizon, self.num_clu, self.num_clu))

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



#        print("job:")
#        print(job.trans)
#        print(job.trans[1])
        # print("end")

        leng = 0
        for i in range(self.num_clu):
        	if job.res_vec[i] != 0: leng += job.len[i]

        trans = 0
        for i in range(self.num_clu):
        	if job.res_vec[i] != 0: trans += job.trans[i]

        total_t = leng + trans

#        print("total time:")
#        print(total_t)

        # for i in range(1, self.num_clu+1):

        # 	tem_f = np.where(job.order == i)[0]
        # 	# print(job.order)
        # 	# print("tem_f")
        # 	# print(tem_f)
        # 	tem_f = int(tem_f)
        # 	if job.res_vec[tem_f] != 0: break


        # print(tem_f)

        begin_time = [None] * pa.num_clu

        begin_trans_time = [None] * pa.num_clu

        for i in range(pa.num_clu):
        	if job.res_vec[i] == 0:
        		begin_time[i] = -1 #0
        		begin_trans_time[i] = -1 #0
        	elif job.trans[i] == 0:
        		begin_trans_time[i] = -1 #0



        pass_time = 0

        future_time = total_t

        for i in range(1, self.num_clu+1):

        	first_ord = np.where(job.order == i)[0]

	        first_ord = int(first_ord)
        	if job.res_vec[first_ord] != 0: break


        tem_prev = None
        new_avbl_res = self.avbl_slot.copy()
        new_aval_trans = self.avbl_trans.copy()
        # print("size:")
        # print(new_aval_trans.shape)
        # print(new_avbl_res.shape)
        for i in range(1, self.num_clu+1):

        	# print(job.trans)

        	tem_f = np.where(job.order == i)[0]
        	# print(job.order)
        	# print("tem_f")
        	# print(tem_f)
        	if tem_f != None:
	        	tem_f = int(tem_f)
#	        	print("test")
#	        	print(job.trans)
        		if job.res_vec[tem_f] != 0: # break

        			# print(tem_f)
        			# print(job.trans)
        			# print(job.trans[tem_prev])
        			# print(tem_prev)
        			if tem_prev != None:
        				if job.trans[tem_prev] != 0:
        					# print("tem_f")
        					# print(tem_f)
        					# print(tem_prev)
        					# print(pass_time)
        					# print(self.time_horizon - future_time)
        					# tem_aval_trans = new_aval_trans
	        				for t in range(pass_time, self.time_horizon): # - future_time):
	        					if t+job.trans[tem_prev] <= self.time_horizon:
		        					tem_aval_trans = new_aval_trans[t:t+job.trans[tem_prev], tem_prev, tem_f]-1 
		        					# new_aval_trans[t:t+job.trans[tem_prev], tem_prev, tem_f] = new_aval_trans[t:t+job.trans[tem_prev], tem_prev, tem_f]-1 #self.avbl_trans[t:t+job.trans[tem_prev], tem_prev, tem_f] - 1
	    	    					# print("new_aval_trans:")
	        						# # print(new_aval_trans)
	        						# print(new_aval_trans[t:t+job.trans[tem_prev], tem_prev, tem_f])
	        						if np.all(tem_aval_trans[:] >= 0): # np.all(new_aval_trans[:] >= 0):
	        							begin_trans_time[tem_prev] = t #+ pass_time
	        							#print("begin_trans:")
	        							#print(t+pass_time)
	        							new_aval_trans[t:t+job.trans[tem_prev], tem_prev, tem_f] = tem_aval_trans
	        							tem_aval_trans = None
	        							pass_time = t + job.trans[tem_prev]
	        							future_time = total_t - pass_time
	        							# print(begin_trans_time[tem_prev])
	        							break
	        					# else: new_aval_trans = tem_aval_trans
	        		# tem_aval_res = new_avbl_res
        			for t in range(pass_time, self.time_horizon): # - future_time):
        				
        				# print("check")
        				# print(new_avbl_res.shape)
        				# print(job.res_vec[tem_f])
        				# print(t+job.len[tem_f])
        				# print(new_avbl_res[0])
        				if t+job.len[tem_f] <= self.time_horizon:
	        				tem_aval_res = new_avbl_res[t:t+job.len[tem_f], tem_f]- job.res_vec[tem_f] #self.avbl_slot[t:t+job.len[tem_f], tem_f] - job.res_vec[tem_f]
    	    				# print("new_aval_res:")
        					# print(t)
        					# print(tem_f)
        					# print(new_avbl_res[t, tem_f])
        					# print(new_avbl_res[t:t+job.len[tem_f], tem_f])
        					# print(new_avbl_res.shape)
        					# if np.all(new_avbl_res[t:t+job.len[tem_f], tem_f] >= 0):
        					# 	print("test")
        					# 	print(new_avbl_res)
        					if np.all(tem_aval_res[:] >= 0):   # (new_avbl_res[:] >= 0):
        						if first_ord == tem_f: job_begin = t
	        					begin_time[tem_f] = t #+ pass_time
    	    					#print("begin_t:")
        						#print(t+pass_time)
        						new_avbl_res[t:t+job.len[tem_f], tem_f] = tem_aval_res
        						tem_aval_res = None
        						pass_time = t + job.len[tem_f] #+ job.trans[tem_f]
        						future_time = total_t - pass_time
        						#time_prev = time_f
        						tem_prev = tem_f
        						break
        					# else: new_avbl_res = tem_aval_res

        				# break
        # print("begin")
       	# print(begin_time)
       	# print(begin_trans_time)

        if all([v != None for v in begin_time]) and all([v != None for v in begin_trans_time]):

        	allocated = True
        	# self.avbl_slot  = new_avbl_res #[t: t + total_t, :] = new_avbl_res[t: t + total_t, :]
        	# self.avbl_trans = new_aval_trans

        	last_begin_index = begin_time.index(max(begin_time))
        	last_begin_time = begin_time[last_begin_index] + job.len[last_begin_index]

        	job.start_time = curr_time + job_begin
        	job.finish_time = curr_time + last_begin_time #job.start_time + total_t

        	self.avbl_slot[job_begin:last_begin_time, :]  = new_avbl_res[job_begin:last_begin_time, :]
        	self.avbl_trans[job_begin:last_begin_time,:,:] = new_aval_trans[job_begin:last_begin_time,:,:]

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
        	# canvas_start_time = job.start_time - curr_time
        	# canvas_end_time = job.finish_time - curr_time

        	tem_time = job.start_time - curr_time
        	tem_prev1 = None


        	for i in range(1, self.num_clu+1):
        		tem_index = np.where(job.order == i)[0]
        			
        		if tem_index != None:
	        		tem_index = int(tem_index)
        		if tem_index != None and tem_prev1 != None and job.res_vec[tem_index] != 0 and job.trans[tem_prev1] != 0 and begin_trans_time[tem_prev] != -1:
        			assert tem_time == job.len[tem_prev1] + begin_time[tem_prev1]
        			trans_end_time = job.trans[tem_prev1]
        			tem_time = begin_trans_time[tem_prev1]
        			assert tem_time + trans_end_time < self.time_horizon
        			for t1 in range(tem_time, tem_time+trans_end_time):
#        					print("again")
#        					print(tem_prev1)
#        					print(tem_index)
#        					print(t1)
        					# print(total_t)
#        					print(tem_time)
#        					print(trans_end_time)
        					# print(tem_time+trans_end_time)
        					# print(job.trans)
        					# print(job.order)
        					# print(job.len)
        					# print(self.trans_canvas[tem_prev,tem_index, t1, :])
        					# avbl_trans_slot = np.where(self.trans_canvas[tem_prev1,tem_index, t1, :] == 0)[0]
        					# print("avbl_trans_slot[: pa.trans_rate[tem_prev1, tem_index]]")
        					# print(avbl_trans_slot[: pa.trans_rate[tem_prev1, tem_index]])
        					# exit()
        				self.trans_canvas[tem_prev1,tem_index, t1, 0:pa.trans_rate[tem_prev1, tem_index]] = new_color #avbl_trans_slot[: pa.trans_rate[tem_prev1, tem_index]]] = new_color
        			tem_time += trans_end_time

        		if tem_index != None and job.res_vec[tem_index] != 0 and begin_time[tem_index] != -1:
        			sub_end_time = job.len[tem_index]
        			tem_time = begin_time[tem_index]
        			assert tem_index + sub_end_time <= self.time_horizon
        			for t2 in range(tem_time, tem_time+sub_end_time):
#        				print("stat:")
#        				print(tem_index)
#        				print(t2)
#        				print(tem_time)
#        				print(sub_end_time)
        				avbl_slot = np.where(self.canvas[tem_index, t2, :] == 0)[0]
        				self.canvas[tem_index, t2, avbl_slot[: job.res_vec[tem_index]]] = new_color
        			tem_time += sub_end_time
        			tem_prev1 = tem_index

        # # for t in range(0, self.time_horizon -total_t):
        # # 	# print(job.len[tem_f][0])
        # # 	new_avbl_res[t:t+job.len[tem_f], tem_f] = self.avbl_slot[t:t+job.len[tem_f], tem_f] - job.res_vec[tem_f]
        # # 	tem_t = t+job.len[tem_f]+job.trans[tem_f]
        # # 	# print(tem_f)
        # # 	# print(int(tem_f[0]))
        # # 	begin_time = [None] * pa.num_clu
        # # 	begin_time[tem_f] = t

        # # 	ind_prev = tem_f
        # # 	for j in range(1, self.num_clu-tem_f+1):
        # # 		tem_in = np.where(job.order == tem_f+j)[0]
        # # 		# print(tem_in)
        # # 		if tem_in != None:
	       # #  		tem_in = int(tem_in[0])
        # # 		# print(tem_in)
        # # 		if tem_in != None and job.res_vec[tem_in] != 0:

        # # 			new_avbl_res[tem_t:tem_t+job.len[tem_in], tem_in] = self.avbl_slot[tem_t:tem_t+job.len[tem_in], tem_in] - job.res_vec[tem_in]
        # # 			if job.trans[tem_in] != 0:
        # # 				tem_t = tem_t+job.len[tem_in]+job.trans[tem_in]
        # # 			else: tem_t = tem_t+job.len[tem_in]





        # 	if np.all(new_avbl_res[:] >= 0):

        # 		# print("job:")
        # 		# print(job.trans)
        # 		# print("end")

        # 		allocated = True

        # 		self.avbl_slot[t: t + total_t, :] = new_avbl_res[t: t + total_t, :]
        # 		job.start_time = curr_time + t
        # 		job.finish_time = job.start_time + total_t

        # 		self.running_job.append(job)

        # 		# update graphical representation

        # 		used_color = np.unique(self.canvas[:])
        # 		# WARNING: there should be enough colors in the color map
        # 		for color in self.colormap:
        # 			if color not in used_color:
        # 				new_color = color
        # 				break

        # 		assert job.start_time != -1
        # 		assert job.finish_time != -1
        # 		assert job.finish_time > job.start_time
        # 		canvas_start_time = job.start_time - curr_time
        # 		canvas_end_time = job.finish_time - curr_time

        # 		tem_time = job.start_time - curr_time
        # 		tem_prev = None

        # 		# print("job:")
        # 		# print(job.trans)
        # 		# print("end")

        # 		for i in range(tem_f, self.num_clu+1):
        # 			tem_index = np.where(job.order == i)[0]
        			
        # 			if tem_index != None:
	       #  			tem_index = int(tem_index)
        # 			if tem_index != None and tem_prev != None and job.res_vec[tem_index] != 0 and job.trans[tem_prev] != 0:
        # 				trans_end_time = job.trans[tem_prev]
        # 				for t1 in range(tem_time, tem_time+trans_end_time):
        # 					# print("start")
        # 					# print(tem_prev)
        # 					# print(tem_index)
        # 					# print(t1)
        # 					# print(total_t)
        # 					# print(tem_time)
        # 					# print(trans_end_time)
        # 					# print(tem_time+trans_end_time)
        # 					# print(job.trans)
        # 					# print(job.order)
        # 					# print(job.len)
        # 					# print(self.trans_canvas[tem_prev,tem_index, t1, :])
        # 					avbl_trans_slot = np.where(self.trans_canvas[tem_prev,tem_index, t1, :] == 0)[0]
        # 					self.trans_canvas[tem_prev,tem_index, t1, avbl_trans_slot[: pa.trans_rate[tem_prev, tem_index]]] = new_color
        # 				tem_time += trans_end_time

        # 			if tem_index != None and job.res_vec[tem_index] != 0:
        # 				sub_end_time = job.len[tem_index]
        # 				for t2 in range(tem_time, tem_time+sub_end_time):
        # 					avbl_slot = np.where(self.canvas[tem_index, t2, :] == 0)[0]
        # 					self.canvas[tem_index, t2, avbl_slot[: job.res_vec[tem_index]]] = new_color
        # 				tem_time += sub_end_time
        # 				tem_prev = tem_index

        # 		break

        return allocated

    def time_proceed(self, curr_time, pa):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot
        # print("avbl slot")
        # print(self.avbl_slot)
        # print(self.res_slot)
        # print(self.avbl_slot.shape)
        self.avbl_trans[:-1, :, :] = self.avbl_trans[1:, :, :]
        # print(self.avbl_trans[0])
        # print(self.avbl_trans.shape[-1,:,:])
        self.avbl_trans[-1,:,:] = 1

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



######################modified 07/22 below ################################
# 	def generate_sequence_work(self, joblist_len):
# 		nw_len_seq = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
# 		nw_size_seq = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
# 		nw_sub_seqs = np.zeros((joblist_len, self.pa.num_clu), dtype=int)
# 		nw_trans_seqs = np.zeros((joblist_len, self.pa.num_clu), dtype=int)

# 		for i in range(joblist_len):

# 			if np.random.rand() < self.pa.new_job_rate:  # a new job comes
# 				nw_len_seq[i, :], nw_size_seq[i, :], nw_sub_seqs[i,:], nw_trans_seqs[i, :] = self.nw_dist()
# 		return nw_len_seq, nw_size_seq, nw_sub_seqs, nw_trans_seqs

# 	def get_new_job_from_seq(self, seq_no, seq_idx):
# 		new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
#                       job_len=self.nw_len_seqs[seq_no, seq_idx, :],
#                       job_id=len(self.job_record.record),
#                       enter_time=self.curr_time,
#                       job_order=self.nw_sub_seqs[seq_no, seq_idx, :],
#                       job_trans=self.nw_sub_seqs[seq_no, seq_idx, :])
# 		return new_job

# 	def observe(self):
# 		backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
# 		image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

# 		ir_pt = 0

# 		for i in range(self.pa.num_clu):
# 			image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
# 			ir_pt += self.pa.res_slot

# 			for k in range(self.pa.num_clu):
# 				image_repr[:,ir_pt:ir_pt + self.pa.max_transmiss_rate] = self.machine.trans_canvas[i, k, :, :]
# 				ir_pt += self.pa.max_transmiss_rate
			
# 			for j in range(self.pa.num_nw):
# 				if self.job_slot.slot[j] is not None:  # fill in a block of work
# 					if self.job_slot.slot[j].res_vec[i] != 0:
# 						image_repr[: self.job_slot.slot[j].len[i], ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1
# 				ir_pt += self.pa.max_job_size

# 		###########modified by zc###############
# 		# if self.job_backlog.curr_size == 0:
# 		# 	image_repr[0,ir_pt: ir_pt + backlog_width] = 1

# 		# else:
# 		# 	image_repr[0: self.job_backlog.curr_size / backlog_width,ir_pt: ir_pt + backlog_width] = 1
# 		########################################
# 		image_repr[: int(self.job_backlog.curr_size / backlog_width),
#                        ir_pt: ir_pt + backlog_width] = 1 # 250


# 		# if self.job_backlog.curr_size % backlog_width > 0:
# 		# 	image_repr[self.job_backlog.curr_size / backlog_width,
#   #                      ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1

# 		if self.job_backlog.curr_size % backlog_width > 0: # the last line of backlog is not complete
# 			image_repr[int(self.job_backlog.curr_size / backlog_width),
#                        ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1 # 250

# 		ir_pt += backlog_width

# 		# image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
#   #                                             float(self.extra_info.max_tracking_time_since_last_job)

# 		image_repr[:, ir_pt: ir_pt + 1] = self.disslog.size

# 		ir_pt += 1

# 		print(ir_pt)
# 		print(image_repr.shape[1])

# 		assert ir_pt == image_repr.shape[1]

# 		return image_repr

	
# 	# def plot_state(self):
# 	# 	plt.figure("screen", figsize=(20, 5))

# 	# 	skip_row = 0

# 	# 	for i in range(self.pa.num_res):

# 	# 		plt.subplot(self.pa.num_res,
#  #                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
#  #                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
			
# 	# 		plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

# 	# 		for j in range(self.pa.num_nw):

# 	# 			job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))

# 	# 			if self.job_slot.slot[j] is not None:  # fill in a block of work

# 	# 				job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

# 	# 			plt.subplot(self.pa.num_res,
#  #                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
#  #                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

# 	# 			plt.imshow(job_slot, interpolation='nearest', vmax=1)

# 	# 			if j == self.pa.num_nw - 1:
# 	# 				skip_row += 1

# 	# 	skip_row -= 1
# 	# 	backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
# 	# 	backlog = np.zeros((self.pa.time_horizon, backlog_width))

# 	# 	# backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
# 	# 	backlog[: int(self.job_backlog.curr_size / backlog_width), : backlog_width] = 1

# 	# 	# backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1
# 	# 	tem_ra = (self.job_backlog.curr_size % backlog_width)
# 	# 	backlog[int(self.job_backlog.curr_size / backlog_width), : tem_ra] = 1

# 	# 	plt.subplot(self.pa.num_res,
#  #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
#  #                    self.pa.num_nw + 1 + 1)

# 	# 	plt.imshow(backlog, interpolation='nearest', vmax=1)

# 	# 	plt.subplot(self.pa.num_res,
#  #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
#  #                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

# 	# 	extra_info = np.ones((self.pa.time_horizon, 1)) * \
#  #                     self.extra_info.time_since_last_new_job / \
#  #                     float(self.extra_info.max_tracking_time_since_last_job)

# 	# 	plt.imshow(extra_info, interpolation='nearest', vmax=1)

# 	# 	plt.show()     # manual
#  #        # plt.pause(0.01)  # automatic

# 	def get_reward(self):

# 		reward = 0
# 		for j in self.machine.running_job:
# 			length = 0
# 			for clu in range(self.pa.num_clu):
# 				if j.res_vec[clu] != 0:
# 					length += j.len[clu] 
# 					length += j.trans[clu]
# 			reward += self.pa.delay_penalty / float(length)

# 		for j in self.job_slot.slot:
# 			if j is not None:
# 				# reward += self.pa.hold_penalty / float(j.len)
# 				length = 0
# 				for clu in range(self.pa.num_clu):
# 					if j.res_vec[clu] != 0:
# 						length += j.len[clu] 
# 						length += j.trans[clu]
# 				reward += self.pa.hold_penalty / float(length)

# 		for j in self.job_backlog.backlog:
# 			if j is not None:
# 				# reward += self.pa.dismiss_penalty / float(j.len)
# 				length = 0
# 				for clu in range(self.pa.num_clu):
# 					if j.res_vec[clu] != 0:
# 						length += j.len[clu] 
# 						length += j.trans[clu]
# 				reward += self.pa.dismiss_penalty / float(length)

# 		for j in self.disslog.log:
# 			if j is not None:
# 				# reward += self.pa.dismiss_penalty / float(j.len)
# 				length = 0
# 				for clu in range(self.pa.num_clu):
# 					if j.res_vec[clu] != 0:
# 						length += j.len[clu] 
# 						length += j.trans[clu]
# 				reward += self.pa.drop_penalty / float(length)
# 		return reward

# 	def step(self, a, repeat=False):
# 		status = None

# 		done = False
# 		reward = 0
# 		info = False #None

# 		if a == self.pa.num_nw:  # explicit void action
# 			status = 'MoveOn'
# 		elif self.job_slot.slot[a] is None:  # implicit void action
# 			status = 'MoveOn'
# 		else:
# 			allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time, self.pa)
# 			if not allocated:  # implicit void action
# 				status = 'MoveOn'
# 			else:
# 				status = 'Allocate'

# 		if status == 'MoveOn':
# 			self.curr_time += 1
# 			self.machine.time_proceed(self.curr_time)
# 			self.extra_info.time_proceed()

#             # add new jobs
# 			self.seq_idx += 1

# 			info = True

# 			if self.end == "no_new_job":  # end of new job sequence
# 				if self.seq_idx >= self.pa.simu_len:
# 					done = True
# 			elif self.end == "all_done":  # everything has to be finished
# 				if self.seq_idx >= self.pa.simu_len and \
#                    len(self.machine.running_job) == 0 and \
#                    all(s is None for s in self.job_slot.slot) and \
#                    all(s is None for s in self.job_backlog.backlog):
# 					done = True
# 				elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
# 					done = True

# 				if not done:

# 					for i in range(self.pa.num_nw):
# 						if self.job_slot.slot[i] is None and self.job_backlog.curr_size > 0:
# 							tem_job = self.job_backlog.backlog.popleft()
# 							self.job_slot.slot[i] = tem_job

# 					if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
# 						new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

# 						if np.any(new_job.res_vec > 0):  # a new job comes

# 							print("new job comes")
							
# 							to_backlog = True
# 							for i in range(self.pa.num_nw):
# 								if self.job_slot.slot[i] is None:  # put in new visible job slots
# 									self.job_slot.slot[i] = new_job
# 									self.job_record.record[new_job.id] = new_job
# 									to_backlog = False
# 									break
# 							if to_backlog:
# 								if self.job_backlog.curr_size < self.pa.backlog_size:
# 									self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
# 									self.job_backlog.curr_size += 1
# 									self.job_record.record[new_job.id] = new_job
# 								else:  # abort, backlog full
# 									print("Backlog is full.")
# 									self.disslog.log[self.disslog.size] = new_job
# 									self.disslog.size += 1
#                                 	# exit(1)
# 							self.extra_info.new_job_comes()
# 			reward = self.get_reward()

# 		elif status == 'Allocate':
# 			self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
# 			self.job_slot.slot[a] = None

#             # dequeue backlog
# 			if self.job_backlog.curr_size > 0:
# 				# self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
# 				# self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
# 				# self.job_backlog.backlog[-1] = None
# 				# self.job_backlog.curr_size -= 1
# 				self.job_slot.slot[a] = self.job_backlog.backlog.popleft()
# 				self.job_backlog.backlog.append(None)
# 				self.job_backlog.curr_size -= 1



# 		ob = self.observe()

# 		# info = self.job_record

# 		# if done:
# 		# 	self.seq_idx = 0

# 		# 	if not repeat:
# 		# 		self.seq_no = (self.seq_no + 1) % self.pa.num_ex

# 		# 	self.reset()

# 		#if self.render:
# 		#	self.plot_state()
# 		print("status:")
# 		print(status)
# 		# print(done)
# 		# print(info)
# 		print("reward:")
# 		print(reward)


# 		return ob, reward, done, info



# 	def reset(self):


# 		tem_seed = np.random.randint(9999)
# 		self.ransta = np.random.RandomState(tem_seed)
# 		self.pa = Parameters(self.ransta)
# 		# self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
# 		# # self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
# 		# # self.action_space = spaces.Box(0, self.pa.num_nw-1, shape=(1,), dtype = np.int16)
# 		# self.action_space = spaces.Discrete(self.pa.num_nw+1)
# 		# self.nw_dist = self.pa.dist.bi_model_dist

# 		nw_len_seqs = None
# 		nw_size_seqs = None

# 		if nw_len_seqs is None or nw_size_seqs is None:
# 			# generate new work
# 			self.nw_len_seqs, self.nw_size_seqs, self.nw_sub_seqs, self.nw_trans_seqs = \
#             	self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
# 			# self.workload = np.zeros(self.pa.num_res)
# 			# for i in range(self.pa.num_res):
# 				# print(self.nw_size_seqs[:, i])
# 				# print(self.nw_len_seqs)
# 				# exit()
# 				# self.workload[i] = \
#     #                 np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
#     #                 float(self.pa.res_slot) / \
#     #                 float(len(self.nw_len_seqs))
# 				# print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
# 			self.nw_len_seqs = np.reshape(self.nw_len_seqs,
#                                            [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu]) 
#                                            #[self.pa.num_ex, self.pa.simu_len])
# 			self.nw_size_seqs = np.reshape(self.nw_size_seqs,
#                                            [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
# 			self.nw_sub_seqs = np.reshape(self.nw_sub_seqs,
#                                            [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
# 			self.nw_trans_seqs = np.reshape(self.nw_trans_seqs,
#                                            [self.pa.num_ex, self.pa.simu_len, self.pa.num_clu])
# 		else:
# 			self.nw_len_seqs = nw_len_seqs
# 			self.nw_size_seqs = nw_size_seqs
# 			self.nw_sub_seqs = nw_sub_seqs
# 			self.nw_trans_seqs = nw_trans_seqs


# 		f1=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwlen.txt", "a")
# #		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
# 		for item in self.nw_len_seqs:
# 			f1.write("%s\n" % item)
# 		f1.close()
# #		f1.write(tem_str1)
# #		f1.close()

# 		f2=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwsize.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_size_seqs:
# 			f2.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f2.close()


# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/nworder.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_sub_seqs:
# 			f.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f.close()

# 		f=open("/home/veronica/Desktop/env/job_profile/0726/normal/nwtrans.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		for item in self.nw_trans_seqs:
# 			f.write("%s\n" % item)
# #		f2.write(tem_str2)
# 		f.close()


# 		total_load = [0] * self.pa.num_clu
		
# 		for seqt in range(self.pa.simu_len):
# 			for i in range(self.pa.num_clu):
# 				tem_load = self.nw_len_seqs[self.pa.num_ex-1][seqt][i] * self.nw_size_seqs[self.pa.num_ex-1][seqt][i]
# 				total_load[i] += tem_load

# 		nord = (self.pa.episode_max_length + self.pa.time_horizon) * self.pa.res_slot
# 		workload = []
# 		for i in range(self.pa.num_clu):
# 			workload.append(total_load[i] / float(nord))

# 		f3=open("/home/veronica/Desktop/env/job_profile/0726/normal/workload.txt", "a")
# #		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
# 		f3.write("%s\n" % workload)
# #		f2.write(tem_str2)
# 		f3.close()		
# 		self.seq_no = 0  # which example sequence
# 		self.seq_idx = -1  # index in that sequence
# 		self.curr_time = 0

#         # initialize system
# 		self.machine = Machine(self.pa, self.ransta)
# 		self.job_slot = JobSlot(self.pa)
# 		self.job_backlog = JobBacklog(self.pa)
# 		self.job_record = JobRecord()
# 		self.extra_info = ExtraInfo(self.pa)

# 		self.disslog = Dismisslog(self.pa)
# 		self.usedcolormap = []
# 		self.usedcolormap.append(0)
# 		self.state = self.observe()



# 		return self.state

# 	# def render(self, mode='human', close=False):
# 	# 	if not close:
#  #            raise NotImplementedError(
#  #                "This environment does not support rendering")





# class Job:
#     def __init__(self, res_vec, job_len, job_id, enter_time, job_order, job_trans):
#         self.id = job_id
#         self.res_vec = res_vec
#         self.len = job_len
#         self.enter_time = enter_time
#         self.start_time = -1  # not being allocated
#         self.finish_time = -1
#         self.order = job_order
#         self.trans = job_trans


# class JobSlot:
#     def __init__(self, pa):
#         self.slot = [None] * pa.num_nw


# class JobBacklog:
#     def __init__(self, pa):
#         # self.backlog = [None] * pa.backlog_size
#         self.backlog = deque()
#         for i in range(pa.backlog_size):
#         	self.backlog.append(None)
#         self.curr_size = 0


# class JobRecord:
#     def __init__(self):
#         self.record = {}

# class Dismisslog:
# 	def __init__(self, pa):
# 		self.log = [None] * pa.episode_max_length
# 		self.size = 0

# class Transqueue_clus:
# 	def __init__(self, pa):
# 		self.queue = [None] * pa.num_clu
# 		# self.queue = deque()
# 		for i in range(pa.num_clu):
# 			self.queue[i] = deque()
# 			for j in range(pa.episode_max_length):
# 				self.queue[i].append(None)

# 		self.size = [0] * pa.num_clu

# 		# for i in range(pa.num_clu):
# 		# 	self.queue[i] = [None] * pa.episode_max_length

# class Machine:
#     def __init__(self, pa, ranstate):
#         self.num_res = pa.num_res
#         self.num_clu = pa.num_clu
#         self.time_horizon = pa.time_horizon
#         self.res_slot = pa.res_slot
#         self.ranstate = ranstate

#         self.avbl_slot = np.ones((self.time_horizon, self.num_clu)) * self.res_slot

#         self.running_job = []

#         # colormap for graphical representation
#         # self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
#         self.colormap = np.arange(5, 201, 2)
#         np.random.shuffle(self.colormap)

#         # graphical representation
#         self.canvas = np.zeros((self.num_clu, self.time_horizon, self.res_slot))

#         # self.trans_queue = [None] * self.num_clu

#         # for i in range(pa.num_clu):
#         # 	self.trans_queue[i] = [None] * self.num_clu

#         self.trans_canvas = np.zeros((self.num_clu, self.num_clu,self.time_horizon, pa.max_transmiss_rate))

#         # for i in range(pa.num_clu):
#         # 	for j in range(pa.num_clu):
#         # 		for k in range(pa.time_horizon):
#         # 			tem_num = pa.trans_rate[i][j]
#         # 			self.trans_canvas[i][j][k][0:tem_num] = 1

#     def allocate_job(self, job, curr_time, pa):

#         allocated = False

#         new_avbl_res = self.avbl_slot

#         leng = 0
#         for i in range(self.num_clu):
#         	if job.res_vec[i] != 0: leng += job.len[i]

#         trans = 0
#         for i in range(self.num_clu):
#         	trans += job.trans[i]

#         total_t = leng + trans

#         for i in range(1, self.num_clu+1):

#         	tem_f = np.where(job.order == i)[0]
#         	tem_f = int(tem_f)
#         	if job.res_vec[tem_f] != 0: break


#         for t in range(0, self.time_horizon -total_t):
#         	# print(job.len[tem_f][0])
#         	new_avbl_res[t:t+job.len[tem_f], :] = self.avbl_slot[t:t+job.len[tem_f], :] - job.res_vec[tem_f]
#         	tem_t = t+job.len[tem_f]+job.trans[tem_f]
#         	# print(tem_f)
#         	# print(int(tem_f[0]))
#         	for j in range(1, self.num_clu-tem_f+1):
#         		tem_in = np.where(job.order == tem_f+j)[0]
#         		# print(tem_in)
#         		if tem_in != None:
# 	        		tem_in = int(tem_in[0])
#         		# print(tem_in)
#         		if tem_in != None and job.res_vec[tem_in] != 0:
#         			new_avbl_res[tem_t:tem_t+job.len[tem_in], :] = self.avbl_slot[tem_t:tem_t+job.len[tem_in], :] - job.res_vec[tem_f]
#         			if job.trans[tem_in] != 0:
#         				tem_t = tem_t+job.len[tem_in]+job.trans[tem_in]

#         	if np.all(new_avbl_res[:] >= 0):

#         		allocated = True

#         		self.avbl_slot[t: t + total_t, :] = new_avbl_res[t: t + total_t, :]
#         		job.start_time = curr_time + t
#         		job.finish_time = job.start_time + total_t

#         		self.running_job.append(job)

#         		# update graphical representation

#         		used_color = np.unique(self.canvas[:])
#         		# WARNING: there should be enough colors in the color map
#         		for color in self.colormap:
#         			if color not in used_color:
#         				new_color = color
#         				break

#         		assert job.start_time != -1
#         		assert job.finish_time != -1
#         		assert job.finish_time > job.start_time
#         		canvas_start_time = job.start_time - curr_time
#         		canvas_end_time = job.finish_time - curr_time

#         		tem_time = job.start_time - curr_time
#         		tem_prev = None
#         		for i in range(tem_f, self.num_clu):
#         			tem_index = np.where(job.order == i)[0]
        			
#         			if tem_index != None:
# 	        			tem_index = int(tem_index)
#         			if tem_index != None and tem_prev != None and job.res_vec[tem_index] != 0 and job.trans[tem_prev] != 0:
#         				trans_end_time = job.trans[tem_prev]
#         				for t in range(tem_time, tem_time+trans_end_time):
#         					avbl_slot = np.where(self.trans_canvas[tem_prev,tem_index, t, :] == 0)[0]
#         					self.trans_canvas[tem_prev,tem_index, t, avbl_slot[: pa.trans_rate[tem_prev, tem_index]]] = new_color
#         				tem_time += trans_end_time

#         			if job.res_vec[tem_index] != 0:
#         				sub_end_time = job.len[tem_index]
#         				for t in range(tem_time, tem_time+sub_end_time):
#         					avbl_slot = np.where(self.canvas[tem_index, t, :] == 0)[0]
#         					self.canvas[tem_index, t, avbl_slot[: job.res_vec[tem_index]]] = new_color
#         				tem_time += sub_end_time
#         				tem_prev = tem_index

#         		break

#         return allocated

#     def time_proceed(self, curr_time):

#         self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
#         self.avbl_slot[-1, :] = self.res_slot

#         for job in self.running_job:

#             if job.finish_time <= curr_time:
#                 self.running_job.remove(job)

#         # update graphical representation

#         self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
#         self.canvas[:, -1, :] = 0

#         self.trans_canvas[:,:,:-1,:] = self.trans_canvas[:,:, 1:,:]
#         self.trans_canvas[:,:,-1,:] = 0


# class ExtraInfo:
#     def __init__(self, pa):
#         self.time_since_last_new_job = 0
#         self.max_tracking_time_since_last_job = pa.max_track_since_new

#     def new_job_comes(self):
#         self.time_since_last_new_job = 0

#     def time_proceed(self):
#         if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
#             self.time_since_last_new_job += 1

