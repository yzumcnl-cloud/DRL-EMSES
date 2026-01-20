#import tensorflow as tf
from cProfile import label
import tensorflow.compat.v1 as tf
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
from env_SAGIN import SFC
import sys
import pickle
import time
import datetime

UE = 100

RUN_WoMA = False # if run DRL-EMSES-WoMA, set True.
   
# path = "./result/DDPG/" + datetime.datetime.now().strftime("%m-%d_%H-%M-%S_") + description_str + "/"
path = "./result/DDPG/" + "114514" + "/"

start = time.time()


#測試是哪個gpu or not
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.disable_eager_execution()
np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################
env2 = SFC(UE)
save_dir = path #"result/first_compare/no_algorithm/DDPG/"+str(env2.MAX_USER)+"_User/"
MAX_EPISODES = 800 #不用訓練到80000，可以訓練3000
MAX_EP_STEPS = UE*9
# MAX_EP_STEPS = env2.MAX_USER*7.7 #for 服務連續性，不同數量的使用者 可以跑相同的比例的migration(服務連續)。
# MAX_EP_STEPS = int(MAX_EP_STEPS)
#如果step 與user有關係
#服務連續性指的是使用者migration中斷率 ex 只有一個user 走10步 都不會斷 100%不中斷
#                     #ex 只有一個user 走10步 有一步不會斷 90%不中斷 10%中斷

TIMES_OF_DO_AVERAGE = 10
TIMES_OF_DO_OUTPUT = 100

TIMES_OF_DO_SFC_AVERAGE = 100

GAMMA = 0.9     # reward discount
OUTPUT_GRAPH = False

MAX_INT = sys.maxsize

#DDPG
DDPG_LR_A = 0.00000000005    # learning rate for actor
DDPG_LR_C = 0.0000000001    # learning rate for critic

# DDPG_LR_A = 0.00000000000005    # learning rate for actor
# DDPG_LR_C = 0.0000000000001    # learning rate for critic

TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

###############  Record average reward graph data  ###############
def draw_picture(p_r_lst_DDPG, p_accept_lst_DDPG, p_E2E_delay_lst_DDPG, 
                 p_long_r_lst_DDPG, p_long_accept_lst_DDPG, p_long_E2E_delay_lst_DDPG,
                 m_r_lst_DDPG, m_accept_lst_DDPG, m_E2E_delay_lst_DDPG, m_mig_time_lst_DDPG, 
                 m_long_r_lst_DDPG, m_long_accept_lst_DDPG, total_long_accept_lst_DDPG, m_long_E2E_delay_lst_DDPG, m_long_mig_time_lst_DDPG,
                 max_exist_user_lst_DDPG, long_max_exist_user_lst_DDPG,
                 p_SFC1_long_num_lst, p_SFC2_long_num_lst, p_SFC3_long_num_lst,
                 m_SFC1_long_num_lst, m_SFC2_long_num_lst, m_SFC3_long_num_lst,
                 p_SFC1_long_sucess_rate_lst, p_SFC2_long_sucess_rate_lst, p_SFC3_long_sucess_rate_lst,
                 m_SFC1_long_sucess_rate_lst, m_SFC2_long_sucess_rate_lst, m_SFC3_long_sucess_rate_lst,
                 p_service_time_lst, m_service_time_lst, p_long_service_time_lst, m_long_service_time_lst,
                 long_PDT_lst, long_MDT_lst, long_average_response_time_lst, long_average_migration_time_lst,
                 keep_mig_times_lst, long_keep_mig_times_lst, space_migrate_avg, air_migrate_avg, ground_migrate_avg, node_loading_avg,
                 p_SFC_energy_avg_lst,m_SFC_energy_avg_lst, m_mig_energy_avg_lst,
                 p_E2E_delay_avg_lst, m_E2E_delay_avg_lst, mix_E2E_delay_avg_lst,
                 E2E_delay_violate_avg_lst, PSD_violate_avg_lst, MSD_violate_avg_lst
                 ):
    if not os.path.exists(path):
        os.makedirs(path)
    ###############  Record reward data  ###############

    name = 'p_Reward_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(p_r_lst_DDPG)):
        f.write(str(p_r_lst_DDPG[i]))
        if i != len(p_r_lst_DDPG) - 1:
            f.write(',')

    f.close()

    name = 'm_Reward_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(m_r_lst_DDPG)):
        f.write(str(m_r_lst_DDPG[i]))
        if i != len(m_r_lst_DDPG) - 1:
            f.write(',')

    f.close()

    ###############  Record keep mig times graph ###############
   
    name = 'keep_mig_times_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(keep_mig_times_lst)):
        f.write(str(keep_mig_times_lst[i]))
        if i != len(keep_mig_times_lst) - 1:
            f.write(',')

    f.close()

    ###############  Record acceptence rate graph data  ###############

    name = 'p_Acceptence_rate_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(p_accept_lst_DDPG)):
        f.write(str(p_accept_lst_DDPG[i]))
        if i != len(p_accept_lst_DDPG) - 1:
            f.write(',')

    f.close()

    name = 'm_Acceptence_rate_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(m_accept_lst_DDPG)):
        f.write(str(m_accept_lst_DDPG[i]))
        if i != len(m_accept_lst_DDPG) - 1:
            f.write(',')

    f.close()

    name = 'total_Acceptence_rate_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(total_long_accept_lst_DDPG)):
        f.write(str(total_long_accept_lst_DDPG[i]))
        if i != len(total_long_accept_lst_DDPG) - 1:
            f.write(',')

    f.close()

    ###############  Record E2E delay graph data  ###############

    name = 'p_Average_E2E_delay_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(p_E2E_delay_lst_DDPG)):
        f.write(str(p_E2E_delay_lst_DDPG[i]))
        if i != len(p_E2E_delay_lst_DDPG) - 1:
            f.write(',')

    f.close()

    name = 'm_Average_E2E_delay_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(m_E2E_delay_lst_DDPG)):
        f.write(str(m_E2E_delay_lst_DDPG[i]))
        if i != len(m_E2E_delay_lst_DDPG) - 1:
            f.write(',')

    f.close()

    ###############  Record PDT and MDT graph data  ###############

    name = 'PDT_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(PDT_lst)):
        f.write(str(PDT_lst[i]))
        if i != len(PDT_lst) - 1:
            f.write(',')

    f.close()

    name = 'MDT_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(MDT_lst)):
        f.write(str(MDT_lst[i]))
        if i != len(MDT_lst) - 1:
            f.write(',')

    f.close()

    ###############  Record Average response time graph data  ###############
    name = 'Average_E2E_delay_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(average_response_time_lst)):
        f.write(str(average_response_time_lst[i]))
        if i != len(average_response_time_lst) - 1:
            f.write(',')

    f.close()

    ###############  Record Average migration time graph data  ###############
    name = 'Average_migration_time_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(average_migration_time_lst)):
        f.write(str(average_migration_time_lst[i]))
        if i != len(average_migration_time_lst) - 1:
            f.write(',')

    f.close()

    ###############  Record service time graph data  ###############

    name = 'p_Average_service_time_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(p_service_time_lst)):
        f.write(str(p_service_time_lst[i]))
        if i != len(p_service_time_lst) - 1:
            f.write(',')

    f.close()

    name = 'm_Average_service_time_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(m_service_time_lst)):
        f.write(str(m_service_time_lst[i]))
        if i != len(m_service_time_lst) - 1:
            f.write(',')

    f.close()

    ###############  Record migration time graph data  ###############

    name = 'm_Average_migration_time_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(m_mig_time_lst_DDPG)):
        f.write(str(m_mig_time_lst_DDPG[i]))
        if i != len(m_mig_time_lst_DDPG) - 1:
            f.write(',')

    f.close()

    ###############  Record max exist user graph data  ###############

    name = 'Average_max_exist_user_graph.txt'
    f = open(path + name, 'w')

    for i in range(len(max_exist_user_lst_DDPG)):
        f.write(str(max_exist_user_lst_DDPG[i]))
        if i != len(max_exist_user_lst_DDPG) - 1:
            f.write(',')

    f.close()

    ###############  Keep migration times graph ###############  

    name = "Keep_migration_times_graph"
    plt.figure()
    plt.plot(np.array(long_keep_mig_times_lst))
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_SFC_AVERAGE) + ' episodes')
    plt.ylabel('Average keep migration times')

    figure = plt.gcf()
    figure.savefig(path + name)

    ###############  p_all ###############

    name = "p_all"
    plt.figure()
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    
    # plt.subplot(221)
    # plt.plot(np.array(p_long_accept_lst_DDPG),color = 'orange',label = 'p_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episode')
    # plt.ylabel('p_Acceptence rate')

    plt.subplot(221)
    plt.plot(np.array(long_PDT_lst),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average PSD (ms)')    

    plt.subplot(222)
    plt.plot(np.array(mix_E2E_delay_avg_lst),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average traverse delay (ms)')

    plt.subplot(223)
    plt.plot(np.array(p_long_r_lst_DDPG),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('p_Average reward')
  
    plt.subplot(224)
    plt.plot( ((np.array(p_SFC_energy_avg_lst)+np.array(m_SFC_energy_avg_lst))/2 ),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average traverse energy consumption (J)')
    # plt.subplot(224)
    # plt.plot(np.array(delay_violate_avg_lst),color = 'orange',label = 'p_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    # plt.ylabel('delay violation rate')
    
    figure = plt.gcf()
    figure.savefig(path + name)
    

    ###############  m_all ###############

    name = "m_all"
    plt.figure()
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    # plt.subplot(221)
    # plt.plot(np.array(m_long_accept_lst_DDPG),color = 'orange',label = 'm_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episode')
    # plt.ylabel('m_Acceptence rate')
    
    plt.subplot(221)
    plt.plot(np.array(long_MDT_lst),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average MSD (ms)')

    plt.subplot(222)
    # plt.plot(np.array(long_average_migration_time_lst),color = 'orange',label = 'DDPG')
    plt.plot(np.array(m_long_mig_time_lst_DDPG),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average migration time (ms)')
    # plt.ylabel('Average migration downtime (MSD) (ms)')

    plt.subplot(223)
    plt.plot(np.array(m_long_r_lst_DDPG),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('m_Average reward')

    plt.subplot(224)
    plt.plot(np.array(m_mig_energy_avg_lst),color = 'orange',label = 'DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('Average migrate energy consumption (J)')
    
    figure = plt.gcf()
    figure.savefig(path + name)
    # plt.show()
    
    ###############  migrate ###############
    # name = "NS_migrate_percent_graph"
    # plt.figure()
    # plt.plot(np.array(space_migrate_avg), label='Space')
    # plt.plot(np.array(air_migrate_avg), label='Air')
    # plt.plot(np.array(ground_migrate_avg), label='Ground')
    # plt.legend()
    # plt.xlabel('Episodes')
    # plt.ylabel('NS migration counts percent')

    # figure = plt.gcf()
    # figure.savefig(path + name)
    
    ###############  node loading ###############
    name = "NS_loading_graph"
    plt.figure()
    plt.plot(np.array(node_loading_avg[0]), label='Space')
    plt.plot(np.array(node_loading_avg[1]), label='Air')
    plt.plot(np.array(node_loading_avg[2]), label='Ground')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('NS loading percent')

    figure = plt.gcf()
    figure.savefig(path + name)
    
    ###############  energy consumption ###############
    # name = "energy_all"
    # plt.figure()
    # plt.figure(figsize=(12, 10))
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    # plt.subplot(221)
    # plt.plot(np.array(p_SFC_energy_avg_lst),color = 'orange',label = 'p_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    # plt.ylabel('P_ddpg SFC energy consumption')

    # plt.subplot(222)
    # plt.plot(np.array(m_SFC_energy_avg_lst),color = 'orange',label = 'm_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    # plt.ylabel('M_ddpg SFC energy consumption')

    # plt.subplot(223)
    # plt.plot(np.array(m_mig_energy_avg_lst),color = 'orange',label = 'm_DDPG')
    # plt.legend()
    # plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    # plt.ylabel('M_ddpg migrate energy consumption')

    # figure = plt.gcf()
    # figure.savefig(path + name)
    
    ###############  violation rate ###############
    name = " violation_rate_all"
    plt.figure()
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    plt.subplot(221)
    plt.plot(np.array(E2E_delay_violate_avg_lst),color = 'orange',label = 'p_DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('traverse delay violation rate')

    plt.subplot(222)
    plt.plot(np.array(PSD_violate_avg_lst),color = 'orange',label = 'm_DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('PSD violation rate')

    plt.subplot(223)
    plt.plot(np.array(MSD_violate_avg_lst),color = 'orange',label = 'm_DDPG')
    plt.legend()
    plt.xlabel(str(TIMES_OF_DO_AVERAGE) + ' episodes')
    plt.ylabel('MSD violation rate')

    figure = plt.gcf()
    figure.savefig(path + name)
    
    ###############  traverse delay ###############
    name = "traverse_delay"
    plt.figure()
    plt.plot(np.array(p_E2E_delay_avg_lst), label='p_DDPG')
    plt.plot(np.array(m_E2E_delay_avg_lst), label='m_DDPG')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('traverse delay (ms)')
    figure = plt.gcf()
    figure.savefig(path + name)
    
###############################  DDPG  ####################################
                    
class DDPG2_placement(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor1_t'):
            self.a = self._build_a(self.S, scope='eval1', trainable=True)
            a_ = self._build_a(self.S_, scope='target1', trainable=False)
        with tf.variable_scope('Critic1_t'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval1', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target1', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1_t/eval1')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1_t/target1')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1_t/eval1')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1_t/target1')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(DDPG_LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q6
        self.atrain = tf.train.AdamOptimizer(DDPG_LR_A).minimize(a_loss, var_list=self.ae_params)    #(dQ/da)(da/d(w,b))

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s[0].tolist(), a[0].tolist(), [r], s_[0].tolist()))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 600, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            a = tf.layers.dense(l5, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable) #sigmoid: 0~1
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1],trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            q = tf.layers.dense(l5, 1,trainable=trainable)  # Q(s,a)
            return q

class DDPG2_migration(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor2_t'):
            self.a = self._build_a(self.S, scope='eval2', trainable=True)
            a_ = self._build_a(self.S_, scope='target2', trainable=False)
        with tf.variable_scope('Critic2_t'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval2', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target2', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2_t/eval2')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2_t/target2')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2_t/eval2')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2_t/target2')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(DDPG_LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q6
        self.atrain = tf.train.AdamOptimizer(DDPG_LR_A).minimize(a_loss, var_list=self.ae_params)    #(dQ/da)(da/d(w,b))

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s[0].tolist(), a[0].tolist(), [r], s_[0].tolist()))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 600, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            a = tf.layers.dense(l5, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable) #sigmoid: 0~1
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1],trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            q = tf.layers.dense(l5, 1,trainable=trainable)  # Q(s,a)
            return q

###############################  training  ####################################



#placement parameter
p_s_dim = env2.observation_space.shape[1]
print(p_s_dim)
p_a_dim = env2.a_dim
p_a_bound = env2.a_bound

#migratopm paremeter
m_s_dim = env2.observation_space.shape[1]
m_a_dim = env2.a_dim
m_a_bound = env2.a_bound

p_ddpg2 = DDPG2_placement(p_a_dim, p_s_dim, p_a_bound)
m_ddpg2 = DDPG2_migration(m_a_dim, m_s_dim, m_a_bound)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", p_ddpg2.sess.graph) #

###############################################

log_data_lst = []

p_r_lst_DDPG = []
m_r_lst_DDPG = []

p_accept_lst_DDPG = []
m_accept_lst_DDPG = []
total_accept_lst_DDPG = []

p_E2E_delay_lst_DDPG = []
m_E2E_delay_lst_DDPG = []

PDT_lst = []
MDT_lst = []

average_response_time_lst = []
average_migration_time_lst = []

m_mig_time_lst_DDPG = []

max_exist_user_lst_DDPG = []

p_SFC1_num_lst = []
p_SFC2_num_lst = []
p_SFC3_num_lst = []

p_SFC1_sucess_num_lst = []
p_SFC2_sucess_num_lst = []
p_SFC3_sucess_num_lst = []

p_SFC1_sucess_rate_lst = []
p_SFC2_sucess_rate_lst = []
p_SFC3_sucess_rate_lst = []

m_SFC1_num_lst = []
m_SFC2_num_lst = []
m_SFC3_num_lst = []

m_SFC1_sucess_num_lst = []
m_SFC2_sucess_num_lst = []
m_SFC3_sucess_num_lst = []

m_SFC1_sucess_rate_lst = []
m_SFC2_sucess_rate_lst = []
m_SFC3_sucess_rate_lst = []

###############################################

p_long_r_lst_DDPG = []
m_long_r_lst_DDPG = []

keep_mig_times_lst = []
long_keep_mig_times_lst = []

p_long_accept_lst_DDPG = []
m_long_accept_lst_DDPG = []
total_long_accept_lst_DDPG = []

p_long_E2E_delay_lst_DDPG = []
m_long_E2E_delay_lst_DDPG = []

long_PDT_lst = []
long_MDT_lst = []

long_average_response_time_lst = []
long_average_migration_time_lst = []

m_long_mig_time_lst_DDPG = []

p_SFC1_long_num_lst = []
p_SFC2_long_num_lst = []
p_SFC3_long_num_lst = []

p_SFC1_long_sucess_rate_lst = []
p_SFC2_long_sucess_rate_lst = []
p_SFC3_long_sucess_rate_lst = []

m_SFC1_long_num_lst = []
m_SFC2_long_num_lst = []
m_SFC3_long_num_lst = []

m_SFC1_long_sucess_rate_lst = []
m_SFC2_long_sucess_rate_lst = []
m_SFC3_long_sucess_rate_lst = []

p_service_time_lst = []
m_service_time_lst = []

p_long_service_time_lst = []
m_long_service_time_lst = []

long_max_exist_user_lst_DDPG = []

log_node_vnf = []

space_migrate = []
air_migrate = []
ground_migrate = []
need_migrate = []

space_migrate_avg = []
air_migrate_avg = []
ground_migrate_avg = []

node_loading = [ [], [], []]
node_loading_avg = [[], [], []]

p_SFC_energy_lst = []
m_SFC_energy_lst = []
m_mig_energy_lst = []
p_SFC_energy_avg_lst = []
m_SFC_energy_avg_lst = []
m_mig_energy_avg_lst = []

E2E_delay_violate_lst = []
PSD_violate_lst = []
MSD_violate_lst = []
E2E_delay_violate_avg_lst = []
PSD_violate_avg_lst = []
MSD_violate_avg_lst = []

p_E2E_delay_lst = []
m_E2E_delay_lst = []
mix_E2E_delay_lst = []
p_E2E_delay_avg_lst = []
m_E2E_delay_avg_lst = []
mix_E2E_delay_avg_lst = []

SFC_type_count_lst = []
SFC_type_count_avg_lst = []

env_E2E_lst = []
env_E2E_avg_lst = []
###############################################

var_p = 0.2  # control exploration
var_m = 0.2  # control exploration
for i in range(MAX_EPISODES):

    p_episode_reward = 0
    m_episode_reward = 0

    p_episode_E2E_delay = 0
    m_episode_E2E_delay = 0

    episode_PDT = 0
    episode_MDT = 0

    average_response_time = 0
    average_migration_time = 0
    average_migration_time_count = 0
    epsiode_average_migration_time = 0
    
    m_episode_mig_time = 0

    p_event_num = 0
    m_event_num = 0

    p_service_time = 0
    m_service_time = 0

    space_loading = 0
    air_loading = 0
    ground_loading = 0
    
    p_SFC_energy = 0
    m_SFC_energy = 0
    m_mig_energy = 0
    
    episode_avg_env_E2E = 0
    s = env2.reset()
    ep_start_time = time.time()
    for j in range(MAX_EP_STEPS):
        # print(f"EP:{i}, step:{j}")
        if math.isnan(s[0][0]):
            os._exit()
        # Add exploration noise
        # placement
        if(int(env2.event[3])==-1):
            for k in range(env2.get_SFC_len()):
                env2.current_SFC_index = k
                a = p_ddpg2.choose_action(s)
                a = np.clip(np.random.normal(a, var_p), 1e-2,
                            1 - 1e-2)  # add randomness to action selection for exploration
                # a = np.random.random((1, env2.a_dim))
                s_, r,already_satisfy = env2.step(a)

                if env2.no_resource == True:
                    s = s_
                    break
                # update p_ddpg
                p_ddpg2.store_transition(s, a, r * 10, s_)
                if p_ddpg2.pointer > MEMORY_CAPACITY:
                    var_p *= .9995  # decay the action randomness
                    p_ddpg2.learn()
                s = s_
                p_episode_reward += r
                if(already_satisfy == True):
                    break

            ##########   Used for output   ########## 
            p_event_num += 1

            #p_episode_reward += r

            p_episode_E2E_delay += env2.total_E2E_delay

            p_service_time += env2.service_time

            episode_PDT += env2.Placement_Down_Time
            
            p_SFC_energy += env2.p_SFC_energy
            ##########   Used for output   ########## 
             
        # migration
        else:
            for k in range(env2.get_SFC_len()):
                env2.current_SFC_index = k
                a = m_ddpg2.choose_action(s)
                a = np.clip(np.random.normal(a, var_m), 1e-2, 1 - 1e-2)
                if RUN_WoMA:
                    s_, r,already_satisfy = env2.dont_migrate_step() #DRL-EMSES-WoMA
                else:
                    s_, r,already_satisfy = env2.step(a) #DRL-EMSES
                if env2.no_resource == True:
                    s = s_
                    break
                # s = s_
                # m_episode_reward += r
                if(not already_satisfy):
                #     break
                # update m_ddpg
                    m_ddpg2.store_transition(s, a, r * 10, s_)
                if m_ddpg2.pointer > MEMORY_CAPACITY:
                    var_m *= .9995  # decay the action randomness
                    m_ddpg2.learn()
                s = s_
                m_episode_reward += r
                if(already_satisfy == True):
                    break

            ##########   Used for output   ##########
            m_event_num += 1

            #m_episode_reward += r
            
            m_episode_E2E_delay += env2.total_E2E_delay 

            m_episode_mig_time += env2.total_mig_time

            m_service_time += env2.service_time

            episode_MDT += env2.Migration_Down_Time

            m_SFC_energy += env2.m_SFC_energy
            m_mig_energy += env2.m_mig_energy
            ##########   Used for output   ##########

        episode_avg_env_E2E += env2.avg_env_E2E
        
        if (i == 5 and env2.log_data_lst != []):
            log_data_lst.append(env2.log_data_lst)

        average_response_time += env2.response_time
        if(env2.average_migration_time != 0):
            average_migration_time_count += 1
            average_migration_time += env2.average_migration_time

        space_loading += (1-env2.Node_storage[0]/env2.Max_Node_storage[0]) *100
        air_loading += (1 - np.average(env2.Node_storage[env2.node_s:env2.node_s + env2.node_a]) / env2.Max_Node_storage[env2.node_s+1]) *100
        ground_loading += (1 - np.average(env2.Node_storage[env2.node_s + env2.node_a: ]) / env2.Max_Node_storage[ -1 ]) *100
        
        if j == MAX_EP_STEPS - 1:
            log_node_vnf.append(env2.log_node_vnf)
            if env2.times_of_mig == 0:
                space_need_migrate_precent = 0
                air_need_migrate_precent = 0
                ground_need_migrate_precent = 0
            else:
                space_need_migrate_precent = round(env2.space_migrate/env2.times_of_mig*100, 2)
                air_need_migrate_precent = round(env2.air_migrate/env2.times_of_mig*100, 2)
                ground_need_migrate_precent = round( (env2.times_of_mig - env2.space_migrate -env2.air_migrate)/env2.times_of_mig*100, 2)
            
            # if env2.no_need_migration == 0:
            #     space_no_migrate_precent = 0
            #     air_no_migrate_precent = 0
            # else:
            #     space_no_migrate_precent = round(env2.space_no_migrate/env2.no_need_migration*100, 2)
            #     air_no_migrate_precent = round(env2.air_no_migrate/env2.no_need_migration*100, 2)
            
            space_migrate.append(space_need_migrate_precent)
            air_migrate.append(air_need_migrate_precent)
            ground_migrate.append(ground_need_migrate_precent)
            

            node_loading[0].append(space_loading/MAX_EP_STEPS)
            node_loading[1].append(air_loading/MAX_EP_STEPS)
            node_loading[2].append(ground_loading/MAX_EP_STEPS)
            
            # space_no_migrate.append(space_no_migrate_precent)
            # air_no_migrate.append(air_no_migrate_precent)
            # no_need_migrate.append(env2.no_need_migration)
            
            if(average_migration_time_count == 0):
                epsiode_average_migration_time = 0
            else:
                epsiode_average_migration_time = average_migration_time / average_migration_time_count

            #print(episode_reward)
            if (i == 0):
                first_p_episode_reward = p_episode_reward
                first_m_episode_reward = m_episode_reward

            p_episode_failed = env2.p_episode_fail
            m_episode_failed = env2.m_episode_fail
            total_episode_failed = p_episode_failed + m_episode_failed
            
            p_SFC1_sucess_rate = 0
            p_SFC2_sucess_rate = 0
            p_SFC3_sucess_rate = 0
            m_SFC1_sucess_rate = 0
            m_SFC2_sucess_rate = 0
            m_SFC3_sucess_rate = 0

            if(env2.placement_event_SFC[0] == 0):
                p_SFC1_sucess_rate = 0
            else:
                p_SFC1_sucess_rate = env2.placement_event_SFC_sucess[0] / env2.placement_event_SFC[0]

            if(env2.placement_event_SFC[1] == 0):
                p_SFC2_sucess_rate = 0
            else:
                p_SFC2_sucess_rate = env2.placement_event_SFC_sucess[1] / env2.placement_event_SFC[1]

            if(env2.placement_event_SFC[2] == 0):
                p_SFC3_sucess_rate = 0
            else:
                p_SFC3_sucess_rate = env2.placement_event_SFC_sucess[2] / env2.placement_event_SFC[2]


            if(env2.migration_event_SFC[0] == 0):
                m_SFC1_sucess_rate = 0
            else:
                m_SFC1_sucess_rate = env2.migration_event_SFC_sucess[0] / env2.migration_event_SFC[0]

            if(env2.migration_event_SFC[1] == 0):
                m_SFC2_sucess_rate = 0
            else:
                m_SFC2_sucess_rate = env2.migration_event_SFC_sucess[1] / env2.migration_event_SFC[1]

            if(env2.migration_event_SFC[2] == 0):
                m_SFC3_sucess_rate = 0
            else:
                m_SFC3_sucess_rate = env2.migration_event_SFC_sucess[2] / env2.migration_event_SFC[2]

            # if (i == 5):
            #     # Output information
            #     name = 'C:/Users/user/Desktop/5G_contest/test/information.pickle'
            #     with open(name, 'wb') as f:
            #         #for i in range(len(log_data_lst)):
            #         pickle.dump(log_data_lst, f)
            #     f.close()
            #     print("success")
            #     print(log_data_lst)

            #     exit(1)

            print("----------------------------------------------------")
            print(f"Episode = {i}/{MAX_EPISODES} 花費時間:{round((time.time()-ep_start_time), 3)}秒")
            print("----------------------------------------------------")
            # print("Placement Episode reward = ", p_episode_reward)
            # print("Migration Episode reward = ", m_episode_reward)
            # print("----------------------------------------------------")
            print("Placement Episode failed = ", p_episode_failed)
            print("Migration Episode failed = ", m_episode_failed)
            print("----------------------------------------------------")
            print("Placement Episode success =", p_event_num - p_episode_failed)
            print("Migration Episode success =", m_event_num - m_episode_failed)
            print("----------------------------------------------------")
            print("Placement episode progress rate from first episode = ", round(-(first_p_episode_reward - p_episode_reward)/first_p_episode_reward*100, 3), "%")
            print("Migration episode progress rate from first episode = ", round(-(first_m_episode_reward - m_episode_reward)/first_m_episode_reward*100, 3), "%")
            print("----------------------------------------------------")
            print("Max exist user = ",env2.max_exist_user)
            print("----------------------------------------------------")
            print("p_E2E_delay = ", p_episode_E2E_delay / (p_event_num))
            print("m_E2E_delay = ", m_episode_E2E_delay / (m_event_num))
            print("----------------------------------------------------")
            print("PDT = ", episode_PDT / (p_event_num - p_episode_failed))
            print("MDT = ", episode_MDT / (m_event_num - m_episode_failed))
            print("----------------------------------------------------")
            print("Average response time  = ", average_response_time / MAX_EP_STEPS)
            print("Average migration time = ", epsiode_average_migration_time)
            print("----------------------------------------------------")
            # print("p_service_time = ", p_service_time / (p_event_num - p_episode_failed))
            # print("m_service_time = ", m_service_time / (m_event_num - m_episode_failed))
            print("No need migration number = ", env2.no_need_migration)
            # print(f"Space no need migration% = {space_no_migrate_precent}%")
            # print(f"Air no need migration% = {air_no_migrate_precent}%")
            # print("----------------------------------------------------")
            # print("Need migration number = ", env2.times_of_mig)
            # print(f"Need migration% Space={space_need_migrate_precent}%, Air={air_need_migrate_precent}%, Ground={ground_need_migrate_precent}%")
            # print("----------------------------------------------------")
            # print("Node_vnf = ", env2.log_node_vnf)
            # print(f"Node Loading  Space={space_loading/MAX_EP_STEPS:.2f}%, Air={air_loading/MAX_EP_STEPS:.2f}%, Ground={ground_loading/MAX_EP_STEPS:.2f}%")
            # print(f"p_SFC_energy:{p_SFC_energy/p_event_num}")
            # print(f"m_SFC_energy:{m_SFC_energy/m_event_num}")
            print(f"SFC_energy:{((p_SFC_energy/p_event_num)+(m_SFC_energy/m_event_num))/2}, m_mig_energy:{m_mig_energy/m_event_num}")
            print(f"E2E_delay violation rate = {env2.E2E_delay_violate/(p_event_num+m_event_num)}, PSD violation rate = {env2.PSD_violate/(p_event_num)}, MSD violation rate = {env2.MSD_violate/(m_event_num)}")
            print(f"SFC_type_count, Space:{env2.SFC_type_count[0]}, Air:{env2.SFC_type_count[1]}, Ground:{env2.SFC_type_count[2]}")
            print(f"一個episode有{env2.bigger_E2E_delay}個event違反TD conds, {env2.data_rate_violate}個event違反data_rate需求, {env2.affects_other_SFC}個event造成其他SFC違反TD限制")
            print(f"一個episode的平均E2E delay:{episode_avg_env_E2E/MAX_EP_STEPS}")
            
            # print("Wrong effect by new SFC = ", env2.check_result_SFC_false)
            # print("   Data rate failed = ", env2.check_result_SFC_dr_false)
            # print("   E2E_delay failed = ", env2.check_result_SFC_E2E_false)
            # print("   Migration failed = ", env2.check_result_SFC_mig_false)
            # print("----------------------------------------------------")
            # print("Wrong effect by new mig = ", env2.check_result_MIG_false)
            # print("   Data rate failed = ", env2.check_result_MIG_dr_false)
            # print("   E2E_delay failed = ", env2.check_result_MIG_E2E_false)
            # print("   Migration failed = ", env2.check_result_MIG_mig_false)
            # print("      Event      |    Placement    |    Migration   ")
            # print("----------------------------------------------------")
            # print("Num of SFC1      |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC[0], env2.migration_event_SFC[0]))
            # print("Num of SFC2      |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC[1], env2.migration_event_SFC[1]))
            # print("Num of SFC3      |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC[2], env2.migration_event_SFC[2]))
            # print("----------------------------------------------------")
            # print("SFC1 succ        |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC_sucess[0], env2.migration_event_SFC_sucess[0]))
            # print("SFC2 succ        |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC_sucess[1], env2.migration_event_SFC_sucess[1]))
            # print("SFC3 succ        |       {0:^3}       |       {1:^3}".format(env2.placement_event_SFC_sucess[2], env2.migration_event_SFC_sucess[2]))
            # print("----------------------------------------------------")
            # print("SFC1 succ rate   |      {0:^5}      |      {1:^5}".format(round(p_SFC1_sucess_rate, 3), round(m_SFC1_sucess_rate, 3)))
            # print("SFC2 succ rate   |      {0:^5}      |      {1:^5}".format(round(p_SFC2_sucess_rate, 3), round(m_SFC2_sucess_rate, 3)))
            # print("SFC3 succ rate   |      {0:^5}      |      {1:^5}".format(round(p_SFC3_sucess_rate, 3), round(m_SFC3_sucess_rate, 3)))
            

            # print(-episode_reward / env2.episode_complete)
            # r_lst_DDPG.append(np.clip(-episode_reward / env2.episode_complete,0,0.1))

            # Reward list
            p_r_lst_DDPG.append(p_episode_reward / p_event_num)
            m_r_lst_DDPG.append(m_episode_reward / m_event_num)
            # r_lst_DDPG.append((p_episode_reward + m_episode_reward) / env2.episode_complete)

            # Acceptance ratio list
            # print("p_e  :", p_event_num)
            # print("p_e_f:", p_episode_failed)
            p_accept_lst_DDPG.append((p_event_num - p_episode_failed) / p_event_num)
            m_accept_lst_DDPG.append((m_event_num - m_episode_failed) / m_event_num)
            total_accept_lst_DDPG.append((MAX_EP_STEPS - total_episode_failed) / MAX_EP_STEPS)
            # accept_lst_DDPG.append((env2.episode_complete - p_episode_failed - m_episode_failed) / env2.episode_complete)

            # E2E delay list
            p_E2E_delay_lst_DDPG.append(p_episode_E2E_delay / (p_event_num - p_episode_failed))
            m_E2E_delay_lst_DDPG.append(m_episode_E2E_delay / (m_event_num - m_episode_failed))
            # E2E_delay_lst_DDPG.append((p_episode_E2E_delay + m_episode_E2E_delay) / env2.episode_complete)

            # PDT lst
            PDT_lst.append(episode_PDT / (p_event_num - p_episode_failed))

            # MDT lst
            # MDT_lst.append(episode_MDT * 1000 / (m_event_num - m_episode_failed))
            MDT_lst.append(episode_MDT / (m_event_num - m_episode_failed))

            # Average response time lst
            average_response_time_lst.append(average_response_time / MAX_EP_STEPS)

            # Average migration time lst
            # average_migration_time_lst.append(epsiode_average_migration_time * 1000)
            average_migration_time_lst.append(epsiode_average_migration_time )

            # Service Time lst
            p_service_time_lst.append(p_service_time / (p_event_num - p_episode_failed))
            m_service_time_lst.append(m_service_time / (m_event_num - m_episode_failed))

            # Placement 3 SFC type num
            p_SFC1_num_lst.append(env2.placement_event_SFC[0])
            p_SFC2_num_lst.append(env2.placement_event_SFC[1])
            p_SFC3_num_lst.append(env2.placement_event_SFC[2])

            # Migration 3 SFC type num
            m_SFC1_num_lst.append(env2.migration_event_SFC[0])
            m_SFC2_num_lst.append(env2.migration_event_SFC[1])
            m_SFC3_num_lst.append(env2.migration_event_SFC[2])

            # Placement 3 SFC type accept rate
            p_SFC1_sucess_rate_lst.append(p_SFC1_sucess_rate)
            p_SFC2_sucess_rate_lst.append(p_SFC2_sucess_rate)
            p_SFC3_sucess_rate_lst.append(p_SFC3_sucess_rate)

            # Migration 3 SFC type accpet rate
            m_SFC1_sucess_rate_lst.append(m_SFC1_sucess_rate)
            m_SFC2_sucess_rate_lst.append(m_SFC2_sucess_rate)
            m_SFC3_sucess_rate_lst.append(m_SFC3_sucess_rate)

            env_E2E_lst.append(episode_avg_env_E2E/MAX_EP_STEPS)
            
            # Migration time list
            if (env2.times_of_mig != 0):
                m_mig_time_lst_DDPG.append(m_episode_mig_time / env2.times_of_mig)
            else:
                m_mig_time_lst_DDPG.append(0)
            # Max exist user list
            max_exist_user_lst_DDPG.append(env2.max_exist_user)

            # Count average mig times
            tmp_avg_mig = 0
            tmp_avg_mig_count = 0
            for user in range(env2.MAX_USER):
                if(env2.max_mig_times_lst[user] != -1):
                    tmp_avg_mig += env2.max_mig_times_lst[user]
                    tmp_avg_mig_count += 1
                
            tmp_avg_mig /= tmp_avg_mig_count
            keep_mig_times_lst.append(tmp_avg_mig)

            # energy
            p_SFC_energy_lst.append(p_SFC_energy/p_event_num)
            m_SFC_energy_lst.append(m_SFC_energy/m_event_num)
            m_mig_energy_lst.append(m_mig_energy/m_event_num)
            
            E2E_delay_violate_lst.append(env2.E2E_delay_violate/(p_event_num+m_event_num))
            PSD_violate_lst.append(env2.PSD_violate/(p_event_num))
            MSD_violate_lst.append(env2.MSD_violate/(m_event_num))
            
            p_E2E_delay_lst.append(p_episode_E2E_delay / (p_event_num))
            m_E2E_delay_lst.append(m_episode_E2E_delay / (m_event_num))
            mix_E2E_delay_lst.append((p_episode_E2E_delay+m_episode_E2E_delay) / (p_event_num+m_event_num))
            
            SFC_type_count_lst.append(env2.SFC_type_count)
            
            if i % TIMES_OF_DO_SFC_AVERAGE == ( TIMES_OF_DO_SFC_AVERAGE - 1):

                p_SFC1_long_num_lst.append(sum(p_SFC1_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                p_SFC2_long_num_lst.append(sum(p_SFC2_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                p_SFC3_long_num_lst.append(sum(p_SFC3_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)

                m_SFC1_long_num_lst.append(sum(m_SFC1_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                m_SFC2_long_num_lst.append(sum(m_SFC2_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                m_SFC3_long_num_lst.append(sum(m_SFC3_num_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)              

                p_SFC1_long_sucess_rate_lst.append(sum(p_SFC1_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                p_SFC2_long_sucess_rate_lst.append(sum(p_SFC2_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                p_SFC3_long_sucess_rate_lst.append(sum(p_SFC3_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)

                m_SFC1_long_sucess_rate_lst.append(sum(m_SFC1_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                m_SFC2_long_sucess_rate_lst.append(sum(m_SFC2_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                m_SFC3_long_sucess_rate_lst.append(sum(m_SFC3_sucess_rate_lst[i-(TIMES_OF_DO_SFC_AVERAGE - 1):i+1]) / TIMES_OF_DO_SFC_AVERAGE)
                

            if i % TIMES_OF_DO_AVERAGE == (TIMES_OF_DO_AVERAGE - 1):
                
                # Keep mig times list
                long_keep_mig_times_lst.append(sum(keep_mig_times_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Reward list
                p_long_r_lst_DDPG.append(sum(p_r_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_long_r_lst_DDPG.append(sum(m_r_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                # long_r_lst_DDPG.append(sum(r_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Acceptance ratio list
                p_long_accept_lst_DDPG.append(sum(p_accept_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_long_accept_lst_DDPG.append(sum(m_accept_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                total_long_accept_lst_DDPG.append(sum(total_accept_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                # long_accept_lst_DDPG.append(sum(accept_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # E2E delay list
                p_long_E2E_delay_lst_DDPG.append(sum(p_E2E_delay_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_long_E2E_delay_lst_DDPG.append(sum(m_E2E_delay_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                # long_E2E_delay_lst_DDPG.append(sum(E2E_delay_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # PDT list
                long_PDT_lst.append(sum(PDT_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # MDT list
                long_MDT_lst.append(sum(MDT_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Average response time lst
                long_average_response_time_lst.append(sum(average_response_time_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Average migration time lst
                long_average_migration_time_lst.append(sum(average_migration_time_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Service time lst
                p_long_service_time_lst.append(sum(p_service_time_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_long_service_time_lst.append(sum(m_service_time_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                # Migration time list
                m_long_mig_time_lst_DDPG.append(sum(m_mig_time_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                # Max exist user list
                long_max_exist_user_lst_DDPG.append(sum(max_exist_user_lst_DDPG[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                space_migrate_avg.append(sum(space_migrate[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                air_migrate_avg.append(sum(air_migrate[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                ground_migrate_avg.append(sum(ground_migrate[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                node_loading_avg[0].append(sum(node_loading[0][i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                node_loading_avg[1].append(sum(node_loading[1][i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                node_loading_avg[2].append(sum(node_loading[2][i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                # energy
                p_SFC_energy_avg_lst.append(sum(p_SFC_energy_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_SFC_energy_avg_lst.append(sum(m_SFC_energy_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_mig_energy_avg_lst.append(sum(m_mig_energy_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                E2E_delay_violate_avg_lst.append(sum(E2E_delay_violate_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                PSD_violate_avg_lst.append(sum(PSD_violate_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                MSD_violate_avg_lst.append(sum(MSD_violate_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)

                p_E2E_delay_avg_lst.append(sum(p_E2E_delay_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                m_E2E_delay_avg_lst.append(sum(m_E2E_delay_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                mix_E2E_delay_avg_lst.append(sum(mix_E2E_delay_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
                env_E2E_avg_lst.append(sum(env_E2E_lst[i-(TIMES_OF_DO_AVERAGE - 1):i+1]) / TIMES_OF_DO_AVERAGE)
                
    if(i % TIMES_OF_DO_OUTPUT == (TIMES_OF_DO_OUTPUT - 1)):
        draw_picture(p_r_lst_DDPG, p_accept_lst_DDPG, p_E2E_delay_lst_DDPG, 
                     p_long_r_lst_DDPG, p_long_accept_lst_DDPG, p_long_E2E_delay_lst_DDPG,
                     m_r_lst_DDPG, m_accept_lst_DDPG, m_E2E_delay_lst_DDPG, m_mig_time_lst_DDPG, 
                     m_long_r_lst_DDPG, m_long_accept_lst_DDPG, total_long_accept_lst_DDPG, m_long_E2E_delay_lst_DDPG, m_long_mig_time_lst_DDPG,
                     max_exist_user_lst_DDPG, long_max_exist_user_lst_DDPG,
                     p_SFC1_long_num_lst, p_SFC2_long_num_lst, p_SFC3_long_num_lst,
                     m_SFC1_long_num_lst, m_SFC2_long_num_lst, m_SFC3_long_num_lst,
                     p_SFC1_long_sucess_rate_lst, p_SFC2_long_sucess_rate_lst, p_SFC3_long_sucess_rate_lst,
                     m_SFC1_long_sucess_rate_lst, m_SFC2_long_sucess_rate_lst, m_SFC3_long_sucess_rate_lst,
                     p_service_time_lst, m_service_time_lst, p_long_service_time_lst, m_long_service_time_lst,
                     long_PDT_lst, long_MDT_lst, long_average_response_time_lst, long_average_migration_time_lst,
                     keep_mig_times_lst, long_keep_mig_times_lst, space_migrate_avg, air_migrate_avg, ground_migrate_avg, node_loading_avg, 
                     p_SFC_energy_avg_lst, m_SFC_energy_avg_lst, m_mig_energy_avg_lst, 
                     p_E2E_delay_avg_lst, m_E2E_delay_avg_lst, mix_E2E_delay_avg_lst,
                     E2E_delay_violate_avg_lst, PSD_violate_avg_lst, MSD_violate_avg_lst
                     )
        np.save(save_dir+"long_PDT_lst.npy",long_PDT_lst,allow_pickle=True)
        np.save(save_dir+"long_average_response_time_lst.npy", long_average_response_time_lst, allow_pickle=True)
        np.save(save_dir+"p_long_r_lst.npy", p_long_r_lst_DDPG, allow_pickle=True)
        np.save(save_dir+"total_long_accept_lst.npy", total_long_accept_lst_DDPG, allow_pickle=True)

        np.save(save_dir+"long_MDT_lst.npy",long_MDT_lst,allow_pickle=True)
        np.save(save_dir+"long_average_migration_time_lst.npy", long_average_migration_time_lst, allow_pickle=True)
        np.save(save_dir+"m_long_r_lst.npy", m_long_r_lst_DDPG, allow_pickle=True)
        np.save(save_dir+"m_long_accept_lst.npy", m_long_accept_lst_DDPG, allow_pickle=True)
        np.save(save_dir+"log_node_vnf.npy", log_node_vnf, allow_pickle=True)
        np.save(save_dir+"space_migrate_avg.npy", space_migrate_avg, allow_pickle=True)
        np.save(save_dir+"air_migrate_avg.npy", air_migrate_avg, allow_pickle=True)
        np.save(save_dir+"ground_migrate_avg.npy", ground_migrate_avg, allow_pickle=True)
        np.save(save_dir+"node_loading_avg.npy", node_loading_avg, allow_pickle=True)
        np.save(save_dir+"p_SFC_energy_avg_lst.npy", p_SFC_energy_avg_lst, allow_pickle=True)
        np.save(save_dir+"m_SFC_energy_avg_lst.npy", m_SFC_energy_avg_lst, allow_pickle=True)
        np.save(save_dir+"m_mig_energy_avg_lst.npy", m_mig_energy_avg_lst, allow_pickle=True)
        np.save(save_dir+"E2E_delay_violate_avg_lst.npy", E2E_delay_violate_avg_lst, allow_pickle=True)
        np.save(save_dir+"PSD_violate_avg_lst.npy", PSD_violate_avg_lst, allow_pickle=True)
        np.save(save_dir+"MSD_violate_avg_lst.npy", MSD_violate_avg_lst, allow_pickle=True) 
        np.save(save_dir+"p_E2E_delay_avg_lst.npy", p_E2E_delay_avg_lst, allow_pickle=True)
        np.save(save_dir+"m_E2E_delay_avg_lst.npy", m_E2E_delay_avg_lst, allow_pickle=True)
        np.save(save_dir+"mix_E2E_delay_avg_lst.npy", mix_E2E_delay_avg_lst, allow_pickle=True)
        np.save(save_dir+"SFC_type_count_lst.npy", SFC_type_count_lst, allow_pickle=True)
        np.save(save_dir+"m_long_mig_time_lst_DDPG.npy", m_long_mig_time_lst_DDPG, allow_pickle=True)
        np.save(save_dir+"env_E2E_avg_lst.npy",env_E2E_avg_lst, allow_pickle=True)
        
        np.save(save_dir+"p_SFC_energy_lst.npy",p_SFC_energy_lst, allow_pickle=True)
        np.save(save_dir+"m_SFC_energy_lst.npy",m_SFC_energy_lst, allow_pickle=True)
        np.save(save_dir+"m_mig_energy_lst.npy",m_mig_energy_lst, allow_pickle=True)
        np.save(save_dir+"mix_E2E_delay_lst.npy",mix_E2E_delay_lst, allow_pickle=True)
        np.save(save_dir+"env_E2E_lst.npy",env_E2E_lst, allow_pickle=True)
        
        np.save(save_dir+"PDT_lst.npy",PDT_lst,allow_pickle=True)
        np.save(save_dir+"MDT_lst.npy",MDT_lst,allow_pickle=True)
        
        np.save(save_dir+"E2E_delay_violate_lst.npy",E2E_delay_violate_lst, allow_pickle=True)
        np.save(save_dir+"PSD_violate_lst.npy",PSD_violate_lst, allow_pickle=True)
        np.save(save_dir+"MSD_violate_lst.npy",MSD_violate_lst, allow_pickle=True)

        
print("###########   Training down   ###########")
end = time.time()
print("cost time",end - start)
print("MAX User = ",env2.MAX_USER)
print("SFC length = ",env2.get_SFC_len())
print("Episode = ", MAX_EPISODES)
print("Step = ", MAX_EP_STEPS)
print("Redeploy done")
