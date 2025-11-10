# Start redis server locally using : redis-server `./redis.conf` (run from redis directory).

import redis
import time
import os
import os.path
import numpy as np
import pickle
import itertools
import random

INIT, WAIT, MOVE, HOLD, SUCCESS, FAILURE = 0, 1, 2, 3, 4, 5

def N_Choose_K(n,k):
    return np.prod(range(k+1,n+1))/factorial(k)

def factorial(n):
    return np.prod(range(1,n+1))

class ExhaustiveTask():
    
    def __init__(self):
        self.redisServer = redis.StrictRedis(host='localhost', port=6379, db=0)

        #file_name = '/home/nptl/code/tasks/paramScripts/gestureParamScripts/t5/fingers_flex_extend_py2.pkl'
        #file_name = '/home/nptl/code/tasks/paramScripts/gestureParamScripts/t5/fingers_flex_extend_new_py2.pkl'
        file_name = '/home/nptl/code/tasks/paramScripts/gestureParamScripts/t5/fingers_flex_extend3.pkl'
        dat = pickle.load(open(file_name, 'rb'))

        # extract keys
        self.keys = dat['keys_use']
        self.finger_directions = dat['finger_directions']

        # Set key order for redis
        # get left and right hand key indices

        if thumb_dim == 1:
            self.fingers_list = ['thumb', 'index', 'middle', 'ring', 'pinky']
            self.finger_dict = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}
        elif thumb_dim == 2:
            self.fingers_list = ['thumb', 'thumb2', 'index', 'middle', 'ring', 'pinky']
            self.finger_dict = {'thumb': 1, 'thumb2': 2, 'index': 3, 'middle': 4, 'ring': 5, 'pinky': 6}

        # For unity
        self.redisServer.set('estimated/RightHand/keys', '#'.join(self.keys + ['trial_phase'] + self.fingers_list))

        self.assign_finger = np.zeros((len(self.keys)))
        for iikey, ikey in enumerate(self.keys):
            for ifinger in self.finger_dict.keys():
                if ifinger in ikey:
                    self.assign_finger[iikey] = self.finger_dict[ifinger]

    def generate_hand_pos(self, wts):

        values = np.copy(self.finger_directions['idle'])
        for finger in self.finger_dict.keys():
            finger_wt = wts[finger]
            val = np.where(self.assign_finger == self.finger_dict[finger])[0]
            # values[val] = finger_wt * self.finger_directions['%s:flex' % finger][val] + (1 - finger_wt) * self.finger_directions['%s:extend' % finger][val]
            values[val] = finger_wt * self.finger_directions['%s:flex' % finger][val] + (1 - finger_wt) * self.finger_directions['idle'][val]
        return values

    def update(self, estimate_render):
	    
        # scale up thumb because the decoder has corrected for it.
        estimate_render[0] = (estimate_render[0] - 0.5) * (0.7/0.5) + 0.5
        if len(estimate_render) < 5:
            estimate_render = estimate_render + [estimate_render[-1]]
			
        wts_est = dict(zip(self.fingers_list, estimate_render))
        values_est_fix = self.generate_hand_pos(wts_est)
        trial_phase = 1
        self.redisServer.set('estimated/RightHand/values', '#'.join([str(values_est_fix[ikey]) for ikey in range(len(self.keys))] + [str(trial_phase)] + [str(wts_est[ifinger]) for ifinger in self.fingers_list]))
	
class TargetsBimanual():
    def __init__(self, file_name,
                 n_finger_groups=4,
                 finger_scales=np.array([1, 1, 1, 1, 1]),
                 levels=[-1, 0, 1],
                 wts_default={'thumb': 0.0, 'index': 0.0, 'middle': 0.0, 'ring': 0.0, 'pinky': 0.0}):
        self.target_obj = {}
        self.target_obj.update({'Right': Targets(file_name=file_name['Right'],
                                               n_finger_groups=n_finger_groups,
                                               finger_scales=finger_scales,
                                               levels=levels,
                                               wts_default=wts_default,
                                               hand_identity='RightHand')})

        self.target_obj.update({'Left': Targets(file_name=file_name['Left'],
                                               n_finger_groups=n_finger_groups,
                                               finger_scales=finger_scales,
                                               levels=levels,
                                               wts_default=wts_default,
                                               hand_identity='LeftHand')})

        self.fingers_list = []
        for hand in ['Right', 'Left']:
            for ifinger in self.target_obj[hand].fingers_list:
                self.fingers_list += [f'{hand}:{ifinger}']

        wts_list_right = self.target_obj['Right'].wts_list
        wts_list_left = self.target_obj['Left'].wts_list
        wts_list = []
        for wt in wts_list_right:
            wts_list += [wt + [0.5, 0.5, 0.5, 0.5, 0.5]]
        for wt in wts_list_left:
            wts_list += [[0.5, 0.5, 0.5, 0.5, 0.5] + wt]
        self.wts_list = wts_list

        wts_default = {}
        wts_default_right = self.target_obj['Right'].wts_default
        wts_default_left = self.target_obj['Left'].wts_default
        for w in wts_default_right.keys():
            wts_default.update({f'Right:{w}': wts_default_right[w]})
        for w in wts_default_left.keys():
            wts_default.update({f'Left:{w}': wts_default_left[w]})
        self.wts_default = wts_default

        self.finger_scales = np.concatenate([finger_scales, finger_scales], axis=0)
        print(self.finger_scales)


    def write_skeleton(self, redisServer, lowdim_pos, string, char):

        if np.argmax(np.abs(np.array(lowdim_pos) - 0.5)) < 5:
            target_obj_use = self.target_obj['Right']
            lowdim_pos_use = lowdim_pos[:5]
        else:
            target_obj_use = self.target_obj['Left']
            lowdim_pos_use = lowdim_pos[5:]

        wts_target = dict(zip(target_obj_use.fingers_list, lowdim_pos_use))
        values_target_fix = target_obj_use.generate_hand_pos(wts_target)
        redisServer.set(f'{string}/%s/keys' % (char), '#'.join([ikey for ikey in target_obj_use.keys]))
        redisServer.set(f'{string}/%s/values' % (char), '#'.join([str(ival) for ival in values_target_fix]))

    def update(self, estimate_render):
        self.target_obj['Right'].update(estimate_render[:5])
        self.target_obj['Left'].update(estimate_render[5:])

    def generate_hand_pos(self, wts):
        wts_right = {'thumb': wts['Right:thumb'],
                     'index': wts['Right:index'],
                     'middle': wts['Right:middle'],
                     'ring': wts['Right:ring'],
                     'pinky': wts['Right:pinky']}

        wts_left = {'thumb': wts['Left:thumb'],
                     'index': wts['Left:index'],
                     'middle': wts['Left:middle'],
                     'ring': wts['Left:ring'],
                     'pinky': wts['Left:pinky']}

        return (self.target_obj['Right'].generate_hand_pos(wts_right),
                self.target_obj['Left'].generate_hand_pos(wts_left))

class Targets():
    '''Get the set of targets for different variants of the task.'''
    
    def __init__(self, file_name= '/home/nptl/code/tasks/paramScripts/gestureParamScripts/t5/fingers_flex_extend3.pkl', 
                       n_finger_groups=4, 
                       finger_scales=np.array([1, 1, 1, 1, 1]), 
                       levels=[-1, 0, 1], 
                       wts_default={'thumb': 0.0, 'index': 0.0, 'middle': 0.0, 'ring': 0.0, 'pinky': 0.0}, 
                       hand_identity='RightHand',
                       thumb_dim=1,
                       n_trials = False):

            if thumb_dim == 2:
                
                # option 1
                dat = pickle.load(open(file_name, 'rb'))
                
                dat2dThumb = pickle.load(open('/home/nptl/nptlrig2/Tasks/gestureTask/RightHand_flex_extend_MW_v2.pkl', 'rb'))
                # dat2dThumb = pickle.load(open('C:/Users/Willsey/Documents/NishalTree/nptlrig2/Tasks/gestureTask/RightHand_flex_extend_MW_v2.pkl', 'rb'))
                # file_name = 'C:/Users/Willsey/Documents/NishalTree/nptlrig2/Tasks/gestureTask/fingers_flex_extend3.pkl'

                # dat['finger_directions']['thumb:flex'] = dat2dThumb['finger_directions']['thumb:flex']
                dat['finger_directions']['thumb2:flex'] = dat2dThumb['finger_directions']['thumb2:flex']
            else:
                dat = pickle.load(open(file_name, 'rb'))
            
            if thumb_dim == 1:
                self.fingers_list = ['thumb', 'index', 'middle', 'ring', 'pinky']
                self.finger_dict = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}
            elif thumb_dim == 2:
                self.fingers_list = ['thumb', 'thumb2', 'index', 'middle', 'ring', 'pinky']
                self.finger_dict = {'thumb': 1, 'thumb2': 2, 'index': 3, 'middle': 4, 'ring': 5, 'pinky': 6}

            self.n_trials = n_trials

            # extract keys
            self.keys = dat['keys_use']
            self.finger_directions = dat['finger_directions']
            for ifinger in self.fingers_list:
                if '%s:extend' % ifinger not in self.finger_directions.keys():
                    self.finger_directions.update({'%s:extend' % ifinger: self.finger_directions['idle'] - (self.finger_directions['%s:flex' % ifinger] - 
                                                                                                            self.finger_directions['idle'])})
            
            # For unity
            self.redisServer = redis.StrictRedis(host='localhost', port=6379, db=0)
            self.redisServer.set(f'target/{hand_identity}/keys', '#'.join(self.keys + ['trial_phase'] + self.fingers_list ))
            self.redisServer.set(f'estimated/{hand_identity}/keys', '#'.join(self.keys + ['trial_phase'] + self.fingers_list ))
            
            # For Redis to be sent to PC1
            self.redisServer.set(f'GestureKeys{hand_identity}', '#'.join(self.keys + ['/RightHand/trial_phase'] + self.fingers_list))
            
            self.assign_finger = np.zeros((len(self.keys)))
            for iikey, ikey in enumerate(self.keys):
                for ifinger in self.finger_dict.keys():
                    if ifinger in ikey:
                        self.assign_finger[iikey] = self.finger_dict[ifinger]
            
            self.finger_scales = finger_scales
            self.wts_default = wts_default
            self.idle_state =  np.array([self.wts_default[f] for f in self.fingers_list])
            self.levels = levels
            self.wts_list, self.idle =  self.get_wts(n_finger_groups)
            
            # print(self.wts_list)
            # print(len(self.wts_list))
            # print('Idle:', self.wts_list[self.idle])
            self.n_targets = len(self.wts_list)
            self.targets = np.setdiff1d(np.arange( self.n_targets), self.idle)
            self.hand_identity=hand_identity


    def generate_hand_pos(self, wts):
        values = np.copy(self.finger_directions['idle'])
        for finger in self.finger_dict.keys():
            if ('thumb2' in self.finger_dict.keys()) & ((finger == 'thumb') | (finger == 'thumb2')):
                finger_wt1 = wts['thumb']
                finger_wt2 = wts['thumb2']
                val = np.where((self.assign_finger == self.finger_dict['thumb']) | (self.assign_finger == self.finger_dict['thumb2']))[0]
                values[val] = finger_wt1 * self.finger_directions['%s:flex' % 'thumb'][val] + (1 - finger_wt1) * self.finger_directions['%s:extend' % 'thumb'][val] + finger_wt2 * self.finger_directions['%s:flex' % 'thumb2'][val] + (1 - finger_wt2) * self.finger_directions['%s:extend' % 'thumb2'][val]

            else:
                finger_wt = wts[finger]
                val = np.where(self.assign_finger == self.finger_dict[finger])[0]
                values[val] = finger_wt * self.finger_directions['%s:flex' % finger][val] + (1 - finger_wt) * self.finger_directions['%s:extend' % finger][val]
            # values[val] = (2* np.maximum(finger_wt - 0.5, 0) * (self.finger_directions['%s:flex' % finger][val] - self.finger_directions['idle'][val]) + 
            #                2 * np.maximum((1 - finger_wt) - 0.5, 0) * (self.finger_directions['%s:extend' % finger][val] - self.finger_directions['idle'][val]) + 
            #                self.finger_directions['idle'][val]) 

        return values
    
    def get_wts(self, n_finger_groups):

        if n_finger_groups == 4:
            wts_list = list(itertools.product(*([self.levels]*4)))  # Only four finger groups
            wts_list = [list(w) + [w[-1]] for w in wts_list]  # Weight for pinky = wt for ring finger
            
            #idle = 40
        elif n_finger_groups == 3:
            wts_list = list(itertools.product(* ([self.levels]*3)))  # Only three finger groups
            wts_list = [[w[0], w[1], w[1], w[2], w[2]] for w in wts_list]  # 
            #idle = 13
        elif n_finger_groups == 31:
            wts_list = list(itertools.product(* ([self.levels]*3)))  # Only three finger groups
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:3]) > 0.01) <= 1]
            wts_list = [[w[0], w[1], w[1], w[2], w[2]] for w in wts_list]  # 
            #idle = 13
        elif n_finger_groups == 3112:
            # three finger groups, upto 2 move at a time
            wts_list = list(itertools.product(* ([self.levels]*3)))  # Only three finger groups
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:3]) > 0.01) <= 2]
            wts_list = [[w[0], w[1], w[1], w[2], w[2]] for w in wts_list]  # 
            #idle = 13
        elif n_finger_groups == 32:
            # three finger groups, only one group moves at a time, thumb, index, (MRP)
            # TODO(willsey) : change the groups, number of simultaneous, levels, etc..
            wts_list = list(itertools.product(* ([self.levels]*3)))  # Only three finger groups
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:3]) > 0.01) <= 1]
            wts_list = [[w[0], w[1], w[2], w[2], w[2]] for w in wts_list]  # 
            #idle = 13
        elif n_finger_groups == 42:
            wts_list = list(itertools.product(* ([self.levels]*4)))  # Three finger groups with 2D thumb
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:4]) > 0.01) <= 1]
            wts_list = [[w[0], w[1], w[2], w[2], w[3],w[3]] for w in wts_list]  # 
        elif n_finger_groups == 40:
            wts_list = self.idle_state

        elif n_finger_groups == 331:
            wts_list = list(itertools.product(* ([self.levels]*3)))  # Three finger groups with 2D thumb
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:3]) > 0.01) <= 1]
            wts_list = [[w[0], self.idle_state[0], w[1], w[1], w[2], w[2]] for w in wts_list]  # 

        elif n_finger_groups == 22:
            wts_list = list(itertools.product(* ([self.levels]*2)))  # Two finger groups with 2D thumb
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state[:2]) > 0.01) <= 1]
            wts_list = [[w[0], self.idle_state[0], w[1], w[1], self.idle_state[0],self.idle_state[0]] for w in wts_list]  # 

        elif n_finger_groups == 342:
            idx_m = list(itertools.combinations([0,2,3], 2))
            trials_comb = np.ceil(self.n_trials/len(idx_m))
            for k_m in range(int(len(idx_m))):
                m_m = 0
                while m_m < int(len(idx_m[k_m])):
                    if idx_m[k_m][m_m]==0:
                        idx_m[k_m] = idx_m[k_m][0:m_m+1] + (1,) + idx_m[k_m][m_m+1:]
                        m_m = m_m+1
                    m_m = m_m+1
            
            wts_list = [tuple(self.idle_state[0:4].copy())]
            for k_m in range(int(len(idx_m))):
                for m_m in range(int(trials_comb)):
                    wts_list_m = self.idle_state[0:4].copy()
                    wts_list_m[list(idx_m[k_m])] = np.random.uniform(low=0.0, high=1.0, size=len(idx_m[k_m]))
                    # if k_m == 0 and m_m == 0:
                    #     print([tuple(wts_list_m)]+[tuple(wts_list_m)])
                    #     wts_list = [tuple(wts_list_m)]
                    # else:
                    wts_list = wts_list + [tuple(wts_list_m)]
            wts_list = [[w[0], w[1], w[2], w[2], w[3],w[3]] for w in wts_list]  # 
        elif n_finger_groups == 343:
            idx_m = list(itertools.combinations([0,2,3], 3))
            trials_comb = np.ceil(self.n_trials/len(idx_m))
            for k_m in range(int(len(idx_m))):
                m_m = 0
                while m_m < int(len(idx_m[k_m])):
                    if idx_m[k_m][m_m]==0:
                        idx_m[k_m] = idx_m[k_m][0:m_m+1] + (1,) + idx_m[k_m][m_m+1:]
                        m_m = m_m+1
                    m_m = m_m+1
            
            wts_list = [tuple(self.idle_state[0:4].copy())]
            for k_m in range(int(len(idx_m))):
                for m_m in range(int(trials_comb)):
                    wts_list_m = self.idle_state[0:4].copy()
                    wts_list_m[list(idx_m[k_m])] = np.random.uniform(low=0.0, high=1.0, size=len(idx_m[k_m]))
                    # if k_m == 0 and m_m == 0:
                    #     print([tuple(wts_list_m)]+[tuple(wts_list_m)])
                    #     wts_list = [tuple(wts_list_m)]
                    # else:
                    wts_list = wts_list + [tuple(wts_list_m)]
            wts_list = [[w[0], w[1], w[2], w[2], w[3],w[3]] for w in wts_list]  # 
        elif n_finger_groups == 222:
            idx_m = list(itertools.combinations([0,2], 2))
            trials_comb = np.ceil(self.n_trials/len(idx_m))
            for k_m in range(int(len(idx_m))):
                m_m = 0
                while m_m < int(len(idx_m[k_m])):
                    if idx_m[k_m][m_m]==0:
                        idx_m[k_m] = idx_m[k_m][0:m_m+1] + (1,) + idx_m[k_m][m_m+1:]
                        m_m = m_m+1
                    m_m = m_m+1
            
            wts_list = [tuple(self.idle_state[0:4].copy())]
            for k_m in range(int(len(idx_m))):
                for m_m in range(int(trials_comb)):
                    wts_list_m = self.idle_state[0:4].copy()
                    wts_list_m[list(idx_m[k_m])] = np.random.uniform(low=0.0, high=1.0, size=len(idx_m[k_m]))
                    # if k_m == 0 and m_m == 0:
                    #     print([tuple(wts_list_m)]+[tuple(wts_list_m)])
                    #     wts_list = [tuple(wts_list_m)]
                    # else:
                    wts_list = wts_list + [tuple(wts_list_m)]
            wts_list = [[w[0], self.idle_state[1], w[2], w[2], self.idle_state[3],self.idle_state[3]] for w in wts_list]  # 

        elif n_finger_groups == 332:
            idx_m = list(itertools.combinations([0,2,3], 2))
            trials_comb = np.ceil(self.n_trials/len(idx_m))
            for k_m in range(int(len(idx_m))):
                m_m = 0
                while m_m < int(len(idx_m[k_m])):
                    if idx_m[k_m][m_m]==0:
                        idx_m[k_m] = idx_m[k_m][0:m_m+1] + (1,) + idx_m[k_m][m_m+1:]
                        m_m = m_m+1
                    m_m = m_m+1
            
            wts_list = [tuple(self.idle_state[0:4].copy())]
            for k_m in range(int(len(idx_m))):
                for m_m in range(int(trials_comb)):
                    wts_list_m = self.idle_state[0:4].copy()
                    wts_list_m[list(idx_m[k_m])] = np.random.uniform(low=0.0, high=1.0, size=len(idx_m[k_m]))
                    # if k_m == 0 and m_m == 0:
                    #     print([tuple(wts_list_m)]+[tuple(wts_list_m)])
                    #     wts_list = [tuple(wts_list_m)]
                    # else:
                    wts_list = wts_list + [tuple(wts_list_m)]
            wts_list = [[w[0], self.idle_state[1], w[2], w[2], w[3], w[3]] for w in wts_list]  # 

        elif n_finger_groups == 2: 
            wts_list = list(itertools.product(* ([self.levels]*2)))  # Only two finger groups
            wts_list = [list(w) + [w[-1]] + [w[-1]] + [w[-1]] for w in wts_list]  #
            #idle = 4
        # elif n_finger_groups == 22: 
        #     # moves only index and middle fingers. keeps ring and pinky still
        #     wts_list = list(itertools.product(*([self.levels]*2)))  # Only two finger groups
        #     wts_list = [list(w) + [w[-1]] + [self.wts_default['ring']] + [self.wts_default['pinky']] for w in wts_list]  #
        #     #idle = 4
        
        elif n_finger_groups == 1:                 
            wts_list = list(itertools.product(*([self.levels]*5))) 
            # remove wts with more then one finger moving
            wts_list = [w for w in wts_list if np.sum(np.abs(np.array(w) - self.idle_state) > 0.01) <= 1]
            #idle = 5
        elif n_finger_groups == 11:
            '''Thumb only '''                 
            wts_list = list(itertools.product(*([self.levels]*1))) 
            wts_list = [list(w) + [self.wts_default['index']] + [self.wts_default['middle']] + [self.wts_default['ring']] + [self.wts_default['pinky']] for w in wts_list] 
        else:
            raise ValueError('Finger groups not implemented yet')

        wts_list = self.scale_around_idle(wts_list)
        idle = self.get_idle(wts_list)
        return wts_list, idle

    def scale_around_idle(self, wts_list):
        wts_list_new = []
        
        for w in wts_list:
            w = (np.array(w) - self.idle_state) * self.finger_scales + self.idle_state
            wts_list_new += [list(w)]
        return wts_list_new

    def get_idle(self, wts_list):
        for iw, w in enumerate(wts_list):
            if np.sum(np.abs(np.array(w) - self.idle_state)) == 0:
                return iw

    def update(self, estimate_render):
        # scale up thumb because the decoder has corrected for it.
        w = (np.array(estimate_render) - self.idle_state) * self.finger_scales + self.idle_state	
        w = list(w)		
        wts_est = dict(zip(self.fingers_list, w))
        values_est_fix = self.generate_hand_pos(wts_est)
        trial_phase = 1
        self.redisServer.set(f'estimated/{self.hand_identity}/values', '#'.join([str(values_est_fix[ikey]) for ikey in range(len(self.keys))] + [str(trial_phase)] + [str(wts_est[ifinger]) for ifinger in self.fingers_list]))
	
    
    def closed_loop_assist(self, yk_decoded, start, end, attenuation=0, move_towards_target=0, push_speed=0, push_magnitude=0, assist_on=False):
        '''Error attentution and push-to-target for closed loop trials.
        
        This is a strategy to make closed loop trials easier, and look more like open loop trials by reducing the movement orthogonal to target.
        
        Args:
            yk_decoded: decoded position
            start: trial starting position
            end: trial ending position
            attenuation: how much to reduce off-target movements, in [0 = no attenuation, 1=full attenuation]
        '''
        # error attenuation
        if attenuation > 0:
            if assist_on:
                dir_ = (end - start) 
                if np.sqrt(np.sum(dir_**2)) > 0.01:
                    dir_ = dir_ / np.sqrt(np.sum((end - start)**2))
                    yk_diff = yk_decoded - start
                    output_target_dir = (yk_diff.dot(dir_)) * dir_
                    output_error_dir = yk_diff - output_target_dir
                    yk_decoded = output_target_dir + (1 - attenuation) * output_error_dir + start 

        # only move towards target 
        if move_towards_target > 0: 
            if assist_on:
                if not hasattr(self, 'prev_yk_decoded'):
                    self.prev_yk_decoded = yk_decoded
                
                bad_fingers = np.abs(end - yk_decoded) > np.abs(end - self.prev_yk_decoded)
                yk_decoded[bad_fingers] = self.prev_yk_decoded[bad_fingers]
                self.prev_yk_decoded = yk_decoded
                

        # push towards target
        if push_magnitude > 0:
            if not hasattr(self, 'push_pos'):
                    self.push_pos = np.array(self.wts_list[self.idle])
        
            if assist_on:  # its false during init and wait, ON otherwise
                push_dir = (end - self.push_pos)
                dist_target = np.sqrt(np.sum(push_dir ** 2))
                if np.sqrt(np.sum(push_dir**2)) > 0.01:
                    push_dir = push_dir / np.sqrt(np.sum((end - self.push_pos)**2))

                self.push_pos = self.push_pos + np.minimum(push_speed, dist_target) * push_dir
                #print(start, end, yk_decoded, self.push_pos, dist_target, push_speed)

            yk_decoded = (1 - push_magnitude)*yk_decoded + push_magnitude * self.push_pos
            
        #print(start, end, yk_decoded, output)
        return yk_decoded

    def is_on_target(self, yk_decoded, wts_targ_, hold_threshold):
        return np.abs(yk_decoded - wts_targ_) < self.finger_scales * hold_threshold

    def write_skeleton(self, redisServer, lowdim_pos, string, char):

        wts_target = dict(zip(self.fingers_list, lowdim_pos))
        values_target_fix = self.generate_hand_pos(wts_target)
        redisServer.set(f'{string}/keys', '#'.join([ikey for ikey in self.keys]))
        redisServer.set(f'{string}/%s/values' % (char), '#'.join([str(ival) for ival in values_target_fix]))