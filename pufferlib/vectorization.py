from pdb import set_trace as T

import numpy as np
import gymnasium
from itertools import chain
import psutil
import time
import msgpack


from pufferlib import namespace
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.multi_env import create_precheck, GymnasiumMultiEnv, PettingZooMultiEnv, PufferEnvWrapper
from pufferlib.exceptions import APIUsageError
import pufferlib.spaces


RESET = 0
SEND = 1
RECV = 2

space_error_msg = 'env {env} must be an instance of GymnasiumPufferEnv or PettingZooPufferEnv'


def calc_scale_params(num_envs, envs_per_batch, envs_per_worker, agents_per_env):
    '''These calcs are simple but easy to mess up and hard to catch downstream.
    We do them all at once here to avoid that'''

    if num_envs % envs_per_worker != 0:
        raise APIUsageError('num_envs must be divisible by envs_per_worker')
    
    num_workers = num_envs // envs_per_worker
    envs_per_batch = num_envs if envs_per_batch is None else envs_per_batch

    if envs_per_batch > num_envs:
        raise APIUsageError('envs_per_batch must be <= num_envs')
    if envs_per_batch % envs_per_worker != 0:
        raise APIUsageError('envs_per_batch must be divisible by envs_per_worker')
    if envs_per_batch < 1:
        raise APIUsageError('envs_per_batch must be > 0')

    workers_per_batch = envs_per_batch // envs_per_worker
    assert workers_per_batch <= num_workers

    agents_per_batch = envs_per_batch * agents_per_env
    agents_per_worker = envs_per_worker * agents_per_env
 
    return num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker

def setup(env_creator, env_args, env_kwargs):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, GymnasiumPufferEnv):
        multi_env_cls = GymnasiumMultiEnv 
        env_agents = 1
    elif isinstance(driver_env, PettingZooPufferEnv):
        multi_env_cls = PettingZooMultiEnv
        env_agents = len(driver_env.possible_agents)
    else:# isinstance(driver_env, PufferEnv):
        #multi_env_cls = PufferEnv
        env_agents = driver_env.num_agents

    '''
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GymnasiumPufferEnv or PettingZooPufferEnv'
        )
    '''
    multi_env_cls = PufferEnvWrapper
    obs_space = _single_observation_space(driver_env)
    return driver_env, multi_env_cls, env_agents

def _single_observation_space(env):
    if isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    return env.observation_space
 
    if isinstance(env, GymnasiumPufferEnv):
        return env.observation_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    return env.action_space
 
    if isinstance(env, GymnasiumPufferEnv):
        return env.action_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_action_space(state):
    return _single_action_space(state.driver_env)

def structured_observation_space(state):
    return state.driver_env.structured_observation_space

def flat_observation_space(state):
    return state.driver_env.flat_observation_space

def unpack_batched_obs(state, obs):
    return state.driver_env.unpack_batched_obs(obs)

def recv_precheck(state):
    assert state.flag == RECV, 'Call reset before stepping'
    state.flag = SEND

def send_precheck(state):
    assert state.flag == SEND, 'Call reset + recv before send'
    state.flag = RECV

def reset_precheck(state):
    assert state.flag == RESET, 'Call reset only once on initialization'
    state.flag = RECV

def reset(self, seed=None):
    self.async_reset(seed)
    data = self.recv()
    return data[0], data[4]

def step(self, actions):
    actions = np.asarray(actions)
    self.send(actions)
    return self.recv()[:-1]

class Serial:
    '''Runs environments in serial on the main process
    
    Use this vectorization module for debugging environments
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs
    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            num_workers: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            mask_agents: bool = False,
            ) -> None:

        self.driver_env, scale = buffer_scale(env_creator, env_args, env_kwargs,
            num_envs, num_workers, envs_per_batch=None)

        '''
        driver_env, multi_env_cls, agents_per_env = setup(
            env_creator, env_args, env_kwargs)
        num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, agents_per_env)

        agents_per_worker = agents_per_env * envs_per_worker
        observation_shape = _single_observation_space(driver_env).shape
        observation_size = int(np.prod(observation_shape))
        observation_dtype = _single_observation_space(driver_env).dtype
        action_shape = _single_action_space(driver_env).shape
        action_size = int(np.prod(action_shape))
        action_dtype = _single_action_space(driver_env).dtype

        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.workers_per_batch = workers_per_batch
        self.envs_per_worker = envs_per_worker
        self.agents_per_worker = agents_per_worker
        self.agents_per_batch = agents_per_batch
        self.agents_per_env = agents_per_env
        self.agent_ids = np.stack([np.arange(
            i*agents_per_worker, (i+1)*agents_per_worker) for i in range(num_workers)])
        '''

        self.obs_arr = np.ndarray(scale.observation_buffer_shape, dtype=scale.observation_dtype)
        self.rewards_arr = np.ndarray(scale.batch_shape, dtype=np.float32)
        self.terminals_arr = np.ndarray(scale.batch_shape, dtype=bool)
        self.truncated_arr = np.ndarray(scale.batch_shape, dtype=bool)
        self.mask_arr = np.ndarray(scale.batch_shape, dtype=bool)

        self.multi_envs = [
            PufferEnvWrapper(env_creator, env_args, env_kwargs, envs_per_worker,
                obs_mem=self.obs_arr[i], rew_mem=self.rewards_arr[i],
                done_mem=self.terminals_arr[i], trunc_mem=self.truncated_arr[i], mask_mem=self.mask_arr[i])
            for i in range(num_workers)
        ]

        self.flag = RESET
        self.scale = scale

    def recv(self):
        recv_precheck(self)
        ids = self.scale.workers_per_batch
        o = self.obs_arr[:ids].reshape(*self.observation_batch_shape)
        r = self.rewards_arr[:ids].ravel()
        d = self.terminals_arr[:ids].ravel()
        t = self.truncated_arr[:ids].ravel()
        m = self.mask_arr[:ids].ravel()

        agent_ids = self.agent_ids[:ids].ravel()
        return o, r, d, t, self.infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.workers_per_batch, self.agents_per_worker, *self.action_shape)
        self.infos = []
        for worker in range(self.workers_per_batch):
            atns = actions[worker].reshape(self.envs_per_worker, self.agents_per_env, *self.action_shape)
            _, _, _, _, infos, _ = self.multi_envs[worker].step(atns)
            self.infos.extend(infos)

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            kwargs = {}
        else:
            kwargs = {"seed": seed}

        for i in range(self.workers_per_batch):
            _, _, _, _, self.infos, _ = self.multi_envs[i].reset(**kwargs)

    def put(self, *args, **kwargs):
        for e in self.multi_envs:
            e.put(*args, **kwargs)

    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.multi_envs]

    def close(self):
        for e in self.multi_envs:
            e.close()

STEP = b"s"
RESET = b"r"
RESET_NONE = b"n"
CLOSE = b"c"

def _worker_process(multi_env_cls, env_creator, env_args, env_kwargs,
        num_envs, agents_per_env, worker_idx, obs_shape, obs_mem, atn_shape, atn_mem, rewards_mem,
        terminals_mem, truncated_mem, mask_mem, observation_dtype, action_dtype, send_pipe, recv_pipe):
    
    # I don't know if this helps. Sometimes it does, sometimes not.
    # Need to run more comprehensive tests
    curr_process = psutil.Process()
    #curr_process.cpu_affinity([worker_idx+1])
    # Set to min niceness
    #curr_process.nice(19)


    num_agents = num_envs * agents_per_env
    obs_size = int(np.prod(obs_shape))
    obs_n = num_agents * obs_size

    atn_size = int(np.prod(atn_shape))
    atn_n = num_agents * atn_size

    s = worker_idx * num_agents
    e = (worker_idx + 1) * num_agents 
    s_obs = worker_idx * num_agents * obs_size
    e_obs = (worker_idx + 1) * num_agents * obs_size
    s_atn = worker_idx * num_agents * atn_size
    e_atn = (worker_idx + 1) * num_agents * atn_size

    obs_arr = np.frombuffer(obs_mem, dtype=observation_dtype)[s_obs:e_obs].reshape(num_agents, *obs_shape)
    atn_arr = np.frombuffer(atn_mem, dtype=action_dtype)[s_atn:e_atn]
    rewards_arr = np.frombuffer(rewards_mem, dtype=np.float32)[s:e]
    terminals_arr = np.frombuffer(terminals_mem, dtype=bool)[s:e]
    truncated_arr = np.frombuffer(truncated_mem, dtype=bool)[s:e]
    mask_arr = np.frombuffer(mask_mem, dtype=bool)[s:e]

    envs = multi_env_cls(env_creator, env_args, env_kwargs, n=num_envs,
        obs_mem=obs_arr, rew_mem=rewards_arr, done_mem=terminals_arr,
        trunc_mem=truncated_arr, mask_mem=mask_arr)

    while True:
        request = recv_pipe.recv_bytes()
        info = {}
        if request == RESET:
            response = envs.reset()
        elif request == STEP:
            response = envs.step(atn_arr.reshape(num_envs, agents_per_env, *atn_shape))

        #obs, reward, done, truncated, info = response

        # TESTED: There is no overhead associated with 4 assignments to shared memory
        # vs. 4 assigns to an intermediate numpy array and then 1 assign to shared memory
        #obs_arr[:] = obs
        #rewards_arr[:] = reward
        #terminals_arr[:] = done
        #truncated_arr[:] = truncated
        #mask_arr[:] = envs.preallocated_masks
        send_pipe.send(info)

def buffer_scale(env_creator, env_args, env_kwargs, num_envs, num_workers, envs_per_batch=None):
    '''These calcs are simple but easy to mess up and hard to catch downstream.
    We do them all at once here to avoid that'''
    if num_envs % num_workers != 0:
        raise APIUsageError('num_envs must be divisible by num_workers')
    if num_envs < num_workers:
        raise APIUsageError('num_envs must be >= num_workers')
    if envs_per_batch is None:
        envs_per_batch = num_envs
    if envs_per_batch > num_envs:
        raise APIUsageError('envs_per_batch must be <= num_envs')
    if envs_per_batch % envs_per_worker != 0:
        raise APIUsageError('envs_per_batch must be divisible by envs_per_worker')
    if envs_per_batch % num_workers != 0:
        raise APIUsageError('envs_per_batch must be divisible by num_workers')
    if envs_per_batch < 1:
        raise APIUsageError('envs_per_batch must be > 0')

    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, GymnasiumPufferEnv):
        agents_per_env = 1
    elif isinstance(driver_env, PettingZooPufferEnv):
        agents_per_env = len(driver_env.possible_agents)
    else:# isinstance(driver_env, PufferEnv):
        agents_per_env = driver_env.num_agents

    '''
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GymnasiumPufferEnv or PettingZooPufferEnv'
        )
    '''

    workers_per_batch = envs_per_batch // num_workers
    agents_per_batch = envs_per_batch * agents_per_env

    envs_per_worker = num_envs // num_workers
    agents_per_worker = envs_per_worker * agents_per_env

    observation_shape = _single_observation_space(driver_env).shape
    observation_dtype = _single_observation_space(driver_env).dtype
    action_shape = _single_action_space(driver_env).shape
    action_dtype = _single_action_space(driver_env).dtype

    observation_buffer_shape = (num_workers, agents_per_worker, *observation_shape)
    observation_batch_shape = (agents_per_batch, *observation_shape)
    action_buffer_shape = (num_workers, agents_per_worker, *action_shape)
    action_batch_shape = (workers_per_batch, *action_shape)
    batch_shape = (num_workers, agents_per_worker)

    agent_ids = np.stack([np.arange(
        i*agents_per_worker, (i+1)*agents_per_worker) for i in range(num_workers)])

    return driver_env, pufferlib.namespace(
        num_envs=num_envs,
        num_workers=num_workers,
        envs_per_batch=envs_per_batch,
        workers_per_batch=workers_per_batch,
        agents_per_batch=agents_per_batch,
        agents_per_worker=agents_per_worker,
        agents_per_env=agents_per_env,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        observation_buffer_shape=observation_buffer_shape,
        observation_batch_shape=observation_batch_shape,
        action_buffer_shape=action_buffer_shape,
        action_batch_shape=action_batch_shape,
        batch_shape=batch_shape,
        agent_ids=agent_ids,
    )

class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            env_pool: bool = False,
            mask_agents: bool = False,
            ) -> None:
        driver_env, multi_env_cls, agents_per_env = setup(
            env_creator, env_args, env_kwargs)
        num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, agents_per_env)

        agents_per_worker = agents_per_env * envs_per_worker
        observation_shape = _single_observation_space(driver_env).shape
        observation_dtype = _single_observation_space(driver_env).dtype
        action_shape = _single_action_space(driver_env).shape
        action_dtype = _single_action_space(driver_env).dtype

        # Shared memory for obs, rewards, terminals, truncateds
        from multiprocessing import Process, Manager, Pipe, Array
        from multiprocessing.sharedctypes import RawArray

        observation_size = int(np.prod(observation_shape))
        obs_mem = RawArray(np.ctypeslib.as_ctypes_type(observation_dtype),
                num_workers*agents_per_worker*observation_size)

        action_size = int(np.prod(action_shape))
        atn_mem = RawArray(np.ctypeslib.as_ctypes_type(action_dtype),
                num_workers*agents_per_worker*action_size)

        rewards_mem = RawArray('f', num_workers*agents_per_worker)
        terminals_mem = RawArray('b', num_workers*agents_per_worker)
        truncated_mem = RawArray('b', num_workers*agents_per_worker)
        mask_mem = RawArray('b', num_workers*agents_per_worker)

        obs_arr = np.ndarray((num_workers, agents_per_worker, *observation_shape), dtype=observation_dtype, buffer=obs_mem)
        atn_arr = np.ndarray((num_workers, agents_per_worker, *action_shape), dtype=action_dtype, buffer=atn_mem)
        rewards_arr = np.ndarray((num_workers, agents_per_worker), dtype=np.float32, buffer=rewards_mem)
        terminals_arr = np.ndarray((num_workers, agents_per_worker), dtype=bool, buffer=terminals_mem)
        truncated_arr = np.ndarray((num_workers, agents_per_worker), dtype=bool, buffer=truncated_mem)
        mask_arr = np.ndarray((num_workers, agents_per_worker), dtype=bool, buffer=mask_mem)

        main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        recv_pipe_dict = {p: i for i, p in enumerate(main_recv_pipes)}

        num_cores = psutil.cpu_count()
        '''
        from multiprocessing import Pool
        from multiprocessing import get_context
        pool = get_context('spawn').Pool(num_cores)
        for i in range(num_workers):
            pool.apply_async(_worker_process, args=(multi_env_cls, env_creator, env_args, env_kwargs, envs_per_worker, agents_per_env, i,
                    observation_shape, obs_mem, action_shape, atn_mem, rewards_mem, terminals_mem, truncated_mem,
                    mask_mem, observation_dtype, action_dtype,
                    work_send_pipes[i], work_recv_pipes[i]))
 

        '''
        processes = []
        for i in range(num_workers):
            p = Process(
                target=_worker_process,
                args=(multi_env_cls, env_creator, env_args, env_kwargs, envs_per_worker, agents_per_env, i,
                    observation_shape, obs_mem, action_shape, atn_mem, rewards_mem, terminals_mem, truncated_mem,
                    mask_mem, observation_dtype, action_dtype,
                    work_send_pipes[i], work_recv_pipes[i])
            )
            p.start()
            processes.append(p)

        # Register all receive pipes with the selector
        import selectors
        sel = selectors.DefaultSelector()
        for pipe in main_recv_pipes:
            sel.register(pipe, selectors.EVENT_READ)

        self.agent_ids = np.stack([np.arange(
            i*agents_per_worker, (i+1)*agents_per_worker) for i in range(num_workers)])

        self.processes = processes
        self.sel = sel
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.obs_arr = obs_arr
        self.action_shape = action_shape
        self.atn_arr = atn_arr
        self.rewards_arr = rewards_arr
        self.terminals_arr = terminals_arr
        self.truncated_arr = truncated_arr
        self.mask_arr = mask_arr
        self.send_pipes = main_send_pipes
        self.recv_pipes = main_recv_pipes
        self.recv_pipe_dict = recv_pipe_dict
        self.driver_env = driver_env
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers_per_batch = workers_per_batch
        self.envs_per_batch = envs_per_batch
        self.envs_per_worker = envs_per_worker
        self.agents_per_batch = agents_per_batch
        self.agents_per_worker = agents_per_worker
        self.agents_per_env = agents_per_env
        self.async_handles = None
        self.flag = RESET
        self.prev_env_id = []
        self.env_pool = env_pool
        self.mask_agents = mask_agents

    def recv(self):
        recv_precheck(self)
        worker_ids = []
        infos = []
        if self.env_pool:
            while len(worker_ids) < self.workers_per_batch:
                for key, _ in self.sel.select(timeout=None):
                    response_pipe = key.fileobj
                    info = response_pipe.recv()
                    infos.append(info)
                    env_id = self.recv_pipe_dict[response_pipe]
                    worker_ids.append(env_id)

                    if len(worker_ids) == self.workers_per_batch:                    
                        break
        else:
            for env_id in range(self.workers_per_batch):
                response_pipe = self.recv_pipes[env_id]
                info = response_pipe.recv()
                infos.append(info)
                worker_ids.append(env_id)

        infos = [i for ii in infos for i in ii]

        # Does not copy if workers_per_batch == 1
        if self.workers_per_batch == 1:
            worker_ids = worker_ids[0]
        else:
            worker_ids = np.array(worker_ids)

        o = self.obs_arr[worker_ids].reshape(self.agents_per_batch, *self.observation_shape)
        r = self.rewards_arr[worker_ids].ravel()
        d = self.terminals_arr[worker_ids].ravel()
        t = self.truncated_arr[worker_ids].ravel()
        m = self.mask_arr[worker_ids].ravel()

        self.prev_env_id = worker_ids
        agent_ids = self.agent_ids[worker_ids].ravel()
        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.workers_per_batch, self.agents_per_worker, *self.action_shape)
        self.atn_arr[self.prev_env_id] = actions
        if self.workers_per_batch == 1:
            self.send_pipes[self.prev_env_id].send_bytes(STEP)
        else:
            for i in self.prev_env_id:
                self.send_pipes[i].send_bytes(STEP)

    def async_reset(self, seed=None):
        reset_precheck(self)
        for pipe in self.send_pipes:
            pipe.send_bytes(RESET)

        return
        # TODO: Seed

        if seed is None:
            for pipe in self.send_pipes:
                pipe.send(RESET)
        else:
            for idx, pipe in enumerate(self.send_pipes):
                pipe.send(("reset", [], {"seed": seed+idx}))

    def put(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("put", args, kwargs))

    def get(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("get", args, kwargs))

        idx = -1
        recvs = []
        while len(recvs) < self.workers_per_batch // self.envs_per_worker:
            idx = (idx + 1) % self.num_workers
            queue = self.response_queues[idx]

            if queue.empty():
                continue

            response = queue.get()
            if response is not None:
                recvs.append(response)

        return recvs

    def close(self):
        for pipe in self.send_pipes:
            pipe.send(("close", [], {}))

        #self.pool.terminate()
        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()

class Ray():
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            env_pool: bool = False,
            mask_agents: bool = False,
            ) -> None:
        driver_env, multi_env_cls, agents_per_env = setup(
            env_creator, env_args, env_kwargs)
        num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, agents_per_env)

        import ray
        if not ray.is_initialized():
            import logging
            ray.init(
                include_dashboard=False,  # WSL Compatibility
                logging_level=logging.ERROR,
            )

        multi_envs = [
            ray.remote(multi_env_cls).remote(
                env_creator, env_args, env_kwargs, envs_per_worker
            ) for _ in range(num_workers)
        ]

        self.agent_ids = np.stack([np.arange(
            i*agents_per_worker, (i+1)*agents_per_worker) for i in range(num_workers)])
        self.observation_shape = _single_observation_space(driver_env).shape
        self.action_shape = _single_action_space(driver_env).shape
        self.multi_envs = multi_envs
        self.driver_env = driver_env
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers_per_batch = workers_per_batch
        self.envs_per_batch = envs_per_batch
        self.envs_per_worker = envs_per_worker
        self.agents_per_batch = agents_per_batch
        self.agents_per_worker = agents_per_worker
        self.agents_per_env = agents_per_env
        self.async_handles = None
        self.flag = RESET
        self.ray = ray
        self.prev_env_id = []
        self.env_pool = env_pool
        self.mask_agents = mask_agents

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        if self.env_pool:
            recvs = self.ray.get(self.async_handles[:self.workers_per_batch])
            env_id = [_ for _ in range(self.workers_per_batch)]
        else:
            ready, busy = self.ray.wait(
                self.async_handles, num_returns=self.workers_per_batch)
            env_id = [self.async_handles.index(e) for e in ready]
            recvs = self.ray.get(ready)

        
        o, r, d, t, i, m = zip(*recvs)
        self.prev_env_id = env_id

        o = np.stack(o, axis=0).reshape(self.agents_per_batch, *self.observation_shape)
        r = np.stack(r, axis=0).ravel()
        d = np.stack(d, axis=0).ravel()
        t = np.stack(t, axis=0).ravel()
        m = np.stack(m, axis=0).ravel()
        agent_ids = self.agent_ids[env_id].ravel()
        return o, r, d, t, i, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.workers_per_batch, self.agents_per_worker, *self.action_shape)
        handles = []
        for i, e in enumerate(self.prev_env_id):
            atns = actions[i].reshape(self.envs_per_worker, self.agents_per_env, *self.action_shape)
            handles.append(self.multi_envs[e].step.remote(atns))

        self.async_handles = handles

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            kwargs = {}
        else:
            kwargs = {"seed": seed}

        handles = []
        for idx, e in enumerate(self.multi_envs):
            handles.append(e.reset.remote(**kwargs))

        self.async_handles = handles

    def put(self, *args, **kwargs):
        for e in self.multi_envs:
            e.put.remote(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.ray.get([e.get.remote(*args, **kwargs) for e in self.multi_envs])

    def close(self):
        self.ray.get([e.close.remote() for e in self.multi_envs])
        self.ray.shutdown()
