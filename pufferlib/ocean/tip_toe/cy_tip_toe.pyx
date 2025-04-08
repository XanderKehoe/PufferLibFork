cimport numpy as cnp
from libc.stdlib cimport calloc, free  # Use calloc for zero-initialized allocation
from libc.stdint cimport uint64_t

cdef extern from "tip_toe.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CTipToeEnv:
        char* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer

        int grid_size_x
        int grid_size_y
        int num_agents
        int max_steps

    ctypedef struct Client

    void initialize_env(CTipToeEnv* env)
    void free_allocated(CTipToeEnv* env)

    Client* make_client(CTipToeEnv* env)
    void close_client(Client* client)
    void render(Client* client, CTipToeEnv* env) 

    void reset(CTipToeEnv* env)
    void step(CTipToeEnv* env)

cdef class CyTipToe:
    cdef:
        CTipToeEnv* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, char[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, 
            int num_agents=3, int grid_size_x=10, int grid_size_y=10, int max_steps=300, 
            float negative_reward=-0.01, float positive_reward=0.5):
        self.num_envs = num_envs
        self.envs = <CTipToeEnv*>calloc(num_envs, sizeof(CTipToeEnv))
        if self.envs == NULL:
            raise MemoryError("Failed to allocate memory for CTipToeEnv")
        self.client = NULL

        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int inc = num_agents
        
        cdef int i
        for i in range(num_envs):
            self.envs[i] = CTipToeEnv(
                observations=&observations[inc*i, 0],
                actions=&actions[inc*i],
                rewards=&rewards[inc*i],
                dones=&terminals[inc*i],
                log_buffer=self.logs, 
                grid_size_x=grid_size_x,
                grid_size_y=grid_size_y, 
                num_agents=num_agents,
                max_steps=max_steps,
            )
            initialize_env(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef CTipToeEnv* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
