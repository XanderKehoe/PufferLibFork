#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "raylib.h"

#define ACTION_UP 0
#define ACTION_UP_RIGHT 1
#define ACTION_RIGHT 2
#define ACTION_RIGHT_DOWN 3
#define ACTION_DOWN 4
#define ACTION_DOWN_LEFT 5
#define ACTION_LEFT 6
#define ACTION_LEFT_UP 7
#define ACTION_NOOP 8

#define TILE_UNKNOWN 0
#define TILE_REMOVED 1
#define TILE_REAL    2

#define CELL_SIZE 40

#define LOG_BUFFER_SIZE 1024

typedef struct Log {
    float episode_return;
    float episode_length;
} Log;

typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// LogBuffer functions
LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    logs->idx = 0;
    return log;
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

typedef struct {
    int state;  // TILE_UNKNOWN, TILE_REMOVED, TILE_REAL
    bool is_fake;  // True if stepped on and fake
} Tile;

typedef struct {
    float pos_x;
    float pos_y;

    float velocity;

    float total_episode_reward;
} Agent;

typedef struct {
    float dx;
    float dy;
    float dist_sq;
} RelativePos;

typedef struct {
    // Interface for PufferLib
    char* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;

    int grid_size_x;
    int grid_size_y;
    int num_agents;
    int max_steps;
    int current_step;

    Agent* agents;
    Tile* grid;
    float start_y;
    float finish_y;

    int total_num_obs;

    bool do_human_control;
} CTipToeEnv;

int get_grid_index(CTipToeEnv* env, int x, int y) {
    return (y * env->grid_size_x) + x;
}

int compare_by_relative_distance(const void* a, const void* b) {
    const RelativePos* ra = (const RelativePos*)a;
    const RelativePos* rb = (const RelativePos*)b;
    if (ra->dist_sq < rb->dist_sq) return -1;
    else if (ra->dist_sq > rb->dist_sq) return 1;
    return 0;
}

void compute_observations(CTipToeEnv* env) {
    char* obs = env->observations;
    memset(obs, 0, env->total_num_obs * sizeof(char));

    int obs_per_agent = env->grid_size_x * env->grid_size_y + (env->num_agents - 1) * 2;

    for (int i = 0; i < env->num_agents; i++) {
        int offset = i * obs_per_agent;

        // Encode tile states
        for (int y = 0; y < env->grid_size_y; y++) {
            for (int x = 0; x < env->grid_size_x; x++) {
                Tile* tile = &env->grid[get_grid_index(env, x, y)];
                obs[offset++] = (char)(tile->state);
            }
        }

        Agent* a = &env->agents[i];

        // Encode current agent position
        obs[offset++] = (char) (a->pos_x / env->grid_size_x * CELL_SIZE);
        obs[offset++] = (char) (a->pos_y / env->grid_size_y * CELL_SIZE);

        // Collect relative positions to other agents and sort them
        int num_others = env->num_agents - 1;
        RelativePos* rels = (RelativePos*)alloca(sizeof(RelativePos) * num_others); // stack-allocated

        int idx = 0;
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) continue;

            Agent* b = &env->agents[j];
            float dx = (b->pos_x - a->pos_x);
            float dy = (b->pos_y - a->pos_y);
            rels[idx].dx = dx;
            rels[idx].dy = dy;
            rels[idx].dist_sq = dx * dx + dy * dy;
            idx++;
        }

        // Sort by distance
        qsort(rels, num_others, sizeof(RelativePos), compare_by_relative_distance);

        // Normalize and encode
        for (int k = 0; k < num_others; k++) {
            float norm_dx = rels[k].dx / (env->grid_size_x * CELL_SIZE);
            float norm_dy = rels[k].dy / (env->grid_size_y * CELL_SIZE);
        
            int dx_byte = (int)((norm_dx + 1.0f) * 0.5f * 255.0f);
            int dy_byte = (int)((norm_dy + 1.0f) * 0.5f * 255.0f);
        
            if (dx_byte < 0) dx_byte = 0;
            if (dx_byte > 255) dx_byte = 255;
            if (dy_byte < 0) dy_byte = 0;
            if (dy_byte > 255) dy_byte = 255;
        
            obs[offset++] = (char)dx_byte;
            obs[offset++] = (char)dy_byte;
        }        
    }
}

void reset(CTipToeEnv* env); // declaration here so we can use before defined.

void reset_agent_position(CTipToeEnv* env, int agent_index){
    Agent* agent = &env->agents[agent_index];
    agent->pos_x = (rand() % env->grid_size_x) * CELL_SIZE;
    agent->pos_y = env->start_y;
}

void check_agent_in_invalid_position_agent_position(CTipToeEnv* env, int agent_idx){
    bool invalid = false;
    Agent* agent = &env->agents[agent_idx];
    if (agent->pos_x < 0.0f || agent->pos_x >= env->grid_size_x * CELL_SIZE || 
        agent->pos_y < 0.0f || agent->pos_y >= (env->grid_size_y + 3) * CELL_SIZE) { // 2 cells for start zone and 1 cell for finish line = 3
        invalid = true;
    }
    else{
        // Check for stepping on a fake or removed tile
        int finish_line_cell_height = 1;
        int tile_x = (int) agent->pos_x / CELL_SIZE;
        int tile_y = (int) (agent->pos_y - CELL_SIZE * finish_line_cell_height) / CELL_SIZE; // adjust for finish line cell size here

        if (tile_y < env->grid_size_y) // check if agent is even on the grid (no need to check for x since should always align in the x-axis with grid)
        {
            Tile* tile = &env->grid[get_grid_index(env, tile_x, tile_y)];

            if (tile != NULL){
                if (tile->state == TILE_REMOVED || tile->is_fake) {
                    tile->state = TILE_REMOVED;
                    invalid = true;
                }
                else if (tile->state == TILE_UNKNOWN){
                    tile->state = TILE_REAL;
                }
            }
        }
    }

    if (invalid){
        reset_agent_position(env, agent_idx);
        env->rewards[agent_idx] -= 0.1f;
    }
}

void apply_agent_repulsion(CTipToeEnv* env, int agent_idx) {
    Agent* self = &env->agents[agent_idx];
    float repulsion_radius = CELL_SIZE * 0.75f;
    float repulsion_strength = 2.0f;

    for (int i = 0; i < env->num_agents; i++) {
        if (i == agent_idx) continue;

        Agent* other = &env->agents[i];
        float dx = self->pos_x - other->pos_x;
        float dy = self->pos_y - other->pos_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < repulsion_radius * repulsion_radius && dist_sq > 0.0001f) {
            float dist = sqrtf(dist_sq);
            float repulsion = repulsion_strength / dist;

            self->pos_x += repulsion * dx;
            self->pos_y += repulsion * dy;
            check_agent_in_invalid_position_agent_position(env, agent_idx);

            other->pos_x -= repulsion * dx;
            other->pos_y -= repulsion * dy;
            check_agent_in_invalid_position_agent_position(env, i);
        }
    }
}

void move_agent(CTipToeEnv* env, int agent_idx, int action) {
    Agent* agent = &env->agents[agent_idx];

    int move_dir_x = 0, move_dir_y = 0;
    switch (action) {
        case ACTION_UP: move_dir_y = -1; break;
        case ACTION_UP_RIGHT: move_dir_y = -1; move_dir_x = 1; break;
        case ACTION_RIGHT: move_dir_x = 1; break;
        case ACTION_RIGHT_DOWN: move_dir_y = 1; move_dir_x = 1; break;
        case ACTION_DOWN: move_dir_y = 1; break;
        case ACTION_DOWN_LEFT: move_dir_y = 1; move_dir_x = -1; break;
        case ACTION_LEFT: move_dir_x = -1; break;
        case ACTION_LEFT_UP: move_dir_y = -1; move_dir_x = -1; break;
        case ACTION_NOOP: return;
    }

    agent->pos_x += move_dir_x * agent->velocity;
    agent->pos_y += move_dir_y * agent->velocity;

    // Check if too close to other agents, if so then push them apart
    apply_agent_repulsion(env, agent_idx);

    //
    check_agent_in_invalid_position_agent_position(env, agent_idx);

    // Reward: encourage upward motion
    float neg_reward = -0.00005f * agent->pos_y;
    env->rewards[agent_idx] += neg_reward;
    agent->total_episode_reward += neg_reward;

    // Check for episode end
    int tile_y = (int) (agent->pos_y / CELL_SIZE);

    if (tile_y == 0) { // if at finish line
        env->rewards[agent_idx] += 1.0f;
        env->dones[agent_idx] = 1;

        Log log = { .episode_return = env->rewards[agent_idx], .episode_length = env->current_step };
        add_log(env->log_buffer, &log);

        reset(env);
    }
}

void reset(CTipToeEnv* env) {
    env->current_step = 0;

    // Clear grid
    for (int i = 0; i < env->grid_size_x * env->grid_size_y; i++) {
        env->grid[i].state = TILE_UNKNOWN;
        env->grid[i].is_fake = true;
    }

    // Create path for agents to follow to finish line
    // Start from bottom row - 1 (right above the start zone)
    int path_x = rand() % env->grid_size_x;
    int y = env->grid_size_y - 1; // last grid row index

    while (y >= 0) {
        // Step 1: Lay down a horizontal row in a direction (left or right)
        bool go_right = rand() % 2;
        if (path_x == 0)
            go_right = true;
        else if (path_x == env->grid_size_x - 1)
            go_right = false;

        int length = 0 + rand() % 8; // Zigzag length: 0 to 9 cells

        for (int i = 0; i < length && path_x >= 0 && path_x < env->grid_size_x; i++) {
            env->grid[get_grid_index(env, path_x, y)].is_fake = false;

            // Move in horizontal direction
            path_x += go_right ? 1 : -1;
            if (path_x < 0) { path_x = 0; break; }
            if (path_x >= env->grid_size_x) { path_x = env->grid_size_x - 1; break; }
        }
        env->grid[get_grid_index(env, path_x, y)].is_fake = false;

        // Step 2: Drop down one row and place a single bridge tile
        y -= 1;
        if (y < 0) 
            break;

        env->grid[get_grid_index(env, path_x, y)].is_fake = false;

        y -= 1;
        if (y < 0) 
            break;
    }

    // Initialize agents
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent_position(env, i);
        env->agents[i].velocity = 2.5f;
        env->agents[i].total_episode_reward = 0;
    }
    
    compute_observations(env);
}

// Environment functions
void initialize_env(CTipToeEnv* env) {
    env->current_step = 0;
    env->grid = (Tile*)calloc(env->grid_size_x * env->grid_size_y, sizeof(Tile));
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->total_num_obs = env->num_agents * (2 + (env->grid_size_x * env->grid_size_y) + ((env->num_agents - 1) * 2)); // Ensure this matches in allocate()
    env->finish_y = (float)(CELL_SIZE);  // pixel offset of top of grid
    env->start_y = (float)((env->grid_size_y + 2) * CELL_SIZE);  // pixel offset into start zone

    reset(env);
}

void allocate(CTipToeEnv* env) {
    env->total_num_obs = env->num_agents * (2 + (env->grid_size_x * env->grid_size_y) + ((env->num_agents - 1) * 2)); // Ensure this matches in initialize_env()
    env->observations = (char*)calloc(env->total_num_obs, sizeof(char));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    initialize_env(env);
}

float calculate_avg_total_episode_reward(CTipToeEnv* env){
    // Calculate average total episode reward across all agents
    float avg_total_episode_reward = 0;
    for (int i = 0; i < env->num_agents; i++) {
        avg_total_episode_reward = env->agents[i].total_episode_reward;
    }
    avg_total_episode_reward /= env->num_agents;
    return avg_total_episode_reward;
}


void step(CTipToeEnv* env) {
    // Reset reward for each agent
    memset(env->rewards, 0, sizeof(float) * env->num_agents);
    memset(env->dones, 0, sizeof(unsigned char) * env->num_agents);

    for (int i = 0; i < env->num_agents; i++) {
        move_agent(env, i, env->actions[i]);
    }

    env->current_step++;
    if (env->current_step >= env->max_steps) 
    {
        memset(env->dones, 1, sizeof(unsigned char) * env->num_agents);

        Log log = {0};

        log.episode_length = env->current_step;

        log.episode_return = calculate_avg_total_episode_reward(env);

        add_log(env->log_buffer, &log);

        reset(env);
    }

    compute_observations(env);
}

void free_initialized(CTipToeEnv* env) {
    free(env->agents);
    free(env->grid);
}

void free_allocated(CTipToeEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_LINES = (Color){50, 50, 50, 255};

typedef struct Client {
    int window_width;
    int window_height;
    int header_offset;
    int cell_size;
    Texture2D agent_texture;
} Client;

// Initialize a rendering client
Client* make_client(CTipToeEnv* env) {
    Client* client = (Client*)malloc(sizeof(Client));
    client->cell_size = CELL_SIZE;
    client->header_offset = 60;
    client->window_width = env->grid_size_x * CELL_SIZE;
    client->window_height = env->grid_size_y * CELL_SIZE
                      + client->header_offset
                      + 2 * CELL_SIZE  // start zone
                      + CELL_SIZE;     // finish line

    InitWindow(client->window_width, client->window_height, "Tip Toe Environment");
    SetTargetFPS(60);

    client->agent_texture = LoadTexture("resources/puffers_128.png");

    return client;
}

// Render the environment
void render(Client* client, CTipToeEnv* env) {
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Header
    DrawText(
        TextFormat("Step: %d\nTotal Episode Reward: %.2f",
            env->current_step,
            calculate_avg_total_episode_reward(env)
        ),
        5, 2, 10, PUFF_WHITE
    );

    // Draw finish line (top)
    DrawRectangle(0,
                  client->header_offset,
                  env->grid_size_x * CELL_SIZE,
                  CELL_SIZE,
                  (Color){0, 200, 0, 80});  // Light green

    DrawRectangleLines(0,
                       client->header_offset,
                       env->grid_size_x * CELL_SIZE,
                       CELL_SIZE,
                       (Color){0, 255, 0, 255});

    bool debug_mode = true;

    // Draw grid tiles
    for (int y = 0; y < env->grid_size_y; y++) {
        for (int x = 0; x < env->grid_size_x; x++) {
            int screen_x = x * CELL_SIZE;
            int screen_y = y * CELL_SIZE + client->header_offset + CELL_SIZE;  // after finish line

            Tile* tile = &env->grid[get_grid_index(env, x, y)];

            Color tile_color;
            if (!debug_mode){
                if (tile->is_fake) {
                    tile_color = PUFF_RED;
                } else if (tile->state == TILE_REAL) {
                    tile_color = PUFF_CYAN;
                } else {
                    tile_color = (Color){80, 80, 80, 255};
                }
            }
            else{
                if (tile->state == TILE_REMOVED) {
                    tile_color = PUFF_RED;
                } 
                else if (tile->state == TILE_UNKNOWN) {
                    if (tile->is_fake){
                        tile_color = (Color){100, 10, 10, 255};
                    }
                    else{
                        tile_color = (Color){10, 100, 100, 255};
                    }
                }
                else if (tile->state == TILE_REAL) {
                    tile_color = PUFF_CYAN;
                }
                else {
                    tile_color = (Color){255, 255, 255, 255};
                }
            }

            DrawRectangle(screen_x, screen_y, CELL_SIZE, CELL_SIZE, tile_color);
            DrawRectangleLines(screen_x, screen_y, CELL_SIZE, CELL_SIZE, PUFF_LINES);
        }
    }

    // Draw start zone (bottom)
    DrawRectangle(0,
                  client->header_offset + CELL_SIZE + env->grid_size_y * CELL_SIZE,
                  env->grid_size_x * CELL_SIZE,
                  2 * CELL_SIZE,
                  PUFF_CYAN);

    DrawRectangleLines(0,
                       client->header_offset + CELL_SIZE + env->grid_size_y * CELL_SIZE,
                       env->grid_size_x * CELL_SIZE,
                       2 * CELL_SIZE,
                       (Color){255, 255, 0, 255});

    // Draw agents
    for (int i = 0; i < env->num_agents; i++) {
        Agent* agent = &env->agents[i];

        float screen_x = agent->pos_x;
        float screen_y = agent->pos_y + client->header_offset;

        Color agent_color = (env->do_human_control && i == 0)
                            ? (Color){255, 128, 128, 255}
                            : WHITE;

        DrawTexturePro(
            client->agent_texture,
            (Rectangle){0, 0, 128, 128},
            (Rectangle){
                screen_x, screen_y,
                client->cell_size, client->cell_size
            },
            (Vector2){client->cell_size / 2, client->cell_size / 2},
            0,
            agent_color
        );
                            
    }


    EndDrawing();
}


// Cleanup and free the rendering client
void close_client(Client* client) {
    UnloadTexture(client->agent_texture);
    CloseWindow();
    free(client);
}
