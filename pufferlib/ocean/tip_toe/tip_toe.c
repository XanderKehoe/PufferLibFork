#include <time.h>
#include "tip_toe.h"
#include "puffernet.h"

// Demo function for visualizing the TrashPickupEnv
void demo(int grid_size_x, int grid_size_y, int num_agents, int max_steps) {
    CTipToeEnv env = {
        .grid_size_x = grid_size_x,
        .grid_size_y = grid_size_y,
        .num_agents = num_agents,
        .max_steps = max_steps,
        .do_human_control = true
    };

    bool use_pretrained_model = false;

    Weights* weights;
    ConvLSTM* net;

    if (use_pretrained_model){
        weights = load_weights("resources/trash_pickup_weights.bin", 150374);
        int input_dim = 0; // TODO
        net = make_convlstm(weights, env.num_agents, input_dim, 5, 32, 128, 5);
    }

    allocate(&env);
    Client* client = make_client(&env);

    reset(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 12 == 0) {
            // Random actions for all agents
            for (int i = 0; i < env.num_agents; i++) {
                if (use_pretrained_model)
                {
                    for (int e = 0; e < env.total_num_obs; e++) {
                        net->obs[e] = env.observations[e];
                    }
                    forward_convlstm(net, net->obs, env.actions);    
                }
                else{
                    env.actions[i] = rand() % 9; // 9 possible actions
                }
                // printf("action: %d \n", env.actions[i]);
            }

            // Override human control actions
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[0] = ACTION_NOOP;

                // Handle keyboard input only for selected agent
                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    env.actions[0] = ACTION_UP;
                }
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    env.actions[0] = ACTION_LEFT;
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    env.actions[0] = ACTION_RIGHT;
                }
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) { 
                    env.actions[0] = ACTION_DOWN; 
                }
            }

            // Step the environment and render the grid
            step(&env);
            
        }
        tick++;

        render(client, &env);
    }

    free_convlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

// Performance test function for benchmarking
void performance_test() {
    long test_time = 10; // Test duration in seconds

    CTipToeEnv env = {
        .grid_size_x = 10,
        .grid_size_y = 10,
        .num_agents = 4,
        .max_steps = 150,
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    int inc = env.num_agents;
    while (time(NULL) - start < test_time) {
        for (int e = 0; e < env.num_agents; e++) {
            env.actions[e] = rand() % 5;
        }

        step(&env);
        i += inc;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}


// Main entry point
int main() {
    demo(10, 10, 3, 150); // Visual demo
    //performance_test(); // Uncomment for benchmarking
    return 0;
}
