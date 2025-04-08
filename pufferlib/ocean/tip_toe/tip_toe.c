#include <time.h>
#include "tip_toe.h"
#include "puffernet.h"

// Define model struct for TipToe
typedef struct TipToeNet TipToeNet;
struct TipToeNet {
    float* obs_grid;   // (num_agents, 1, grid_y, grid_x)
    float* obs_vector; // (num_agents, 2 + (num_agents - 1)*2)
    Conv2D* conv1;
    ReLU* relu1;
    Conv2D* conv2;
    ReLU* relu2;
    Linear* vector_fc;
    CatDim1* cat;
    ReLU* relu3;
    Linear* mlp;
    ReLU* relu4;
    Linear* actor;
    Linear* value;
    Multidiscrete* multidiscrete;
};

TipToeNet* init_tiptoe_net(Weights* weights, int num_agents, int grid_x, int grid_y, int action_dim) {
    TipToeNet* net = calloc(1, sizeof(TipToeNet));
    int hidden = 128;
    int cnn_channels = 32;
    int grid_cells = grid_x * grid_y;
    int vector_dim = 2 + (num_agents - 1) * 2;

    net->obs_grid = calloc(num_agents * grid_cells, sizeof(float));
    net->obs_vector = calloc(num_agents * vector_dim, sizeof(float));

    net->conv1 = make_conv2d(weights, num_agents, grid_y, grid_x, 1, cnn_channels, 3, 1);
    net->relu1 = make_relu(num_agents, cnn_channels * grid_y * grid_x);
    net->conv2 = make_conv2d(weights, num_agents, grid_y, grid_x, cnn_channels, cnn_channels, 3, 1);
    net->relu2 = make_relu(num_agents, cnn_channels * grid_y * grid_x);

    net->vector_fc = make_linear(weights, num_agents, vector_dim, hidden);
    net->cat = make_cat_dim1(num_agents, cnn_channels * grid_y * grid_x, hidden);
    net->relu3 = make_relu(num_agents, cnn_channels * grid_y * grid_x + hidden);
    net->mlp = make_linear(weights, num_agents, cnn_channels * grid_y * grid_x + hidden, hidden);
    net->relu4 = make_relu(num_agents, hidden);
    net->actor = make_linear(weights, num_agents, hidden, action_dim);
    net->value = make_linear(weights, num_agents, hidden, 1);

    int logit_sizes[1] = {action_dim};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);

    return net;
}

void free_tiptoe_net(TipToeNet* net) {
    free(net->obs_grid);
    free(net->obs_vector);
    free(net->conv1);
    free(net->relu1);
    free(net->conv2);
    free(net->relu2);
    free(net->vector_fc);
    free(net->cat);
    free(net->relu3);
    free(net->mlp);
    free(net->relu4);
    free(net->actor);
    free(net->value);
    free(net->multidiscrete);
    free(net);
}

void forward_tiptoe(TipToeNet* net, char* observations, int* actions, int num_agents, int grid_x, int grid_y) {
    int grid_cells = grid_x * grid_y;
    int vector_dim = 2 + (num_agents - 1) * 2;

    for (int i = 0; i < num_agents; i++) {
        int offset = i * (grid_cells + vector_dim);
        for (int j = 0; j < grid_cells; j++) {
            net->obs_grid[i * grid_cells + j] = observations[offset + j] / 255.0f;
        }
        for (int j = 0; j < vector_dim; j++) {
            net->obs_vector[i * vector_dim + j] = observations[offset + grid_cells + j] / 255.0f;
        }
    }

    conv2d(net->conv1, net->obs_grid);
    relu(net->relu1, net->conv1->output);
    conv2d(net->conv2, net->relu1->output);
    relu(net->relu2, net->conv2->output);

    linear(net->vector_fc, net->obs_vector);
    cat_dim1(net->cat, net->conv2->output, net->vector_fc->output);
    relu(net->relu3, net->cat->output);
    linear(net->mlp, net->relu3->output);
    relu(net->relu4, net->mlp->output);
    linear(net->actor, net->relu4->output);
    linear(net->value, net->relu4->output);

    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

// Demo function for visualizing the TrashPickupEnv
void demo(int grid_size_x, int grid_size_y, int num_agents, int max_steps) {
    CTipToeEnv env = {
        .grid_size_x = grid_size_x,
        .grid_size_y = grid_size_y,
        .num_agents = num_agents,
        .max_steps = max_steps,
        .do_human_control = true
    };

    bool use_pretrained_model = true;

    Weights* weights;
    TipToeNet* net;

    if (use_pretrained_model){
        weights = load_weights("resources/tip_toe_weights.bin", 571754);
        net = init_tiptoe_net(weights, num_agents, grid_size_x, grid_size_y, 9);
    }

    allocate(&env);
    Client* client = make_client(&env);

    reset(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 3 == 0) {
            if (use_pretrained_model)
            {
                forward_tiptoe(net, env.observations, env.actions, num_agents, grid_size_x, grid_size_y);   
            }
            else{
                // Random actions for all agents
                for (int i = 0; i < env.num_agents; i++) {
                    env.actions[i] = rand() % 9; // 9 possible actions
                }
            }

            // Override human control actions
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                env.actions[0] = ACTION_NOOP;

                // Handle keyboard input only for selected agent
                // TODO - handle UP-RIGHT, LEFT_DOWN, LEFT_RIGHT, AND RIGHT_UP
                if ((IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) && (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))){
                    env.actions[0] = ACTION_UP_RIGHT;
                }
                else if ((IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) && (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))){
                    env.actions[0] = ACTION_RIGHT_DOWN;
                }
                else if ((IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) && (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))){
                    env.actions[0] = ACTION_DOWN_LEFT;
                }
                else if ((IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) && (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))){
                    env.actions[0] = ACTION_LEFT_UP;
                }
                else if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    env.actions[0] = ACTION_UP;
                }
                else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    env.actions[0] = ACTION_LEFT;
                }
                else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    env.actions[0] = ACTION_RIGHT;
                }
                else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) { 
                    env.actions[0] = ACTION_DOWN; 
                }
            }

            // Step the environment and render the grid
            step(&env);
            
        }
        tick++;

        render(client, &env);
    }

    if (use_pretrained_model) {
        free_tiptoe_net(net);
        free(weights);
    }
    free_allocated(&env);
    close_client(client);
}

// Performance test function for benchmarking
void performance_test() {
    long test_time = 10; // Test duration in seconds

    CTipToeEnv env = {
        .grid_size_x = 10,
        .grid_size_y = 10,
        .num_agents = 10,
        .max_steps = 600,
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    int inc = env.num_agents;
    while (time(NULL) - start < test_time) {
        for (int e = 0; e < env.num_agents; e++) {
            env.actions[e] = rand() % 9;
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
    demo(10, 10, 10, 600); // Visual demo
    // performance_test(); // Uncomment for benchmarking - SPS is 400k-ish on my computer
    return 0;
}
