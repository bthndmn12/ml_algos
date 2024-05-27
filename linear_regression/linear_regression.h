#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <time.h>

typedef struct {
    float* train_x;
    float* train_y;
    float* test_x;
    float* test_y;
    int train_size;
    int test_size;
} Data;

typedef struct {
    float* train_inputs;
    float* train_outputs;
    float* weights;
    float bias;
    float learning_rate;
    int input_size;
} LinearRegression;

LinearRegression* create_model(int input_size);
void destroy_model(LinearRegression* model);
void train(LinearRegression* lr, Data* data, int epochs, float learning_rate);
float cost_function(LinearRegression* lr, float* predictions, float* targets, int size);
void forward_propagation(LinearRegression* lr, float* predictions, int size);
void update_weights(LinearRegression* lr, float* predictions, float* targets, int size);
float predict(LinearRegression* lr, float input);
float accuracy(LinearRegression* lr, Data* data);

#endif /* LINEAR_REGRESSION_H */


// #ifndef LINEAR_REGRESSION_H
// #define LINEAR_REGRESSION_H

// typedef struct {
//     float train_x;
//     float train_y;
//     float test_x;
//     float test_y;
// } Data;

// typedef struct{
//     float train_inputs;
//     float train_outputs;
//     float weights;
//     float bias;
//     float learning_rate;
// }LinearRegression;

// void train(LinearRegression* lr, Data* data, int epochs, float learning_rate);
// float cost_function(LinearRegression* lr, Data* data);
// float forward_propagation(LinearRegression* lr, Data* data);
// float update_weights(LinearRegression* lr, Data* data,float learning_rate);
// float predict(LinearRegression* lr, Data* data);
// float accuracy(LinearRegression* lr, Data* data);

// #endif /* LINEAR_REGRESSION_H */


