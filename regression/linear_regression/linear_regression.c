#include "linear_regression.h"

#define LEARNING_RATE 0.0001
#define EPOCHS 50

LinearRegression* create_model(int input_size) {
    LinearRegression* model = (LinearRegression*)malloc(sizeof(LinearRegression));
       
    if (model == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    model->train_inputs = (float*)malloc(input_size * sizeof(float));
    model->train_outputs = (float*)malloc(input_size * sizeof(float));
    model->weights = (float*)malloc(input_size * sizeof(float));

    if (model->train_inputs == NULL || model->train_outputs == NULL || model->weights == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    model->bias = 0.0;
    model->learning_rate = LEARNING_RATE;
    model->input_size = input_size;

    for (int i = 0; i < input_size; i++) {
        model->weights[i] = 0.0;
    }

    return model;
}

void destroy_model(LinearRegression* model) {
    free(model->train_inputs);
    free(model->train_outputs);
    free(model->weights);
    free(model);
}

// float cost_function(LinearRegression* lr, float* predictions, float* targets, int size) {
  
//     float error = 0.0;
//     for (int i = 0; i < size; i++) {
//         // Scale data if needed (e.g., divide by FLT_MAX)
//         float scaled_prediction = predictions[i] / FLT_MAX;
//         float scaled_target = targets[i] / FLT_MAX;
//         // Use fma for squaring (if available)
//         error += fma(scaled_prediction - scaled_target, scaled_prediction - scaled_target, error); 
//     }

//     return error / size;
// }
float cost_function(LinearRegression* lr, float* predictions, float* targets, int size) {
    if (size == 0) {
        return 0.0; 
    }

    float error = 0.0;
    for (int i = 0; i < size; i++) {
        
        float scaled_prediction = predictions[i] / FLT_MAX;
        float scaled_target = targets[i] / FLT_MAX;
        
        error += fma(scaled_prediction - scaled_target, scaled_prediction - scaled_target, error); 
    }

    return error / size;
}

void forward_propagation(LinearRegression* lr, float* predictions, int size) {
    for (int i = 0; i < size; i++) {
        predictions[i] = 0.0;
        for (int j = 0; j < lr->input_size; j++) {
            predictions[i] += lr->train_inputs[i * lr->input_size + j] * lr->weights[j];
        }
        predictions[i] += lr->bias;
    }
}


void update_weights(LinearRegression* lr, float* predictions, float* targets, int size) {
    if (size == 0) {
        return; 
    }

    float* gradients = (float*)calloc(lr->input_size, sizeof(float));
    float bias_gradient = 0.0;

    
    bool all_inputs_zero = true; 
    for (int i = 0; i < size * lr->input_size; i++) {
        if (lr->train_inputs[i] != 0.0) {
            all_inputs_zero = false;
            break;
        }
    }

    if (all_inputs_zero) {
    
        fprintf(stderr, "Error: All training inputs are zero, cannot update weights.\n");
        return; 
    }
    for (int i = 0; i < size; i++) {
        float error = predictions[i] - targets[i];
        for (int j = 0; j < lr->input_size; j++) {
            gradients[j] += error * lr->train_inputs[i * lr->input_size + j];
        }
        bias_gradient += error;
    }

    for (int j = 0; j < lr->input_size; j++) {
        lr->weights[j] -= lr->learning_rate * gradients[j] / size;
    }
    lr->bias -= lr->learning_rate * bias_gradient / size;

    free(gradients);
}

// void train(LinearRegression* lr, Data* data, int epochs, float learning_rate) {
//     lr->learning_rate = learning_rate;

//     for (int epoch = 0; epoch < epochs; epoch++) {
//         float* predictions = (float*)malloc(data->train_size * sizeof(float));

//         forward_propagation(lr, predictions, data->train_size);

//         float cost = cost_function(lr, predictions, data->train_y, data->train_size);
//         printf("Cost at epoch %d: %f\n", epoch, cost);

//         update_weights(lr, predictions, data->train_y, data->train_size);

//         free(predictions);
//     }
// }

void train(LinearRegression* lr, Data* data, int epochs, float learning_rate) {
    lr->learning_rate = learning_rate;
    
    time_t start_time = time(NULL);  
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float* predictions = (float*)malloc(data->train_size * sizeof(float));

        forward_propagation(lr, predictions, data->train_size);

        float cost = cost_function(lr, predictions, data->train_y, data->train_size);
        printf("Cost at epoch %d: %f\n", epoch, cost);

        update_weights(lr, predictions, data->train_y, data->train_size);

        free(predictions);

        if ((epoch + 1) % 5 == 0) {
            time_t current_time = time(NULL);
            printf("Epoch: %d, Time Elapsed: %ld seconds\n", epoch + 1, current_time - start_time);
        }
    }
}

float predict(LinearRegression* lr, float input) {
    float prediction = lr->bias;

    for (int i = 0; i < lr->input_size; i++) {
        prediction += input * lr->weights[i];
    }

    return prediction;
}

float accuracy(LinearRegression* lr, Data* data) {
    float* predictions = (float*)malloc(data->test_size * sizeof(float));

    for (int i = 0; i < data->test_size; i++) {
        predictions[i] = predict(lr, data->test_x[i]);
    }

    float cost = cost_function(lr, predictions, data->test_y, data->test_size);
    float mean_target = 0.0;

    for (int i = 0; i < data->test_size; i++) {
        mean_target += data->test_y[i];
    }

    mean_target /= data->test_size;
    float variance = 0.0;

    for (int i = 0; i < data->test_size; i++) {
        variance += pow(data->test_y[i] - mean_target, 2);
    }

    free(predictions);

    return 1 - (cost / variance);
}

Data* load_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    int train_size, test_size;
    fscanf(file, "%d,%d\n", &train_size, &test_size);

    Data* data = (Data*)malloc(sizeof(Data));
    if (data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    data->train_x = (float*)malloc(train_size * sizeof(float));
    data->train_y = (float*)malloc(train_size * sizeof(float));
    data->test_x = (float*)malloc(test_size * sizeof(float));
    data->test_y = (float*)malloc(test_size * sizeof(float));

    if (data->train_x == NULL || data->train_y == NULL || data->test_x == NULL || data->test_y == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }

    data->train_size = train_size;
    data->test_size = test_size;

    for (int i = 0; i < train_size; i++) {
        fscanf(file, "%f,%f\n", &data->train_x[i], &data->train_y[i]);
    }

    for (int i = 0; i < test_size; i++) {
        fscanf(file, "%f,%f\n", &data->test_x[i], &data->test_y[i]);
    }

    fclose(file);

    return data;
}

void free_data(Data* data) {
    free(data->train_x);
    free(data->train_y);
    free(data->test_x);
    free(data->test_y);
    free(data);
}

int main() {
    Data* data = load_data("data_for_lr.csv");

    LinearRegression* model = create_model(1);

    model->train_inputs = data->train_x;
    model->train_outputs = data->train_y;

    train(model, data, EPOCHS, LEARNING_RATE);

    float accuracy_score = accuracy(model, data);
    printf("Accuracy: %f\n", accuracy_score);

    float prediction = predict(model, 24.0);
    printf("Predicted value: %f\n", prediction);

    destroy_model(model);
    free_data(data);

    return 0;
}