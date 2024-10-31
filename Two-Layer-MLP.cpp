#include<iostream>
#include<math.h>

using namespace std;

class MLP
{
private:
    int input_dim, hidden_dim, output_dim;
    double **W1, **W2;
    double *b1, *b2;
    double **grad_W1, **grad_W2;
    double *grad_b1, *grad_b2;
    double *previous_input;
    double *previous_mid_layer;
    double *previous_output;
    double sigmoid(double);
    double grad_sigmoid(double);
public:
    MLP(int input_features_dim, int hidden_features_dim, int output_features_dim);
    ~MLP();
    double *forward(double *x);
    double mseLoss(double *y, double *y_);
    void backward(double *y_);
    void clear_gradients();
    void update_weights(double lr);
};

MLP::MLP(int input_features_dim, int hidden_features_dim, int output_features_dim)
{
    input_dim = input_features_dim;
    hidden_dim = hidden_features_dim;
    output_dim = output_features_dim;

    W1 = new double *[hidden_dim];
    grad_W1 = new double *[hidden_dim];

    for(int i = 0; i < hidden_dim; ++i)
    {
        W1[i] = new double [input_dim];
        grad_W1[i] = new double [input_dim];
    }
    
    b1 = new double [hidden_dim];
    grad_b1 = new double [hidden_dim];
    
    W2 = new double *[output_dim];
    grad_W2 = new double *[output_dim];

    for(int i = 0; i < output_dim; ++i)
    {
        W2[i] = new double [hidden_dim];
        grad_W2[i] = new double [hidden_dim];
    }
    
    b2 = new double [output_dim];
    grad_b2 = new double [output_dim];

    previous_input = new double [input_dim];
    previous_mid_layer = new double [hidden_dim];
    previous_output = new double [output_dim];
}

MLP::~MLP()
{
    int i;
    for(i = 0; i < hidden_dim; ++i)
    {
        delete(W1[i]);
        delete(grad_W1[i]);
    }
    for(i = 0; i < output_dim; ++i)
    {
        delete(W2[i]);
        delete(grad_W2[i]);
    }
    
    delete(W1);
    delete(grad_W1);
    delete(W2);
    delete(grad_W2);
    delete(b1);
    delete(grad_b1);
    delete(b2);
    delete(grad_b2);
    delete(previous_input);
    delete(previous_mid_layer);
    delete(previous_output);
    W1 = W2 = NULL;
    b1 = b2 = previous_input = previous_mid_layer = previous_output = NULL;
}

double MLP::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double MLP::grad_sigmoid(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double* MLP::forward(double *x)
{
    int i, j;
    double tempSum;
    double *hidden_representation = new double [hidden_dim];
    double *output = new double [output_dim];

    for(i = 0; i < input_dim; ++i)
        previous_input[i] = x[i];
    
    //Apply W1 on input, add b1------>apply sigmoid
    for(i = 0; i < hidden_dim; ++i)
    {
        tempSum = 0.0;
        for(j = 0; j < input_dim; ++j)
            tempSum += (W1[i][j] * x[j]);
        previous_mid_layer[i] = hidden_representation[i] = sigmoid(tempSum + b1[i]);
    }
    //Apply W2 on mid layer and add b2
    for(i = 0; i < output_dim; ++i)
    {
        tempSum = 0.0;
        for(j = 0; j < hidden_dim; ++j)
            tempSum += (W2[i][j] * hidden_representation[j]);
        previous_output[i] = output[i] = tempSum + b2[i];
    }

    hidden_representation = NULL;
    return output;
}

void MLP::clear_gradients()
{
    int i, j;

    for(i = 0; i < hidden_dim; ++i)
    {
        for(j = 0; j < input_dim; ++j)
            grad_W1[i][j] = 0.0;

        grad_b1[i] = 0.0;
    }
    for(i = 0; i < output_dim; ++i)
    {
        for(j = 0; j < hidden_dim; ++j)
            grad_W2[i][j] = 0.0;
        grad_b2[i] = 0.0;
    }
}

double MLP::mseLoss(double *y, double *y_)
{
    double error_per_feature = 0.0;
    double loss = 0.0;

    for(int i = 0; i < output_dim; ++i)
    {
        error_per_feature = y[i] - y_[i];
        loss += (error_per_feature * error_per_feature);
    }
    return 0.5 * loss;
}

void MLP::backward(double *y_)
{
    double *x = previous_input;
    double *y2 = previous_output;
    double *a = previous_mid_layer;

    double tempSum;
    double *loss_diff_term = new double [output_dim];
    double *a_ = new double [hidden_dim];


    int i, j, k;

    //Calculate the vector (y2 - y_)
    for(i = 0; i < output_dim; ++i)
        loss_diff_term[i] = y2[i] - y_[i];

    //Calculate gradient of middle layer : s'(x) = s(x){1 - s(x)}
    for(i = 0; i < hidden_dim; ++i)
        a_[i] = a[i] * (1.0 - a[i]);

    //Calculate gradients for b2
    for(i = 0; i < output_dim; ++i)
        grad_b2[i] += loss_diff_term[i];
    
    //Calculate gradients for W2
    for(i = 0; i < output_dim; ++i)
        for(j = 0; j < hidden_dim; ++j)
            grad_W2[i][j] += (loss_diff_term[i] * a[j]);
    
    //Calculate gradients for b1
    for(i = 0; i < hidden_dim; ++i)
    {
        tempSum = 0.0;
        for(k = 0; k < output_dim; ++k)
            tempSum += (loss_diff_term[k] * W2[k][i]);
        grad_b1[i] -= (tempSum * a_[i]);
    }

    //Calculate gradients for W1
    for(i = 0; i < hidden_dim; ++i)
        for(j = 0; j < input_dim; ++j)
        {
            tempSum = 0.0;
            for(k = 0; k < output_dim; ++k)
                tempSum += (loss_diff_term[k] * W2[k][i]);
            tempSum *= (a_[i] * x[j]);
            grad_W1[i][j] += tempSum;
        }
}

void MLP::update_weights(double lr = 1e-4)
{
    int i, j;

    //Update W1
    for(i = 0; i < hidden_dim; ++i)
    {
        for(j = 0; j < input_dim; ++j)
            W1[i][j] -= (lr * grad_W1[i][j]);
        b1[i] -= (lr * grad_b1[i]);
    }
    //Update W2 and b2
    for(i = 0; i < output_dim; ++i)
    {
        for(j = 0; j < hidden_dim; ++j)
            W2[i][j] -= (lr * grad_W2[i][j]);
        b2[i] -= (lr * grad_b2[i]);
    }
}

int main()
{
    int input_dim = 256;
    int hidden_dim = 1024;
    int output_dim = 10;
    //--------------------------------------------------------------------------
    int num_data_points = 500;

    double **X = new double *[num_data_points];
    for(int i = 0; i < num_data_points; ++i)
        X[i] = new double [input_dim];
    
    double **Y = new double *[num_data_points];
    for(int i = 0; i < num_data_points; ++i)
        Y[i] = new double [output_dim];
    //--------------------------------------------------------------------------

    int num_epochs = 1000;
    double learning_rate = 1e-4;

    double *output;
    double per_epoch_loss;

    class MLP model(input_dim, hidden_dim, output_dim);

    cout<<"Starting Training\n";
    for(int epoch = 0; epoch < num_epochs; ++epoch)
    {
        model.clear_gradients();
        per_epoch_loss = 0.0;

        for(int i = 0; i < num_data_points; ++i)
        {
            output = model.forward(X[i]);
            per_epoch_loss += model.mseLoss(output, Y[i]);
            model.backward(Y[i]);
        }

        model.update_weights(learning_rate);
        cout<<"Epoch : "<<epoch+1<<"\t||\tLoss = "<<per_epoch_loss<<endl;
    }
    cout<<"\n\nTraining Complete\n";
    return 0;
}