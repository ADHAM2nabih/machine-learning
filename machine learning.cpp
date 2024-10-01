#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


double predict(double x, double w, double b) { //y_predicted
    return w * x + b;
}


double computeCost(const vector<double>& X, const vector<double>& Y, double w, double b) { //ssr
    int n = X.size();
    double total_cost = 0;
    for (int i = 0; i < n; i++) {
        double y_pred = predict(X[i], w, b);
        total_cost += pow((y_pred - Y[i]), 2);
    }
    return total_cost / (2 * n);
}


void gradientDescent(vector<double>& X, vector<double>& Y, double& w, double& b, double alpha, int iterations) {

    int n = X.size();
    for (int i = 0; i < iterations; i++) {
        double dw = 0;
        double db = 0;


        for (int j = 0; j < n; j++) {               //partial derivatives of w and b
            double y_pred = predict(X[j], w, b);
            dw += (y_pred - Y[j]) * X[j];
            db += (y_pred - Y[j]);
        }

        dw/=n;
        db/=n;


        w -= alpha * dw;   // Update the weights and bias
        b -= alpha * db;


        if (i % 100 == 0) {
            double cost = computeCost(X, Y, w, b);
            cout << "Iteration " << i << " | Cost: " << cost << " | w: " << w << " | b: " << b << endl;
        }
    }
}

int main() {


    /*

    F w,b (x) = Wx + B = Y_predicted
    cost function  |  1/2n *  ( sum of [ y_predicted - yi ] ^2 )  = J(w,b)
    gradiant descent  |   W = W - ~ * d/dW J(W,B) |  B = B - ~ *d/dB J(W,B)


    */

    vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0,6.0};
    vector<double> Y = {1.5, 3.1, 4.9, 7.2, 9.0,11.0};


    double w = 0.0;
    double b = 0.0;
    double alpha = 0.01;  // Learning rate
    int iterations = 1000;


    gradientDescent(X, Y, w, b, alpha, iterations);


    cout << "W : " << w << endl;
    cout << "B : " << b << endl;

    return 0;
}
