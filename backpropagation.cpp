//Back Propagation ML Lab
//Shakeel Mohamed

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>

using namespace std;

float e = 2.71828182845904523536;

//Calculates sigmoid output
float sigmoid(float x){

   float output = 1/(1+(pow(e, -x)));
   return output;

}

//Calculate hidden neuron output
float hiddenNeuronOutput(float x[2], float w[2]){
   
   float output = 0;
   for (int i = 0; i < 2; ++i){
      output += x[i]*w[i];
      
   }
   output = sigmoid(output);
   cout << output << "\n" << endl;
   return output;
   
}

//Calculate final node outputs
float outputNodeOutput(float x1, float x2, float w[2]){

   float output = 0;
   
   output += x1 * w[0];
   
   output += x2 * w[1];

   output = sigmoid(output);
   cout << output << "\n" << endl;
   
   return output;
}

//Calculate error term for output node
float outputErrorTerm(float output, float target){

   float errorOutput = output * (1-output) * (target - output);
   cout << errorOutput << "\n" << endl;
   return errorOutput;
}
//Calculate hidden error term for output node
float hiddenErrorTerm(float output, float weight1, float weight2, float out1Error, float out2Error){
   
   float errorOutput = output * (1-output) * (weight1 * out1Error + weight2 * out2Error);
   //cout << errorOutput << "\n" << endl;
   return errorOutput;
}

int main (int argc, char *argv[]) {
   
   cout << "Back propagation ANN\n" << endl;
   
   //Inputs
   float inputs[2] = {0,1};
   
   //Weights into hidden nodes
   float weightsH1[2] = {-1, 0};
   float weightsH2[2] = {0, 1};
   
   //Weights into putputs
   float weightsO1[2] = {1, 0};
   float weightsO2[2] = {-1, 1};
   
   //Hidden node outputs
   float H1Output = 0;
   float H2Output = 0;
   
   //Output node outputs
   float OutputNode1 = 0;
   float OutputNode2 = 0;
   
   //Output targets
   float targetOutput1 = 1;
   float targetOutput2 = 0;
   
   //Error terms for 2 outputs
   float ErrorTerm1 = 0;
   float ErrorTerm2 = 0;
   
   //Error term for 2 hidden nodes
   float hiddenError1 = 0;
   float hiddenError2 = 0;
   
   //Learning rate
   float n = 1;
   
   //Calculate output for Hidden Node 1
   cout << "Hidden neuron 1 output: " << endl;
   H1Output = hiddenNeuronOutput(inputs, weightsH1);
   
   //Calculate output for Hidden Node 2
   cout << "Hidden neuron 2 output: " << endl;
   H2Output = hiddenNeuronOutput(inputs, weightsH2);
   
   //Calculate output for Output Node 1 and MSE
   cout << "Ouput Node 1 Output: " << endl;
   OutputNode1 = outputNodeOutput(H1Output, H2Output, weightsO1);
   cout << "Error for Output Node 1: " << endl;
   ErrorTerm1 = outputErrorTerm(OutputNode1, targetOutput1);
   
   //Calculate output for Output Node 1 and MSE
   cout << "Ouput Node 2 Output: " << endl;
   OutputNode2 = outputNodeOutput(H1Output, H2Output, weightsO2);
   cout << "Error for Output Node 2: " << endl;
   ErrorTerm2 = outputErrorTerm(OutputNode2, targetOutput2);
   
   //Calculate hidden node 
   hiddenError1 = hiddenErrorTerm(H1Output, weightsO1[0], weightsO2[0], ErrorTerm1, ErrorTerm2);
   hiddenError2 = hiddenErrorTerm(H2Output, weightsO1[1], weightsO2[1], ErrorTerm1, ErrorTerm2);
   
   cout << "New weights for output layer: " << endl;
   weightsO1[0] += (n * ErrorTerm1 * H1Output);
   weightsO1[1] += (n * ErrorTerm1 * H2Output);
   weightsO2[0] += (n * ErrorTerm2 * H1Output);
   weightsO2[1] += (n * ErrorTerm2 * H2Output);
   cout << "w11: " << weightsO1[0] << endl;
   cout << "w12: " << weightsO1[1] << endl;
   cout << "w21: " << weightsO2[0]  << endl;
   cout << "w22: " << weightsO2[1]  << endl;
   cout << "\n";
   
   cout << "Error for Hidden Node 1: " << endl;
   cout << hiddenError1 << endl;
   cout << "\n";
   cout << "Error for Hidden Node 2: " << endl;
   cout << hiddenError2 << endl;
   cout << "\n";
   
   cout << "New weights for hidden layer: " << endl;
   weightsH1[0] += (n * hiddenError1 * inputs[0]);
   weightsH1[1] += (n * hiddenError1 * inputs[1]);
   weightsH2[0] += (n * hiddenError2 * inputs[0]);
   weightsH2[1] += (n * hiddenError2 * inputs[1]);
   
   cout << "v11: " << weightsH1[0] << endl;
   cout << "v12: " << weightsH1[1] << endl;
   cout << "v21: " << weightsH2[0] << endl;
   cout << "v22: " << weightsH2[1] << endl;
   
   
   return 0;
   
}