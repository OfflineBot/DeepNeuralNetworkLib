# Deep Neural Network Library written in Rust

## How to use the Library (for now)
### Import it inside the Cargo.toml
+ dnn = { git = "https://github.com/OfflineBot/DeepNeuralNetworkLib" }
### Then code:
1. Define how many iterations it should do.
2. Define the learning rate.
3. Define how big the hidden layer size should be.
4. Select the Activation Function.
5. Add the input, hidden and new data as ndarray::Array2<f64>.
6. Add these to the dnn::train(iterations, learning_rate, hidden_layer_size, activation, input_data, output_data);
7. use nd::calculate_new_data with the new_data, the output.0 and output.1 from the last function.


## Work in Progress!!
The dnn folder contains a not finished library for Deep Neural Network calculations

## ToDo

-   multiple dynamic hidden layers
-   implementation of different activation functions
-   make everything dynamic
