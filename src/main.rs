extern crate rand;
extern crate rulinalg;

use rand::Rng;
use rand::seq::SliceRandom;
use rulinalg::matrix::{Matrix, BaseMatrixMut, BaseMatrix};

/* Math ------------------------------------------ */

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn dsigmoid(y: f64) -> f64 {
    return y * (1.0 - y);
}

/* Data ------------------------------------------ */

#[derive(Debug)]
struct Data {
    inputs: Matrix<f64>,
    targets: Matrix<f64>
}

/* Neural Network -------------------------------- */

struct NeuralNetwork {
    weights_ih: Matrix<f64>,
    weights_ho: Matrix<f64>,
    bias_h: Matrix<f64>,
    bias_o: Matrix<f64>,
    learning_rate: f64
}

impl NeuralNetwork {
    fn new(nb_inputs: usize, nb_hidden: usize, nb_outputs: usize) -> NeuralNetwork {
        NeuralNetwork {

            weights_ih: Matrix::new(nb_hidden, nb_inputs, (0..(nb_hidden * nb_inputs)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            weights_ho: Matrix::new(nb_outputs, nb_hidden, (0..(nb_outputs * nb_hidden)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_h: Matrix::new(nb_hidden, 1, (0..nb_hidden).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_o: Matrix::new(nb_outputs, 1, (0..nb_outputs).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            learning_rate: 0.1
        }
    }

    fn feedforward(&mut self, inputs: Matrix<f64>) -> Matrix<f64> {
        // Feed forward inputs -> hidden
        let mut hidden: Matrix<f64> = &self.weights_ih * &inputs;
        hidden = &hidden + &self.bias_h;
        hidden = hidden.apply(&sigmoid);
        // Feed forward hidden -> outputs
        let mut outputs: Matrix<f64> = &self.weights_ho * &hidden;
        outputs = &outputs + &self.bias_o;
        outputs = outputs.apply(&sigmoid);
        return outputs;
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) { 
        // Feed forward inputs -> hidden
        let mut hidden: Matrix<f64> = &self.weights_ih * inputs;
        hidden = &hidden + &self.bias_h;
        hidden = hidden.apply(&sigmoid);
        // Feed forward hidden -> outputs
        let mut outputs: Matrix<f64> = &self.weights_ho * &hidden;
        outputs = &outputs + &self.bias_o;
        outputs = outputs.apply(&sigmoid);

        // Computing outputs errors
        let output_errors: Matrix<f64> = targets - &outputs;
        // Computing outputs gradient
        let mut outputs_gradients: Matrix<f64> = outputs.apply(&dsigmoid);
        outputs_gradients = outputs_gradients.elemul(&output_errors);
        outputs_gradients = &outputs_gradients * &self.learning_rate;
        // Computing hidden -> outputs deltas
        let hidden_transposed: Matrix<f64> = hidden.transpose();
        let weights_ho_deltas: Matrix<f64> = &outputs_gradients * hidden_transposed;
        // Update hidden -> outputs weights
        self.weights_ho = &self.weights_ho + &weights_ho_deltas;
        // Update outputs bias
        self.bias_o = &self.bias_o + outputs_gradients;

        // Computing hidden errors
        let weights_ho_transposed: Matrix<f64> = self.weights_ho.transpose();
        let hidden_errors: Matrix<f64> = &weights_ho_transposed * &output_errors;
        // Computing hidden gradient
        let mut hidden_gradients: Matrix<f64> = hidden.apply(&dsigmoid);
        hidden_gradients = hidden_gradients.elemul(&hidden_errors);
        hidden_gradients = &hidden_gradients * &self.learning_rate;
        // Computing inputs -> hidden deltas
        let inputs_transposed: Matrix<f64> = inputs.transpose(); 
        let weights_ih_deltas: Matrix<f64> = &hidden_gradients * inputs_transposed;
        // Update inputs -> hidden weights
        self.weights_ih = &self.weights_ih + &weights_ih_deltas;
        // Update outputs bias
        self.bias_h = &self.bias_h + hidden_gradients;
    }
}

/* Functions ------------------------------------- */
fn color_predictor((r, g, b): (i32, i32, i32)) -> String {
    return if r + g + b > 300 { String::from("black") } else { String::from("white") };
}

fn neuralnet_color_predictor(neuralnet: &mut NeuralNetwork, (r, g, b): (i32, i32, i32)) -> String {
    let color_inputs: Matrix<f64> = Matrix::new(3, 1, vec!(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0));

    let result: Matrix<f64> = neuralnet.feedforward(color_inputs);

    return if result.data()[0] > result.data()[1] { String::from("black") } else { String::from("white") };
}

fn main() {
    let mut neuralnet: NeuralNetwork = NeuralNetwork::new(3, 3, 2);

    for _ in 0..10000 {
        let color: (i32, i32, i32) = (rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256));
        let inputs: Matrix<f64> = Matrix::new(3, 1, vec!(color.0 as f64 / 255.0, color.1 as f64 / 255.0, color.2 as f64 / 255.0));
        let answer = color_predictor(color);
        
        let targets_tuple: (f64, f64) = if answer == "black" { (1.0, 0.0) } else { (0.0, 1.0) };

        let targets: Matrix<f64> = Matrix::new(2, 1, vec!(targets_tuple.0, targets_tuple.1));
        &neuralnet.train(&inputs, &targets);
    }

    // Testing result
    for _ in 0..10 {
        let color: (i32, i32, i32) = (rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256));
        println!("color: {:?}", color);
        println!("predictor: {}", color_predictor(color));
        println!("neuralnet: {}\n", neuralnet_color_predictor(&mut neuralnet, color));
    }
}
