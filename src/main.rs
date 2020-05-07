extern crate rand;
extern crate rulinalg;

use rand::Rng;
use rulinalg::matrix::{Matrix, BaseMatrixMut, BaseMatrix};

/* Training data --------------------------------- */

struct Data {
    inputs: Matrix<f64>,
    targets: Matrix<f64>
}

/* Math ------------------------------------------ */

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn dsigmoid(y: f64) -> f64 {
    return y * (1.0 - y);
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

        // println!("o: {:?}\nt: {:?}\ne: {:?}", outputs, targets, output_errors);
    }
}

/* Functions ------------------------------------- */

fn main() {
    let mut neuralnet: NeuralNetwork = NeuralNetwork::new(2, 2, 1);
    let training_data: Vec<Data> = vec![
        Data {
            inputs: Matrix::new(2, 1, vec!(0.0, 1.0)),
            targets: Matrix::new(1, 1, vec!(1.0))
        },
        Data {
            inputs: Matrix::new(2, 1, vec!(1.0, 0.0)),
            targets: Matrix::new(1, 1, vec!(1.0))
        },
        Data {
            inputs: Matrix::new(2, 1, vec!(0.0, 0.0)),
            targets: Matrix::new(1, 1, vec!(0.0))
        },
        Data {
            inputs: Matrix::new(2, 1, vec!(1.0, 1.0)),
            targets: Matrix::new(1, 1, vec!(0.0))
        }
    ];
    
    for _ in 0..50000 {
        let data: &Data = &training_data[rand::thread_rng().gen_range(0, training_data.len())];
        neuralnet.train(&data.inputs, &data.targets);
    }

    println!("0 xor 0: {}", neuralnet.feedforward(Matrix::new(2, 1, vec!(0.0, 0.0))));
    println!("1 xor 1: {}", neuralnet.feedforward(Matrix::new(2, 1, vec!(1.0, 1.0))));
    println!("0 xor 1: {}", neuralnet.feedforward(Matrix::new(2, 1, vec!(0.0, 1.0))));
    println!("1 xor 0: {}", neuralnet.feedforward(Matrix::new(2, 1, vec!(1.0, 0.0))));
}
