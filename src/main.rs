extern crate rand;
extern crate rulinalg;

use rand::Rng;
use rulinalg::matrix::{Matrix, BaseMatrixMut, BaseMatrix};

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
    bias_o: Matrix<f64>
}

impl NeuralNetwork {
    fn new(nb_inputs: usize, nb_hidden: usize, nb_outputs: usize) -> NeuralNetwork {
        NeuralNetwork {

            weights_ih: Matrix::new(nb_hidden, nb_inputs, (0..(nb_hidden * nb_inputs)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            weights_ho: Matrix::new(nb_outputs, nb_hidden, (0..(nb_outputs * nb_hidden)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_h: Matrix::new(nb_hidden, 1, (0..nb_hidden).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_o: Matrix::new(nb_outputs, 1, (0..nb_outputs).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>())
            
        }
    }

    fn train(&mut self, inputs: Matrix<f64>, targets: Matrix<f64>) { 
        // Feed forward inputs -> hidden
        let mut hidden: Matrix<f64> = &self.weights_ih * &inputs;
        hidden = &hidden + &self.bias_h;
        hidden = hidden.apply(&sigmoid);

        // Feed forward hidden -> outputs
        let mut outputs: Matrix<f64> = &self.weights_ho * &hidden;
        outputs = &outputs + &self.bias_o;
        outputs = outputs.apply(&sigmoid);

        // Computing outputs errors
        let output_errors: Matrix<f64> = &targets - &outputs;

        // Computing hidden errors
        let weights_ho_transposed: Matrix<f64> = self.weights_ho.transpose();
        let hidden_errors: Matrix<f64> = &weights_ho_transposed * &output_errors;

        // println!("o: {:?}\nt: {:?}\ne: {:?}", outputs, targets, output_errors);
    }
}

/* Functions ------------------------------------- */ 

fn main() {
    let mut neuralnet: NeuralNetwork = NeuralNetwork::new(2, 2, 2);
    let inputs = Matrix::new(2, 1, vec!(1.0, 0.0));
    let targets = Matrix::new(2, 1, vec!(1.0, 0.0));

    neuralnet.train(inputs, targets);
}
