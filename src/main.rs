extern crate rand;
extern crate rulinalg;

use rand::Rng;
use rulinalg::matrix::Matrix;

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
            weights_ih: Matrix::new(nb_hidden, nb_inputs, (0..nb_hidden * nb_inputs).map(|v| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            weights_ho: Matrix::new(nb_outputs, nb_hidden, (0..nb_outputs * nb_hidden).map(|v| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_h: Matrix::new(nb_hidden, 1, (0..nb_hidden).map(|v| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_o: Matrix::new(nb_outputs, 1, (0..nb_hidden).map(|v| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>())
        }
    }

    fn train(&mut self, inputs: Vec<f64>, targets: Vec<f64>) { 
        // Feed forward inputs -> hidden
        let mut hidden = Matrix::product(&self.weights_ih, &inputs);
        hidden.add(&self.bias_h);
        // Activation function
        hidden.values = (0..hidden.values.len()).map(|y| (0..hidden.values[y].len()).map(|x| sigmoid(hidden.values[y][x])).collect()).collect();

        // Feed forward hidden -> outputs
        let mut outputs = Matrix::product(&self.weights_ho, &hidden);
        outputs.add(&self.bias_o);
        // Activation function
        outputs.values = (0..outputs.values.len()).map(|y| (0..outputs.values[y].len()).map(|x| sigmoid(outputs.values[y][x])).collect()).collect();

        // Computing the outputs errors
        let output_errors: Matrix = Matrix::sub(&targets, &outputs);

        //outputs.values = (0..outputs.values.len()).map(|y| (0..outputs.values[y].len()).map(|x| dsigmoid(outputs.values[y][x])).collect()).collect();
        //outputs.value = Matix::multiply();

        // Computing hidden errors
        let weights_ho_transposed: Matrix = Matrix::transpose(&self.weights_ho);
        let hidden_errors: Matrix = Matrix::product(&weights_ho_transposed, &output_errors);

        //println!("{:?}\n{:?}\n{:?}", outputs, targets, output_errors);
    }
}

/* Functions ------------------------------------- */ 

fn main() {
    let mut neuralnet: NeuralNetwork = NeuralNetwork::new(2, 2, 2);
    let inputs = Matrix::new(2, 1, vec!(1.0, 0.0));
    let targets = Matrix::new(2, 1, vec!(1.0, 0.0));

    neuralnet.train(inputs, targets);
}
