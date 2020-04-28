extern crate rand;

use rand::Rng;

/* Math ------------------------------------------ */

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

/* Matrix ---------------------------------------- */

#[derive(Debug)]
struct Matrix {
    values: Vec<Vec<f64>>
}

impl Matrix { // refactor not self functions and self functions
    fn new(nb_rows: usize, nb_cols: usize) -> Matrix {
        Matrix {
            values: (0..nb_rows).map(|_| (0..nb_cols).map(|_| 0.0).collect()).collect()
        }
    }

    fn from_vec(vec: Vec<f64>) -> Matrix { // to refactor with map()
        let mut m: Matrix = Matrix::new(vec.len(), 1);
        for i in 0..vec.len() {
            m.values[i][0] = vec[i];
        }
        return m;
    }

    fn to_vec(&self) -> Vec<f64> { // to refactor with map()
        let mut v: Vec<f64> = Vec::new();
        for i in 0..self.values.len() {
            for j in 0..self.values[i].len() {
                v.push(self.values[i][j]);
            }
        }
        return v;
    }

    fn randomize(&mut self) {
        self.values = (0..self.values.len()).map(|y| (0..self.values[y].len()).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect()).collect()
    }

    fn add(&mut self, n: f64) {
        self.values = (0..self.values.len()).map(|y| (0..self.values[y].len()).map(|x| self.values[y][x] + n).collect()).collect()
    }

    fn scale(&mut self, n: f64) {
        self.values = (0..self.values.len()).map(|y| (0..self.values[y].len()).map(|x| self.values[y][x] * n).collect()).collect()
    }

    fn multiply(a: &Matrix, b: &Matrix) -> Matrix { // to refactor with map()
        let mut new: Matrix = Matrix::new(a.values.len(), b.values[0].len());

        for y in 0..new.values.len() {
            for x in 0..new.values[y].len() {
                let mut sum: f64 = 0.0;
                for i in 0..b.values.len() {
                    sum += a.values[y][i] * b.values[i][x];
                }
                new.values[y][x] = sum;
            }
        }
        return new;
    }
}

/* Neural Network -------------------------------- */

struct NeuralNetwork {
    weights_ih: Matrix,
    weights_ho: Matrix,
    bias_h: f64,
    bias_o: f64
}

impl NeuralNetwork {
    fn new(nb_inputs: usize, nb_hidden: usize, nb_outputs: usize) -> NeuralNetwork { // to refactor with map()
        let mut n = NeuralNetwork {
            weights_ih: Matrix::new(nb_hidden, nb_inputs),
            weights_ho: Matrix::new(nb_outputs, nb_hidden),
            bias_h: 1.0,
            bias_o: 1.0
        };
        n.weights_ih.randomize();
        n.weights_ho.randomize();
        return n;
    }

    fn feed_forward(&self, input_vec: Vec<f64>) -> Vec<f64> {
        let inputs = Matrix::from_vec(input_vec);

        // Generating hidden outputs
        let mut hidden = Matrix::multiply(&self.weights_ih, &inputs);
        hidden.add(self.bias_h);
        // Activation function
        hidden.values = (0..hidden.values.len()).map(|y| (0..hidden.values[y].len()).map(|x| sigmoid(hidden.values[y][x])).collect()).collect();

        // Generating output's outputs
        let mut outputs = Matrix::multiply(&self.weights_ho, &hidden);
        outputs.add(self.bias_o);
        // Activation function
        outputs.values = (0..outputs.values.len()).map(|y| (0..outputs.values[y].len()).map(|x| sigmoid(outputs.values[y][x])).collect()).collect();

        return outputs.to_vec();
    }
}

/* Functions ------------------------------------- */

fn main() {
    let neural_network: NeuralNetwork = NeuralNetwork::new(2, 4, 2);
    let inputs: Vec<f64> = vec!(1.0, 0.0);

    let output: Vec<f64> = neural_network.feed_forward(inputs);
    println!("{:?}", output);
}
