
/* Matrix ---------------------------------------- */

#[derive(Debug)]
struct Matrix {
    values: Vec<Vec<f64>>
}

impl Matrix {
    fn new(nb_rows: u32, nb_cols: u32) -> Matrix {
        Matrix {
            values: (0..nb_rows).map(|_| (0..nb_cols).map(|_| 0.0).collect()).collect()
        }
    }

    fn add(&mut self, n: f64) {
        self.values = (0..self.values.len()).map(|y| (0..self.values[y].len()).map(|x| self.values[y][x] + n).collect()).collect()
    }

    fn multiply(&mut self, n: f64) {
        self.values = (0..self.values.len()).map(|y| (0..self.values[y].len()).map(|x| self.values[y][x] * n).collect()).collect()
    }
}

/* Neural Network -------------------------------- */

struct NeuralNetwork {
    nb_inputs: u32,
    nb_hidden: u32,
    nb_outputs: u32,
}

impl NeuralNetwork {
    fn new(nb_inputs: u32, nb_hidden: u32, nb_outputs: u32) -> NeuralNetwork {
        NeuralNetwork {
            nb_inputs,
            nb_hidden,
            nb_outputs,
        }
    }
    /*fn feed_forward(&self, input: Vec<i32>) {

    }*/
}

/* Functions ------------------------------------- */

fn main() {
    let neural_network: NeuralNetwork = NeuralNetwork::new(2, 2, 1 );
    let input: [i32; 2] = [1, 0];

    //let output: i32 = neural_network.feed_forward(input);
    //println!("{}", output);

    let mut m: Matrix = Matrix::new(3, 2);
    println!("{:?}", &m);

    &m.add(5.0);
    println!("{:?}", &m);

    &m.multiply(2.0);
    println!("{:?}", &m);

}
