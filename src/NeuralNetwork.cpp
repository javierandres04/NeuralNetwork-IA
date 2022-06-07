//
// Created by Charlie Solorzano  on 12/5/22.
//

#include <iostream>
#include "NeuralNetwork.hpp"
#include <random>

using namespace std;

NeuralNetwork::NeuralNetwork(vector<int> neuronCountPerLayer)
{

  this->learningRate = 0.1;
  this->pesos.resize(neuronCountPerLayer.size() + 1);
  this->biases.resize(neuronCountPerLayer.size() + 1);
  this->neuronCountPerLayer = neuronCountPerLayer;
  this->layerOutputs.resize(neuronCountPerLayer.size() + 1);
  // this->biasOutputLayer  = *(new Matrix(outputLayer, 1));
}

NeuralNetwork::~NeuralNetwork()
{
}

/* PREDICT VIEJO
vector<double> NeuralNetwork::predict(vector<double> values){
    Matrix input  = Matrix::fromArray(values); // Queda una matriz con una sola columna

    Matrix hidden = Matrix::multiply(this->weightsInputHiddenLayers, input); // Explicacion debajo

     * Queda una matriz con las filas de weights input hidden layer =  neuron count y las columnas de input = 1
     //
    hidden.addMatrix(this->biasHiddenLayer); // Se le suman los bias al matriz anterior
    hidden.sigmoid();

    //2 Capa
    // Hidden tiene almacenado el resultado final de esta capa

    Matrix output = Matrix::multiply(this->weightsHiddenOutputLayers, hidden);
    output.addMatrix(this->biasOutputLayer);
    output.sigmoid();

    return output.toArray();
}
 */

// PREDICT NUEVO
vector<double> NeuralNetwork::predict(vector<double> values)
{
  Matrix inputForLayer = Matrix::fromArray(values); // Queda una matriz con una sola columna
  Matrix networkOutput;
  // Forward propagation de 3 pesos
  Matrix tempResults;
  for (int indexPeso = 0; indexPeso < this->pesos.size(); indexPeso++)
  {
    tempResults = Matrix::multiply(this->pesos[indexPeso], inputForLayer);
    // TODO: aca tambien esta el problema de los bias
    //  tempResults.addScalar(this->biases[indexPeso].toArray()[0]);
    tempResults.addMatrix(this->biases[indexPeso]);
    tempResults.sigmoid();
    inputForLayer = tempResults;
  }
  // networkOutput = inputForLayer;

  return inputForLayer.toArray();
}

void NeuralNetwork::trainNuevo(vector<double> entrada, vector<double> expectedOutputValue)
{
  Matrix input = Matrix::fromArray(entrada);              // columna
  Matrix inputForLayer;                                   // = Matrix::fromArray(entrada); //columna
  Matrix target = Matrix::fromArray(expectedOutputValue); // Crea la matriz target a partir del vector con los resultados esperados
  Matrix error;
  Matrix gradient;
  Matrix networkOutput;
  int inputCount = int(entrada.size());

  // Training.
  inputForLayer = Matrix::fromArray(entrada);
  Matrix tempResults;
  // Forward propagation de 3 pesos
  for (int indexPeso = 0; indexPeso < this->pesos.size(); indexPeso++)
  {
    tempResults = Matrix::multiply(this->pesos[indexPeso], inputForLayer);
    // TODO: Cambiamos esto para que se aplique el bias correspondiente a cada neurona
    tempResults.addMatrix(biases[indexPeso]);
    // tempResults.addScalar(biases[indexPeso].toArray()[0]);
    tempResults.sigmoid();
    this->layerOutputs[indexPeso] = tempResults;
    inputForLayer = tempResults;
  }
  networkOutput = inputForLayer;
  // networkOutput.printMatrix();

  // Calcular error en última capa y ajustar gradiente. layer[layersCount-2
  // Crea la matriz error : Calculando el error de todos los output de nuestra red
  error = Matrix::matrixSubstract(target, networkOutput /* Epoch Output */);
  // cout << "Error" << endl;
  // error.printMatrix();
  // cout << endl;
  //  Obtiene la gradiente de la derivada de la funcion de activacion
  // TODO: derviada del coste con respecto a la funcion de activacion o sea en este caso esta es el output de lred
  gradient = networkOutput.dsigmoid();
  // Multiplica la gradiente por la matriz del error y almacena en la matriz gradiente
  gradient.multiply(error);
  // Multiplica los resultados anteriores por el valor de la tasa de aprendizaje
  gradient.multiply(this->learningRate);
  // transponer el output de la capa anterior
  //////2
  Matrix previousLayerOutputTransposed = Matrix::transpose(layerOutputs[this->pesos.size() - 2]);
  // Multiplica la gradiente por la transpuesta
  Matrix weightsPreviousLayerOutputDelta = Matrix::multiply(gradient, previousLayerOutputTransposed);

  this->pesos[this->pesos.size() - 1].addMatrix(weightsPreviousLayerOutputDelta);
  //  biases[biases.size() - 1].addMatrix(gradient);
  this->biases[this->biases.size() - 1].addMatrix(gradient);

  //  this->biases[this->biases.size() - 1].addScalar(gradient.toArray()[0]);

  Matrix weightsNextLayer = pesos[this->pesos.size() - 1];
  Matrix weightsNextLayerTransposed;

  Matrix hiddenLayerErrors;
  Matrix nextHiddenLayerErrors = error;
  Matrix weightsPreviousLayerDelta;

  // Backpropagation
  for (int indexCapa = this->pesos.size() - 2; indexCapa > 0; indexCapa--)
  {
    // transponer el output de la capa anterior
    // Multiplica la gradiente por la transpuesta de la capa anterior = pesosDeCapaDelta = lo que vamos
    // Sumar los pesos de la capa con el delta
    // TODO: Parece que el transpose esta malo y devuelve una matriz vacia
    weightsNextLayerTransposed = Matrix::transpose(weightsNextLayer);

    hiddenLayerErrors = Matrix::multiply(weightsNextLayerTransposed, nextHiddenLayerErrors /*error*/);
    Matrix layerGradient = layerOutputs[indexCapa].dsigmoid();
    layerGradient.multiply(hiddenLayerErrors);
    layerGradient.multiply(this->learningRate);

    // INPUT
    previousLayerOutputTransposed = Matrix::transpose(layerOutputs[indexCapa - 1]);
    weightsPreviousLayerDelta = Matrix::multiply(layerGradient, previousLayerOutputTransposed);

    // Almacena los valores en la matriz de pesos inicial corregida
    this->pesos[indexCapa].addMatrix(weightsPreviousLayerDelta);
    this->biases[indexCapa].addMatrix(layerGradient);

    weightsNextLayer = pesos[indexCapa];
    nextHiddenLayerErrors = hiddenLayerErrors;
  }
  //
  weightsNextLayerTransposed = Matrix::transpose(weightsNextLayer);
  // hiddenLayerErrors = weightsNextLayerTransposed;
  hiddenLayerErrors = Matrix::multiply(weightsNextLayerTransposed, nextHiddenLayerErrors);

  Matrix layerGradient = layerOutputs[0].dsigmoid();
  layerGradient.multiply(hiddenLayerErrors); /*weightsNextLayerTransposed*/
  layerGradient.multiply(this->learningRate);
  //
  previousLayerOutputTransposed = Matrix::transpose(input);
  weightsPreviousLayerDelta = Matrix::multiply(layerGradient, previousLayerOutputTransposed);
  //
  //  // Almacena los valores en la matriz de pesos inicial corregida
  pesos[0].addMatrix(weightsPreviousLayerDelta);
  biases[0].addMatrix(layerGradient);
}
/*
void NeuralNetwork::train(vector<double> values, vector<double> expectedValues){


    //________________________________________________________________________________________________

    // Primera Capa

    //________________________________________________________________________________________________


    //
    // Agarra el vector value y lo transforma en matriz para poder utilizar producto punto-
    // entre el valor de cada entrada por los pesos de cada matriz
    //
    //

    Matrix input  = Matrix::fromArray(values); //columna
    Matrix hidden = Matrix::multiply(this->weightsInputHiddenLayers, input);

    // Le suma el bias al resultado de la multiplicacion
    hidden.addMatrix(this->biasHiddenLayer);
    // Le aplica la funcion de activacion al resultado obtenido
    hidden.sigmoid();

    //________________________________________________________________________________________________

    // Segunda capa

    //________________________________________________________________________________________________



    // Despues cada resultado de las neuronas obtenidas se lo multiplica a los pesos de la segunda capa de neuronas
    Matrix output = Matrix::multiply(this->weightsHiddenOutputLayers, hidden);
    // Le suma el segundo set de bias
    output.addMatrix(this->biasOutputLayer);
    // Le aplica la funcion sigmoid a estos segunods resultados
    output.sigmoid();

    //________________________________________________________________________________________________

    //                   Calculo del Descenso por gradiente para mejorar mas rapido

    //________________________________________________________________________________________________


    Matrix target   = Matrix::fromArray(expectedValues); // Crea la matriz target a partir del vector con los resultados esperados

    Matrix error    = Matrix::matrixSubstract(target, output); // Crea la matriz error : Calculando el error de todos los output de nuestra red

    //Inicio del for

    Matrix gradient = output.dsigmoid(); // Obtiene la gradiente de la derivada de la funcion de activacion
    gradient.multiply(error);           // Multiplica la gradiente por la matriz del error y almacena el resultado en error
    gradient.multiply(this->learningRate); // Multiplica los resultados anteriores por el valor de la tasa de aprendizaje

    Matrix hiddenLayerTransposed = Matrix::transpose(hidden);
    //se calcula cuanto deben cambiar los pesos
    Matrix weightsHiddenOutputLayersDelta = Matrix::multiply(gradient, hiddenLayerTransposed);

    weightsHiddenOutputLayers.addMatrix(weightsHiddenOutputLayersDelta);
    this->biasOutputLayer.addMatrix(gradient);
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    Matrix weightsHiddenOutputLayersTransposed = Matrix::transpose(this->weightsHiddenOutputLayers);
    Matrix hiddenLayerErrors = Matrix::multiply(weightsHiddenOutputLayersTransposed, error);

    Matrix hiddenLayerGradient = hidden.dsigmoid();
    hiddenLayerGradient.multiply(hiddenLayerErrors);
    hiddenLayerGradient.multiply(this->learningRate);

    Matrix inputTransposed = Matrix::transpose(input);
    Matrix weightsInputHiddenLayersDelta = Matrix::multiply(hiddenLayerGradient, inputTransposed);

    // Almacena los valores en la matriz de pesos inicial corregida
    weightsInputHiddenLayers.addMatrix(weightsInputHiddenLayersDelta);
    this->biasHiddenLayer.addMatrix(hiddenLayerGradient);
}
*/

// FIT VIEJO
/*
void NeuralNetwork::fit(vector<vector<double>> values, vector<vector<double>> expectedValues, int epochs){
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,values.size()-1);
    for (int epoch = 0; epoch < epochs; epoch++){
        int sampleN = distribution(generator);;
        this->train(values[sampleN], expectedValues[sampleN]);
    }
}
 */

// FIT NUEVO
void NeuralNetwork::fit(vector<vector<double>> values, vector<vector<double>> expectedValues, int epochs)
{

  int outputLayer = 1;

  int inputCount = int(values[0].size());
  for (int indexCapa = 0; indexCapa < this->pesos.size() - 1; indexCapa++)
  {

    this->pesos[indexCapa] = (Matrix(this->neuronCountPerLayer[indexCapa], inputCount));
    this->biases[indexCapa] = (Matrix(this->neuronCountPerLayer[indexCapa], 1));
    inputCount = this->neuronCountPerLayer[indexCapa];
  }
  // Ultimo set de pesos
  this->pesos[this->pesos.size() - 1] = (Matrix(outputLayer, neuronCountPerLayer[neuronCountPerLayer.size() - 1]));
  this->biases[this->biases.size() - 1] = Matrix(1, 1);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, values.size() - 1);
  // TODO: poneer epochs
  for (int epoch = 0; epoch < epochs; epoch++)
  {
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++)
    {
      this->trainNuevo(values[valueIndex], expectedValues[valueIndex]);
    }

    // int sampleN = distribution(generator);
    // this->trainNuevo(values[sampleN], expectedValues[sampleN]);
  }
}
