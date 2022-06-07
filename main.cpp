#include <iostream>
#include <vector>
#include "./src/NeuralNetwork.hpp"
#include "./src/Parser.hpp"
#include "./src/DatasetManipulator.hpp"
#include "./src/VectorPrinter.hpp"
#include "./src/Matrix.hpp";
using namespace std;

void saveNetwork(string filename, vector<Matrix> &biases, vector<Matrix> &pesos)
{
  fstream new_file;
  new_file.open(filename, ios::out);

  new_file << "b," << biases.size() << ",\n";

  for (size_t i = 0; i < biases.size(); i++)
  {
    new_file << biases[i].toString();
  }

  new_file << "p," << pesos.size() << ",\n";

  for (size_t i = 0; i < pesos.size(); i++)
  {
    new_file << pesos[i].toString();
  }

  new_file.close();
}

int getChargedVectorsCount(stringstream &str_line)
{
  string word;
  getline(str_line, word, ',');
  getline(str_line, word, ',');
  return stoi(word);
}

Matrix chargeMatrixValues(fstream &file, int rows, int columns)
{
  Matrix matrix(rows, columns);
  string line, word;
  for (int i = 0; i < rows; i++)
  {
    getline(file, line);
    stringstream str(line);
    for (int j = 0; j < columns; j++)
    {
      getline(str, word, ',');
      matrix.data[i][j] = stod(word);
    }
  }
  return matrix;
}

void getChargingMatrixSize(stringstream &str_line, int &rows, int &columns)
{
  string word;
  getline(str_line, word, ',');
  getline(str_line, word, ',');
  rows = stoi(word);
  getline(str_line, word, ',');
  columns = stoi(word);
}

void chargeNetwork(string fileName, vector<Matrix> &biases, vector<Matrix> &pesos)
{
  string line, word;
  int weights_count = 0;
  int biases_count = 0;
  int rows = 0;
  int columns = 0;
  fstream file(fileName, ios::in);
  if (file.is_open())
  {
    getline(file, line);
    biases_count = getChargedVectorsCount(stringstream(line));
    biases.resize(biases_count);
    for (int i = 0; i < biases_count; i++)
    {
      getline(file, line);
      getChargingMatrixSize(stringstream(line), rows, columns);
      biases[i] = chargeMatrixValues(file, rows, columns);
    }
  }
  getline(file, line);
  weights_count = getChargedVectorsCount(stringstream(line));
  pesos.resize(weights_count);
  for (int i = 0; i < weights_count; i++)
  {
    getline(file, line);
    getChargingMatrixSize(stringstream(line), rows, columns);
    pesos[i] = (chargeMatrixValues(file, rows, columns));
  }
  file.close();
}

int main()
{
  vector<vector<double>> values{
      vector<double>{0, 0, 0},
      vector<double>{1, 0, 0},
      vector<double>{1, 1, 0},
      vector<double>{1, 1, 1},
      vector<double>{0, 1, 0},
      vector<double>{0, 1, 1},
      vector<double>{0, 0, 1},
      vector<double>{1, 0, 1},
      // vector<double>{ 0, 0},
      // vector<double>{ 0, 1},
      // vector<double>{ 1, 0},
      // vector<double>{ 1, 1},
  };

  vector<vector<double>> expectedValues{
      vector<double>{0},
      vector<double>{1},
      vector<double>{0},
      vector<double>{1},
      vector<double>{1},
      vector<double>{0},
      vector<double>{1},
      vector<double>{0},
      // vector<double>{0},
      // vector<double>{1},
      // vector<double>{1},
      // vector<double>{0},

  };
  // vector<vector<double>> values;
  // vector<vector<double>> expectedValues;
  // vector<vector<string>> categories;

  vector<int> layers = {3, 2, 2};
  NeuralNetwork neuralNetwork = NeuralNetwork(layers);
  vector<double> output;

  // Parser::readCSV("winequality-white.csv", values, categories);
  // DatasetManipulator::normalize(values);
  // DatasetManipulator::extractExpectedValues(values, expectedValues);

  // neuralNetwork.fit(values, expectedValues, 50000);
  // saveNetwork("redXor.txt", neuralNetwork.biases, neuralNetwork.pesos);
  chargeNetwork("redXor.txt", neuralNetwork.biases, neuralNetwork.pesos);
  cout << "biases" << endl;
  for (int i = 0; i < neuralNetwork.biases.size(); i++)
  {

    neuralNetwork.biases[i].printMatrix();
    cout << i << endl;
  }
  cout << "pesos" << endl;
  for (int i = 0; i < neuralNetwork.pesos.size(); i++)
  {
    neuralNetwork.pesos[i].printMatrix();
    cout << i << endl;
  }
  for (auto value : values)
  {

    output.push_back(neuralNetwork.predict(value)[0]);

    //
  }

  for (size_t index = 0; index < expectedValues.size(); index++)
  {
    cout << expectedValues[index][0] << " neural output: " << output[index] << endl;
  }
}
