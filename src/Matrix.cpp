//
// Created by Charlie Solorzano  on 12/5/22.
//

/*
  This code was based on the one presented in the following tutorial:
  https://towardsdatascience.com/understanding-and-implementing-neural-networks-in-java-from-scratch-61421bb6352c
*/
#include "Matrix.hpp"

#include <vector>
#include "iostream"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include <sstream>

using namespace std;

Matrix::Matrix()
{
  this->rows = 0;
  this->columns = 0;
}
// double initNeuron() { return ((double)rand()) / RAND_MAX; }
Matrix::Matrix(int rows, int columns)
{
  srand(time(NULL));
  this->data.resize(rows);

  for (int rowIndex = 0; rowIndex < rows; rowIndex++)
  {
    this->data[rowIndex].resize(columns);
    for (int columnIndex = 0; columnIndex < columns; columnIndex++)
    {
      this->data[rowIndex][columnIndex] = -1 + (double)(rand()) / ((double)(RAND_MAX / (1 - -1)));
      // this->data[rowIndex][columnIndex] = 3;
    }
  }
  this->rows = rows;
  this->columns = columns;
}

Matrix::~Matrix()
{
  this->data.clear();
}

void Matrix::printMatrix()
{
  for (size_t rowIndex = 0; rowIndex < this->data.size(); rowIndex++)
  {
    for (size_t columnIndex = 0; columnIndex < data[0].size(); columnIndex++)
    {
      cout << this->data[rowIndex][columnIndex] << " ";
    }
    cout << "\n";
  }
}

void Matrix::addScalar(double scalar)
{
  for (int rowIndex = 0; rowIndex < rows; rowIndex++)
  {
    for (int column = 0; column < this->columns; column++)
    {
      this->data[rowIndex][column] += scalar;
    }
  }
}

void Matrix::addMatrix(Matrix &matrix)
{
  if (this->columns != matrix.columns || this->rows != matrix.rows)
  {
    cout << "The matrices have different shapes ADDMATRIX" << endl;
  }
  else
  {
    for (int row = 0; row < this->rows; row++)
    {
      for (int column = 0; column < this->columns; column++)
      {
        this->data[row][column] += matrix.data[row][column];
      }
    }
  }
}

Matrix Matrix::matrixSubstract(Matrix &m1, Matrix &m2)
{
  Matrix temp = Matrix(m1.rows, m1.columns);
  if (m1.columns != m2.columns || m1.rows != m2.rows)
  {
    cout << "The matrices have different shapes MATRIXSUBSTRACT" << endl;
  }
  else
  {
    for (int row = 0; row < m1.rows; row++)
    {
      for (int column = 0; column < m1.columns; column++)
      {
        temp.data[row][column] = m1.data[row][column] - m2.data[row][column];
      }
    }
  }
  return temp;
}

Matrix Matrix::transpose(Matrix &a)
{
  Matrix temp = Matrix(a.columns, a.rows);
  for (int row = 0; row < a.rows; row++)
  {
    for (int column = 0; column < a.columns; column++)
    {
      temp.data[column][row] = a.data[row][column];
    }
  }
  return temp;
}

void Matrix::multiply(double scalar)
{
  for (int row = 0; row < this->rows; row++)
  {
    for (int column = 0; column < this->columns; column++)
    {
      this->data[row][column] *= scalar;
    }
  }
}

void Matrix::multiply(Matrix &a)
{
  for (int row = 0; row < a.rows; row++)
  {
    for (int column = 0; column < a.columns; column++)
    {
      this->data[row][column] = this->data[row][column] * a.data[row][column];
    }
  }
}

Matrix Matrix::multiply(Matrix &a, Matrix &b)
{
  Matrix temp = Matrix(a.rows, b.columns);

  if (a.columns != b.rows)
  {
    cout << "The matrices have different shapes MATRIXMULTIPLY" << endl;
  }
  else
  {
    for (int rowIndex = 0; rowIndex < temp.rows; rowIndex++)
    {
      for (int columnIndex = 0; columnIndex < temp.columns; columnIndex++)
      {
        double sum = 0;
        for (int k = 0; k < a.columns; k++)
        {
          sum += a.data[rowIndex][k] * b.data[k][columnIndex];
        }
        temp.data[rowIndex][columnIndex] = sum;
      }
    }
  }
  return temp;
}

// answer = pow(base, power);
void Matrix::sigmoid()
{
  for (int rowIndex = 0; rowIndex < this->rows; rowIndex++)
  {
    for (int columnIndex = 0; columnIndex < this->columns; columnIndex++)
      this->data[rowIndex][columnIndex] = 1 / (1 + exp(-this->data[rowIndex][columnIndex]));
  }
}

Matrix Matrix::dsigmoid()
{
  Matrix temp = Matrix(this->rows, this->columns);
  for (int rowIndex = 0; rowIndex < rows; rowIndex++)
  {
    for (int columnIndex = 0; columnIndex < this->columns; columnIndex++)
    {
      temp.data[rowIndex][columnIndex] = this->data[rowIndex][columnIndex] * (1 - this->data[rowIndex][columnIndex]);
    }
  }
  return temp;
}

Matrix Matrix::fromArray(vector<double> values)
{
  Matrix temp = Matrix(int(values.size()), 1);
  for (int index = 0; index < values.size(); index++)
  {
    temp.data[index][0] = values[index];
  }
  return temp;
}

vector<double> Matrix::toArray()
{
  vector<double> temp;
  for (int row = 0; row < this->rows; row++)
  {
    for (int column = 0; column < columns; column++)
    {
      temp.push_back(this->data[row][column]);
    }
  }
  return temp;
}

string Matrix::toString()
{
  ostringstream os;
  os << "s," << this->rows << "," << this->columns << ",\n";
  for (size_t rowIndex = 0; rowIndex < this->data.size(); rowIndex++)
  {
    for (size_t columnIndex = 0; columnIndex < data[0].size(); columnIndex++)
    {
      os << this->data[rowIndex][columnIndex] << ",";
    }
    os << "\n";
  }
  return os.str();
}