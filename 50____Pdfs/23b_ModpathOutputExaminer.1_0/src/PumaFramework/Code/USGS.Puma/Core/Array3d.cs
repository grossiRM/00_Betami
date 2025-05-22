using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks></remarks>
    public class Array3d<T> : NumberArray<T>
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="Array3d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="layers">The layers.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <remarks></remarks>
        public Array3d(int layers, int rows, int columns)
        {
            CreateNewArray(layers, rows, columns);
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="layers">The layers.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        public Array3d(int layers, int rows, int columns,T initialValue)
        {
            CreateNewArray(layers, rows, columns, initialValue);
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="Array2d&lt;T&gt;"/> class.
        /// </summary>
        /// <param name="matrixData">The matrix data.</param>
        /// <remarks></remarks>
        public Array3d(Array3d<T> matrixData)
        {
            CreateNewArray(matrixData);
        }

        #endregion

        #region Public Members
        /// <summary>
        /// 
        /// </summary>
        private int _LayerCount;
        /// <summary>
        /// Gets or sets the layer count.
        /// </summary>
        /// <value>The layer count.</value>
        /// <remarks></remarks>
        public int LayerCount
        {
            get { return _LayerCount; }
        }
        /// <summary>
        /// 
        /// </summary>
        private int _RowCount;
        /// <summary>
        /// Gets or sets the row count.
        /// </summary>
        /// <value>The row count.</value>
        /// <remarks></remarks>
        public int RowCount
        {
            get { return _RowCount; }
        }
        /// <summary>
        /// 
        /// </summary>
        private int _ColumnCount;
        /// <summary>
        /// Gets or sets the column count.
        /// </summary>
        /// <value>The column count.</value>
        /// <remarks></remarks>
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        /// <summary>
        /// Get or set an array element by layer, row, and column numbers.
        /// The layer, row and column numbers begin at 1.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T this[int layerNumber, int rowNumber, int columnNumber]
        {
            get 
            {
                if (!CheckIndex(layerNumber, rowNumber, columnNumber))
                {
                    throw new ArgumentOutOfRangeException("The specified layer, row, or column is out of range.");
                }
                int i = (_RowCount * _ColumnCount) * (layerNumber - 1) + (rowNumber - 1) * _ColumnCount + columnNumber - 1;
                return this.BaseValues[i]; 
            }
            set 
            {
                if (!CheckIndex(layerNumber, rowNumber, columnNumber))
                {
                    throw new ArgumentOutOfRangeException("The specified layer, row, or column is out of range.");
                }
                int i = (_RowCount * _ColumnCount) * (layerNumber - 1) + (rowNumber - 1) * _ColumnCount + columnNumber - 1;
                this.BaseValues[i] = value; 
            }
        }

        /// <summary>
        /// Sets the values.
        /// </summary>
        /// <param name="matrix">A 3-d array of values.</param>
        /// <remarks></remarks>
        public void SetValues(Array3d<T> matrix)
        {
            if (!CheckDimensions(matrix))
            {
                throw new ArgumentException("The specified matrix does not have the correct dimensions.");
            }
            this.SetValuesFromNumberArray(matrix);
        }

        /// <summary>
        /// Adds the specified matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <remarks></remarks>
        public void Add(Array3d<T> matrix)
        {
            if (!CheckDimensions(matrix))
            {
                throw new ArgumentException("The specified matrix does not have the correct dimensions.");
            }
            this.AddNumberArray(matrix as NumberArray<T>);
        }
        /// <summary>
        /// Subtracts the specified matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <remarks></remarks>
        public void Subtract(Array3d<T> matrix)
        {
            if (!CheckDimensions(matrix))
            {
                throw new ArgumentException("The specified matrix does not have the correct dimensions.");
            }
            this.SubtractNumberArray(matrix as NumberArray<T>);
        }
        /// <summary>
        /// Multiplies the specified matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <remarks></remarks>
        public void Multiply(Array3d<T> matrix)
        {
            if (!CheckDimensions(matrix))
            {
                throw new ArgumentException("The specified matrix does not have the correct dimensions.");
            }
            this.MultiplyNumberArray(matrix as NumberArray<T>);
        }
        /// <summary>
        /// Divides the specified matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <remarks></remarks>
        public void Divide(Array3d<T> matrix)
        {
            if(!CheckDimensions(matrix))
            {
                throw new ArgumentException("The specified matrix does not have the correct dimensions.");
            }
            this.DivideNumberArray(matrix as NumberArray<T>);
        }

        #endregion

        #region Private Members
        /// <summary>
        /// Checks the dimensions.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool CheckDimensions(Array3d<T> matrix)
        {
            if (matrix.LayerCount < 1 || matrix.LayerCount > _LayerCount)
            { return false; }
            else if (matrix.RowCount < 1 || matrix.RowCount > _RowCount)
            { return false; }
            else if (matrix.ColumnCount < 1 || matrix.ColumnCount > _ColumnCount)
            { return false; }
            else
            { return true; }
        }
        /// <summary>
        /// Checks the index.
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool CheckIndex(int layer, int row, int column)
        {
            if (layer < 1 || layer > _LayerCount)
            { return false; }
            else if (row < 1 || row > _RowCount)
            { return false; }
            else if (column < 1 || column > _ColumnCount)
            { return false; }
            else
            { return true; }

        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="layers">The layers.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int layers, int rows, int columns)
        {
            T val = default(T);
            CreateNewArray(layers, rows, columns, val);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="layers">The layers.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <param name="initialValue">The initial value.</param>
        /// <remarks></remarks>
        private void CreateNewArray(int layers, int rows, int columns, T initialValue)
        {
            _LayerCount = layers;
            _RowCount = rows;
            _ColumnCount = columns;
            int elementCount = layers * rows * columns;
            this.ResizeArray(elementCount, initialValue);
        }
        /// <summary>
        /// Creates the new array.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <remarks></remarks>
        private void CreateNewArray(Array3d<T> matrix)
        {
            if (matrix == null)
            { throw new ArgumentNullException("matrix"); }

            _LayerCount = matrix.LayerCount;
            _RowCount = matrix.RowCount;
            _ColumnCount = matrix.ColumnCount;
            this.ResizeArray(matrix.BaseValues);
        }

        #endregion

    }
}
