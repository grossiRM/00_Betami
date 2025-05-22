using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Utilities;

namespace USGS.Puma.Core
{
    /// <summary>
    /// This class serves as the ase class for the generic array classes.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks></remarks>
    public abstract class NumberArray<T>
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        protected NumberArray()
        {
            _gn = new GenericNumberUtility() as IGenericNumberUtility<T>;
            _BaseValues = new T[0];
        }
        #endregion

        #region Protected Members
        /// <summary>
        /// 
        /// </summary>
        private T[] _BaseValues;
        /// <summary>
        /// Gets a standard array that provides direct access to the data values.
        /// </summary>
        /// <remarks></remarks>
        protected T[] BaseValues
        {
            get { return _BaseValues; }
        }

        /// <summary>
        /// 
        /// </summary>
        private IGenericNumberUtility<T> _gn = null;
        /// <summary>
        /// Gets an instance of GenericNumberUtility to provide support for math operations on generic data.
        /// </summary>
        /// <remarks></remarks>
        protected IGenericNumberUtility<T> Math
        {
            get { return _gn; }
        }
        /// <summary>
        /// Sets the array values from a NumberArray.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <remarks></remarks>
        protected void SetValuesFromNumberArray(NumberArray<T> values)
        {
            if (values.ElementCount != this.ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = values.BaseValues[i];
            }
        }
        /// <summary>
        /// Adds each element the specified number array.
        /// </summary>
        /// <param name="values">The specified number array.</param>
        /// <remarks></remarks>
        protected void AddNumberArray(NumberArray<T> values)
        {
            if (values.ElementCount != this.ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Add(_BaseValues[i], values.BaseValues[i]);
            }
        }
        /// <summary>
        /// Subtracts the number array.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <remarks></remarks>
        protected void SubtractNumberArray(NumberArray<T> values)
        {
            if (values.ElementCount != this.ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Subtract(_BaseValues[i], values.BaseValues[i]);
            }
        }
        /// <summary>
        /// Multiplies the number array.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <remarks></remarks>
        protected void MultiplyNumberArray(NumberArray<T> values)
        {
            if (values.ElementCount != this.ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Multiply(_BaseValues[i], values.BaseValues[i]);
            }
        }
        /// <summary>
        /// Divides the number array.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <remarks></remarks>
        protected void DivideNumberArray(NumberArray<T> values)
        {
            if (values.ElementCount != this.ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Divide(_BaseValues[i], values.BaseValues[i]);
            }
        }
        /// <summary>
        /// Resizes the array.
        /// </summary>
        /// <param name="elementCount">The number of elements in the array.</param>
        /// <remarks></remarks>
        protected void ResizeArray(int elementCount)
        {
            T value = default(T);
            ResizeArray(elementCount, value);
        }
        /// <summary>
        /// Resizes the array.
        /// </summary>
        /// <param name="elementCount">The element count.</param>
        /// <param name="value">The value.</param>
        /// <remarks></remarks>
        protected void ResizeArray(int elementCount, T value)
        {
            _BaseValues = new T[elementCount];
            this.SetValues(value);
        }
        /// <summary>
        /// Resizes the array and sets the value of its elements.
        /// </summary>
        /// <param name="values">The array of values.</param>
        /// <remarks></remarks>
        protected void ResizeArray(T[] values)
        {
            _BaseValues = new T[values.Length];
            this.SetValues(values);
        }
        #endregion

        #region Public Members
        /// <summary>
        /// Gets a copy of the array values as a 1-D array.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T[] GetValues()
        {
            T[] values = new T[ElementCount];
            for (int i = 0; i < ElementCount; i++)
            { values[i] = _BaseValues[i]; }
            return values;
        }

        /// <summary>
        /// Assign a constant constantValue to all elements of the matrix.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <remarks></remarks>
        public void SetValues(T value)
        {
            for (int i = 0; i < ElementCount; i++)
            { _BaseValues[i] = value; }
        }
        /// <summary>
        /// Sets the array values from a standard zero-based 1-D array.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <remarks></remarks>
        public void SetValues(T[] values)
        {
            if (values.Length != ElementCount)
            { throw (new ArgumentException("Array dimensions do not match.")); }

            for (int i = 0; i < ElementCount; i++)
            { _BaseValues[i] = values[i]; }

        }
        /// <summary>
        /// Gets the element count.
        /// </summary>
        /// <remarks></remarks>
        public int ElementCount
        {
            get { return _BaseValues.Length; }
        }

        /// <summary>
        /// Gets the value as double.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public double GetValueAsDouble(int element)
        {
            if (element < 1 || element > this.ElementCount)
            {
                throw new ArgumentOutOfRangeException("element");
            }

            return Convert.ToDouble(_BaseValues[element - 1]);

        }
        /// <summary>
        /// Gets the value as single.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public float GetValueAsSingle(int element)
        {
            if (element < 1 || element > this.ElementCount)
            {
                throw new ArgumentOutOfRangeException("element");
            }

            return Convert.ToSingle(_BaseValues[element - 1]);

        }

        /// <summary>
        /// Get or set an array of element values by the elementNumber index.
        /// The first element in the array is at elementNumber = 1.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public T this[int elementNumber]
        {
            get { return _BaseValues[elementNumber - 1]; }
            set { _BaseValues[elementNumber - 1] = value; }
        }

        /// <summary>
        /// Values the count.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public int ValueCount(T value)
        {
            int count = 0;
            for (int i = 0; i < ElementCount; i++)
            {
                if (this.Math.Equal(_BaseValues[i], value)) count++;
            }
            return count;
        }

        /// <summary>
        /// Set all elements of this matrix equal to default value.
        /// </summary>
        /// <remarks></remarks>
        public void ResetToDefaultValue()
        {
            T val = default(T);
            for (int i = 0; i < ElementCount; i++)
            {
                _BaseValues[i] = val;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this instance is constant.
        /// </summary>
        /// <remarks></remarks>
        public bool IsConstant
        {
            get
            {
                T val = this.BaseValues[0];
                for (int i = 1; i < ElementCount; i++)
                { if (!this.Math.Equal(_BaseValues[i], val)) return false; }
                return true;
            }
        }

        /// <summary>
        /// Adds a constant value to all the elements of the array.
        /// </summary>
        /// <param name="c">The specified constant value.</param>
        /// <remarks></remarks>
        public void Add(T c)
        {
            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Add(_BaseValues[i], c);
            }
        }
        /// <summary>
        /// Subtracts the specified c.
        /// </summary>
        /// <param name="c">The c.</param>
        /// <remarks></remarks>
        public void Subtract(T c)
        {
            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Subtract(_BaseValues[i], c);
            }
        }
        /// <summary>
        /// Mulitiplies the specified c.
        /// </summary>
        /// <param name="c">The c.</param>
        /// <remarks></remarks>
        public void Multiply(T c)
        {
            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Multiply(_BaseValues[i], c);
            }
        }
        /// <summary>
        /// Divides the specified c.
        /// </summary>
        /// <param name="c">The c.</param>
        /// <remarks></remarks>
        public void Divide(T c)
        {
            for (int i = 0; i < this.ElementCount; i++)
            {
                _BaseValues[i] = this.Math.Divide(_BaseValues[i], c);
            }
        }

        #endregion
    }
}
