using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    /// <summary>
    /// 
    /// </summary>
    public class ArrayUtility : IArrayUtility<int>, IArrayUtility<float>, IArrayUtility<double>
    {
        #region IArrayUtility<int> Members

        /// <summary>
        ///  Adds a constant value to the array and returns the result as a new array.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<int> Add(Array2d<int> a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<int> m = new Array2d<int>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row,column] = a[row, column] + c; }
            }

            return m;

        }
        /// <summary>
        /// Adds the corresponding elements of the input arrays and returns the result
        /// as a new array.
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<int> Add(Array2d<int> a1, Array2d<int> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<int> m = new Array2d<int>(a1.RowCount,a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row,column] = a1[row, column] + a2[row,column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public int[] Add(int[] a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            int[] m = new int[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] + c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public int[] Add(int[] a1, int[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            int[] m = new int[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] + a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<int> Subtract(Array2d<int> a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<int> m = new Array2d<int>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] - c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<int> Subtract(Array2d<int> a1, Array2d<int> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<int> m = new Array2d<int>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] - a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public int[] Subtract(int[] a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            int[] m = new int[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] - c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public int[] Subtract(int[] a1, int[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            int[] m = new int[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] - a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<int> Multiply(Array2d<int> a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<int> m = new Array2d<int>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] * c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<int> Multiply(Array2d<int> a1, Array2d<int> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<int> m = new Array2d<int>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] * a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public int[] Multiply(int[] a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            int[] m = new int[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] * c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public int[] Multiply(int[] a1, int[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            int[] m = new int[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] * a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<int> Divide(Array2d<int> a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<int> m = new Array2d<int>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] / c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<int> Divide(Array2d<int> a1, Array2d<int> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<int> m = new Array2d<int>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] / a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public int[] Divide(int[] a, int c)
        {
            if (a == null)
                throw new ArgumentNullException();

            int[] m = new int[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] / c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public int[] Divide(int[] a1, int[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            int[] m = new int[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] / a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(Array2d<int> a, out int MinimumValue, out int MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(Array2d<int> a, out int MinimumValue, out int MaximumValue, int[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = int.MaxValue;
            MaximumValue = int.MinValue;

            Dictionary<int, object> excluded = new Dictionary<int, object>();
            if (excludedValues != null)
            {
                foreach (int val in excludedValues)
                {
                    excluded.Add(val, null);
                }
            }

            int elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if ( !excluded.ContainsKey(elementValue) )
                    {
                        if (elementValue > MaximumValue) MaximumValue = elementValue;
                        if (elementValue < MinimumValue) MinimumValue = elementValue;
                    }
                }
            }


        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(int[] a, out int MinimumValue, out int MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(int[] a, out int MinimumValue, out int MaximumValue, int[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = int.MaxValue;
            MaximumValue = int.MinValue;

            Dictionary<int, object> excluded = new Dictionary<int, object>();
            if (excludedValues != null)
            {
                foreach (int val in excludedValues)
                {
                    excluded.Add(val, null);
                }
            }

            int elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!excluded.ContainsKey(elementValue))
                {
                    if (elementValue > MaximumValue) MaximumValue = elementValue;
                    if (elementValue < MinimumValue) MinimumValue = elementValue;
                }
            }
        }

        public ArrayStatistics<int> FindMinimumAndMaximum(Array2d<int> a, int[] excludedValues)
        {
            try
            {
                int minValue;
                int maxValue;
                FindMinimumAndMaximum(a, out minValue, out maxValue, excludedValues);
                ArrayStatistics<int> stats = new ArrayStatistics<int>();
                stats.MinimumValue = minValue;
                stats.MaximumValue = maxValue;
                return stats;

            }
            catch (Exception)
            {
                return null;
            }

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<int> GetUniqueValues(Array2d<int> a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<int> GetUniqueValues(Array2d<int> a, int[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<int, object> dict = new Dictionary<int, object>();
            List<int> list = new List<int>();

            if (excludedValues != null)
            {
                foreach (int val in excludedValues)
                { dict.Add(val, null); }
            }

            int elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if (!dict.ContainsKey(elementValue))
                    {
                        dict.Add(elementValue, null);
                        list.Add(elementValue);
                    }
                }
            }

            return list;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<int> GetUniqueValues(int[] a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<int> GetUniqueValues(int[] a, int[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<int, object> dict = new Dictionary<int, object>();
            List<int> list = new List<int>();

            if (excludedValues != null)
            {
                foreach (int val in excludedValues)
                { dict.Add(val, null); }
            }

            int elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!dict.ContainsKey(elementValue))
                {
                    dict.Add(elementValue, null);
                    list.Add(elementValue);
                }
            }

            return list;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(Array2d<int> a, int dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { if (a[row, column] == dataValue) count++; }
            }

            return count;
            
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(int[] a, int dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int i = 0; i < a.Length; i++)
            { if (a[i] == dataValue) count++; }

            return count;

        }

        #endregion

        #region IArrayUtility<float> Members

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<float> Add(Array2d<float> a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<float> m = new Array2d<float>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] + c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<float> Add(Array2d<float> a1, Array2d<float> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<float> m = new Array2d<float>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] + a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public float[] Add(float[] a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            float[] m = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] + c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public float[] Add(float[] a1, float[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            float[] m = new float[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] + a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<float> Subtract(Array2d<float> a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<float> m = new Array2d<float>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] - c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<float> Subtract(Array2d<float> a1, Array2d<float> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<float> m = new Array2d<float>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] - a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public float[] Subtract(float[] a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            float[] m = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] - c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public float[] Subtract(float[] a1, float[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            float[] m = new float[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] - a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<float> Multiply(Array2d<float> a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<float> m = new Array2d<float>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] * c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<float> Multiply(Array2d<float> a1, Array2d<float> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<float> m = new Array2d<float>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] * a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public float[] Multiply(float[] a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            float[] m = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] * c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public float[] Multiply(float[] a1, float[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            float[] m = new float[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] * a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<float> Divide(Array2d<float> a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<float> m = new Array2d<float>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] / c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<float> Divide(Array2d<float> a1, Array2d<float> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<float> m = new Array2d<float>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] / a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public float[] Divide(float[] a, float c)
        {
            if (a == null)
                throw new ArgumentNullException();

            float[] m = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] / c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public float[] Divide(float[] a1, float[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            float[] m = new float[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] / a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(Array2d<float> a, out float MinimumValue, out float MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(Array2d<float> a, out float MinimumValue, out float MaximumValue, float[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = int.MaxValue;
            MaximumValue = int.MinValue;

            Dictionary<float, object> excluded = new Dictionary<float, object>();
            if (excludedValues != null)
            {
                foreach (float val in excludedValues)
                {
                    excluded.Add(val, null);
                }
            }

            float elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if (!excluded.ContainsKey(elementValue))
                    {
                        if (elementValue > MaximumValue) MaximumValue = elementValue;
                        if (elementValue < MinimumValue) MinimumValue = elementValue;
                    }
                }
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(float[] a, out float MinimumValue, out float MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(float[] a, out float MinimumValue, out float MaximumValue, float[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = float.MaxValue;
            MaximumValue = float.MinValue;

            Dictionary<float, object> excluded = new Dictionary<float, object>();
            if (excludedValues != null)
            {
                foreach (float val in excludedValues)
                {
                    excluded.Add(val, null);
                }
            }

            float elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!excluded.ContainsKey(elementValue))
                {
                    if (elementValue > MaximumValue) MaximumValue = elementValue;
                    if (elementValue < MinimumValue) MinimumValue = elementValue;
                }
            }

        }
        public ArrayStatistics<float> FindMinimumAndMaximum(Array2d<float> a, float[] excludedValues)
        {
            try
            {
                float minValue;
                float maxValue;
                FindMinimumAndMaximum(a, out minValue, out maxValue, excludedValues);
                ArrayStatistics<float> stats = new ArrayStatistics<float>();
                stats.MinimumValue = minValue;
                stats.MaximumValue = maxValue;
                return stats;

            }
            catch (Exception)
            {
                return null;
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<float> GetUniqueValues(Array2d<float> a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<float> GetUniqueValues(Array2d<float> a, float[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<float, object> dict = new Dictionary<float, object>();
            List<float> list = new List<float>();

            if (excludedValues != null)
            {
                foreach (int val in excludedValues)
                { dict.Add(val, null); }
            }

            float elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if (!dict.ContainsKey(elementValue))
                    {
                        dict.Add(elementValue, null);
                        list.Add(elementValue);
                    }
                }
            }

            return list;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<float> GetUniqueValues(float[] a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<float> GetUniqueValues(float[] a, float[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<float, object> dict = new Dictionary<float, object>();
            List<float> list = new List<float>();

            if (excludedValues != null)
            {
                foreach (float val in excludedValues)
                { dict.Add(val, null); }
            }

            float elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!dict.ContainsKey(elementValue))
                {
                    dict.Add(elementValue, null);
                    list.Add(elementValue);
                }
            }

            return list;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(Array2d<float> a, float dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { if (a[row, column] == dataValue) count++; }
            }

            return count;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(float[] a, float dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int i = 0; i < a.Length; i++)
            { if (a[i] == dataValue) count++; }

            return count;
        }

        #endregion

        #region IArrayUtility<double> Members

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<double> Add(Array2d<double> a, double c)
        {
            if (a == null)
                throw new ArgumentNullException("a");

            Array2d<double> m = new Array2d<double>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] + c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<double> Add(Array2d<double> a1, Array2d<double> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<double> m = new Array2d<double>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] + a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public double[] Add(double[] a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            double[] m = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] + c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public double[] Add(double[] a1, double[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            double[] m = new double[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] + a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<double> Subtract(Array2d<double> a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<double> m = new Array2d<double>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] - c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<double> Subtract(Array2d<double> a1, Array2d<double> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<double> m = new Array2d<double>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] - a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public double[] Subtract(double[] a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            double[] m = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] - c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public double[] Subtract(double[] a1, double[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            double[] m = new double[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] - a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<double> Multiply(Array2d<double> a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<double> m = new Array2d<double>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] * c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<double> Multiply(Array2d<double> a1, Array2d<double> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<double> m = new Array2d<double>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] * a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public double[] Multiply(double[] a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            double[] m = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] * c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public double[] Multiply(double[] a1, double[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            double[] m = new double[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] * a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public Array2d<double> Divide(Array2d<double> a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            Array2d<double> m = new Array2d<double>(a.RowCount, a.ColumnCount);

            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { m[row, column] = a[row, column] / c; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public Array2d<double> Divide(Array2d<double> a1, Array2d<double> a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            if (a1.RowCount != a2.RowCount || a1.ColumnCount != a2.ColumnCount)
            { throw (new ArgumentException("Input matrix dimensions are incorrect.")); }

            Array2d<double> m = new Array2d<double>(a1.RowCount, a1.ColumnCount);

            for (int row = 1; row <= a1.RowCount; row++)
            {
                for (int column = 1; column <= a1.ColumnCount; column++)
                { m[row, column] = a1[row, column] / a2[row, column]; }
            }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        public double[] Divide(double[] a, double c)
        {
            if (a == null)
                throw new ArgumentNullException();

            double[] m = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            { m[i] = a[i] / c; }

            return m;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        public double[] Divide(double[] a1, double[] a2)
        {
            if (a1 == null || a2 == null)
                throw new ArgumentNullException();

            double[] m = new double[a1.Length];

            for (int i = 0; i < a1.Length; i++)
            { m[i] = a1[i] / a2[i]; }

            return m;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(Array2d<double> a, out double MinimumValue, out double MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(Array2d<double> a, out double MinimumValue, out double MaximumValue, double[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = int.MaxValue;
            MaximumValue = int.MinValue;

            Dictionary<double, object> excluded = new Dictionary<double, object>();
            if (excludedValues != null)
            {
                foreach (double val in excludedValues)
                {
                    excluded.Add(val, null);
                }
            }

            double elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if (!excluded.ContainsKey(elementValue))
                    {
                        if (elementValue > MaximumValue) MaximumValue = elementValue;
                        if (elementValue < MinimumValue) MinimumValue = elementValue;
                    }
                }
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        public void FindMinimumAndMaximum(double[] a, out double MinimumValue, out double MaximumValue)
        {
            FindMinimumAndMaximum(a, out MinimumValue, out MaximumValue, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        public void FindMinimumAndMaximum(double[] a, out double MinimumValue, out double MaximumValue, double[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            MinimumValue = double.MaxValue;
            MaximumValue = double.MinValue;

            Dictionary<double, object> excluded = new Dictionary<double, object>();
            if (excludedValues != null)
            {
                foreach (double val in excludedValues)
                {
                    if (!excluded.ContainsKey(val))
                    {
                        excluded.Add(val, val);
                    }
                }
            }

            double elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!excluded.ContainsKey(elementValue))
                {
                    if (elementValue > MaximumValue) MaximumValue = elementValue;
                    if (elementValue < MinimumValue) MinimumValue = elementValue;
                }
            }

        }

        public ArrayStatistics<double> FindMinimumAndMaximum(Array2d<double> a, double[] excludedValues)
        {
            try
            {
                double minValue;
                double maxValue;
                FindMinimumAndMaximum(a, out minValue, out maxValue, excludedValues);
                ArrayStatistics<double> stats = new ArrayStatistics<double>();
                stats.MinimumValue = minValue;
                stats.MaximumValue = maxValue;
                return stats;

            }
            catch (Exception)
            {
                return null;
            }

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<double> GetUniqueValues(Array2d<double> a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<double> GetUniqueValues(Array2d<double> a, double[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<double, object> dict = new Dictionary<double, object>();
            List<double> list = new List<double>();

            if (excludedValues != null)
            {
                foreach (double val in excludedValues)
                { dict.Add(val, null); }
            }

            double elementValue;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                {
                    elementValue = a[row, column];
                    if (!dict.ContainsKey(elementValue))
                    {
                        dict.Add(elementValue, null);
                        list.Add(elementValue);
                    }
                }
            }

            return list;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public List<double> GetUniqueValues(double[] a)
        {
            return GetUniqueValues(a, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        public List<double> GetUniqueValues(double[] a, double[] excludedValues)
        {
            if (a == null)
                throw new ArgumentNullException();

            Dictionary<double, object> dict = new Dictionary<double, object>();
            List<double> list = new List<double>();

            if (excludedValues != null)
            {
                foreach (double val in excludedValues)
                { dict.Add(val, null); }
            }

            double elementValue;
            for (int i = 0; i < a.Length; i++)
            {
                elementValue = a[i];
                if (!dict.ContainsKey(elementValue))
                {
                    dict.Add(elementValue, null);
                    list.Add(elementValue);
                }
            }

            return list;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(Array2d<double> a, double dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int row = 1; row <= a.RowCount; row++)
            {
                for (int column = 1; column <= a.ColumnCount; column++)
                { if (a[row, column] == dataValue) count++; }
            }

            return count;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        public int ValueCount(double[] a, double dataValue)
        {
            if (a == null)
                throw new ArgumentNullException();

            int count = 0;
            for (int i = 0; i < a.Length; i++)
            { if (a[i] == dataValue) count++; }

            return count;
        }

        #endregion

    }


    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IArrayUtility<T>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        Array2d<T> Add(Array2d<T> a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        Array2d<T> Add(Array2d<T> a1, Array2d<T> a2);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        T[] Add(T[] a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        T[] Add(T[] a1, T[] a2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        Array2d<T> Subtract(Array2d<T> a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        Array2d<T> Subtract(Array2d<T> a1, Array2d<T> a2);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        T[] Subtract(T[] a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        T[] Subtract(T[] a1, T[] a2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        Array2d<T> Multiply(Array2d<T> a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        Array2d<T> Multiply(Array2d<T> a1, Array2d<T> a2);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        T[] Multiply(T[] a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        T[] Multiply(T[] a1, T[] a2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        Array2d<T> Divide(Array2d<T> a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        Array2d<T> Divide(Array2d<T> a1, Array2d<T> a2);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        T[] Divide(T[] a, T c);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <returns></returns>
        T[] Divide(T[] a1, T[] a2);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        void FindMinimumAndMaximum(Array2d<T> a, out T MinimumValue, out T MaximumValue);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        void FindMinimumAndMaximum(Array2d<T> a, out T MinimumValue, out T MaximumValue, T[] excludedValues);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        void FindMinimumAndMaximum(T[] a, out T MinimumValue, out T MaximumValue);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="MinimumValue"></param>
        /// <param name="MaximumValue"></param>
        /// <param name="excludedValues"></param>
        void FindMinimumAndMaximum(T[] a, out T MinimumValue, out T MaximumValue, T[] excludedValues);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        ArrayStatistics<T> FindMinimumAndMaximum(Array2d<T> a, T[] excludedValues);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        List<T> GetUniqueValues(Array2d<T> a);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        List<T> GetUniqueValues(Array2d<T> a, T[] excludedValues);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        List<T> GetUniqueValues(T[] a);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="excludedValues"></param>
        /// <returns></returns>
        List<T> GetUniqueValues(T[] a, T[] excludedValues);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        int ValueCount(Array2d<T> a, T dataValue);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dataValue"></param>
        /// <returns></returns>
        int ValueCount(T[] a, T dataValue);

    }

}
