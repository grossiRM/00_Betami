using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Utilities
{
    /// <summary>
    /// 
    /// </summary>
    public class GenericNumberUtility : IGenericNumberUtility<int>, IGenericNumberUtility<float>, IGenericNumberUtility<double>
    {
        #region IGenericNumber<int> Members
        int IGenericNumberUtility<int>.DefaultNoDataValue
        {
            get { return System.Convert.ToInt32(-999999); }
        }

        int IGenericNumberUtility<int>.MinValue
        {
            get { return int.MinValue; }
        }

        int IGenericNumberUtility<int>.MaxValue
        {
            get { return int.MaxValue; }
        }

        bool IGenericNumberUtility<int>.Equal(int n1, int n2)
        {
            return (n1 == n2);
        }

        bool IGenericNumberUtility<int>.GreaterThan(int n1, int n2)
        {
            return (n1 > n2);
        }

        bool IGenericNumberUtility<int>.GreaterThanOrEqual(int n1, int n2)
        {
            return (n1 >= n2);
        }

        bool IGenericNumberUtility<int>.LessThan(int n1, int n2)
        {
            return (n1 < n2);
        }

        bool IGenericNumberUtility<int>.LessThanOrEqual(int n1, int n2)
        {
            return (n1 <= n2);
        }

        int IGenericNumberUtility<int>.Parse(string s)
        {
            return int.Parse(s);
        }

        bool IGenericNumberUtility<int>.TryParse(string s, out int result)
        {
            return int.TryParse(s, out result);
        }

        int IGenericNumberUtility<int>.ConvertTo(object n)
        {
            return System.Convert.ToInt32(n);
        }

        int IGenericNumberUtility<int>.Add(int n1, int n2)
        {
            return n1 + n2;
        }

        int IGenericNumberUtility<int>.Subtract(int n1, int n2)
        {
            return n1 - n2;
        }

        int IGenericNumberUtility<int>.Multiply(int n1, int n2)
        {
            return n1 * n2;
        }

        int IGenericNumberUtility<int>.Divide(int n1, int n2)
        {
            return n1 / n2;
        }

        #endregion

        #region IGenericNumber<float> Members
        float IGenericNumberUtility<float>.DefaultNoDataValue
        {
            get { return System.Convert.ToSingle(-1.0E+35); }
        }

        float IGenericNumberUtility<float>.MinValue
        {
            get { return float.MinValue; }
        }

        float IGenericNumberUtility<float>.MaxValue
        {
            get { return float.MaxValue; }
        }

        bool IGenericNumberUtility<float>.Equal(float n1, float n2)
        {
            return (n1 == n2);
        }

        bool IGenericNumberUtility<float>.GreaterThan(float n1, float n2)
        {
            return (n1 > n2);
        }

        bool IGenericNumberUtility<float>.GreaterThanOrEqual(float n1, float n2)
        {
            return (n1 >= n2);
        }

        bool IGenericNumberUtility<float>.LessThan(float n1, float n2)
        {
            return (n1 < n2);
        }

        bool IGenericNumberUtility<float>.LessThanOrEqual(float n1, float n2)
        {
            return (n1 <= n2);
        }

        float IGenericNumberUtility<float>.Parse(string s)
        {
            return float.Parse(s);
        }

        bool IGenericNumberUtility<float>.TryParse(string s, out float result)
        {
            return float.TryParse(s, out result);
        }

        float IGenericNumberUtility<float>.ConvertTo(object n)
        {
            return System.Convert.ToSingle(n);
        }

        float IGenericNumberUtility<float>.Add(float n1, float n2)
        {
            return n1 + n2;
        }

        float IGenericNumberUtility<float>.Subtract(float n1, float n2)
        {
            return n1 - n2;
        }

        float IGenericNumberUtility<float>.Multiply(float n1, float n2)
        {
            return n1 * n2;
        }

        float IGenericNumberUtility<float>.Divide(float n1, float n2)
        {
            return n1 / n2;
        }

        #endregion

        #region IGenericNumber<double> Members
        double IGenericNumberUtility<double>.DefaultNoDataValue
        {
            get { return System.Convert.ToDouble(-1.0E+35); }
        }

        double IGenericNumberUtility<double>.MinValue
        {
            get { return double.MinValue; }
        }

        double IGenericNumberUtility<double>.MaxValue
        {
            get { return double.MaxValue; }
        }

        bool IGenericNumberUtility<double>.Equal(double n1, double n2)
        {
            return (n1 == n2);
        }

        bool IGenericNumberUtility<double>.GreaterThan(double n1, double n2)
        {
            return (n1 > n2);
        }

        bool IGenericNumberUtility<double>.GreaterThanOrEqual(double n1, double n2)
        {
            return (n1 >= n2);
        }

        bool IGenericNumberUtility<double>.LessThan(double n1, double n2)
        {
            return (n1 < n2);
        }

        bool IGenericNumberUtility<double>.LessThanOrEqual(double n1, double n2)
        {
            return (n1 <= n2);
        }

        double IGenericNumberUtility<double>.Parse(string s)
        {
            return double.Parse(s);
        }

        bool IGenericNumberUtility<double>.TryParse(string s, out double result)
        {
            return double.TryParse(s, out result);
        }

        double IGenericNumberUtility<double>.ConvertTo(object n)
        {
            return System.Convert.ToDouble(n);
        }

        double IGenericNumberUtility<double>.Add(double n1, double n2)
        {
            return n1 + n2;
        }

        double IGenericNumberUtility<double>.Subtract(double n1, double n2)
        {
            return n1 - n2;
        }

        double IGenericNumberUtility<double>.Multiply(double n1, double n2)
        {
            return n1 * n2;
        }

        double IGenericNumberUtility<double>.Divide(double n1, double n2)
        {
            return n1 / n2;
        }

        #endregion

    }

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IGenericNumberUtility<T>
    {
        T DefaultNoDataValue { get; }
        T MinValue { get; }
        T MaxValue { get; }
        bool Equal(T n1, T n2);
        bool GreaterThan(T n1, T n2);
        bool GreaterThanOrEqual(T n1, T n2);
        bool LessThan(T n1, T n2);
        bool LessThanOrEqual(T n1, T n2);
        T Parse(string s);
        bool TryParse(string s, out T result);
        T ConvertTo(object n);
        T Add(T n1, T n2);
        T Subtract(T n1, T n2);
        T Multiply(T n1, T n2);
        T Divide(T n1, T n2);
    }

}
