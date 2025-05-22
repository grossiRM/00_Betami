using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Core
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks></remarks>
    public class ArrayStatistics<T>
    {
        /// <summary>
        /// 
        /// </summary>
        private T _MinimumValue;
        /// <summary>
        /// Gets or sets the minimum value.
        /// </summary>
        /// <value>The minimum value.</value>
        /// <remarks></remarks>
        public T MinimumValue
        {
            get { return _MinimumValue; }
            set { _MinimumValue = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private T _MaximumValue;
        /// <summary>
        /// Gets or sets the maximum value.
        /// </summary>
        /// <value>The maximum value.</value>
        /// <remarks></remarks>
        public T MaximumValue
        {
            get { return _MaximumValue; }
            set { _MaximumValue = value; }
        }

    }
}
