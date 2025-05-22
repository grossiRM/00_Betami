using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// Representation of a MODFLOW stress period
    /// </summary>
    /// <remarks></remarks>
    public class StressPeriod
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="StressPeriodDef"/> class.
        /// </summary>
        /// <param name="periodLength">Length of the period.</param>
        /// <param name="timeStepCount">The time step count.</param>
        /// <param name="timeStepMultiplier">The time step multiplier.</param>
        /// <param name="periodType">Type of the period.</param>
        /// <remarks></remarks>
        public StressPeriod(float periodLength, int timeStepCount, float timeStepMultiplier, StressPeriodType periodType)
        {
            PeriodLength = periodLength;
            TimeStepCount = timeStepCount;
            TimeStepMultiplier = timeStepMultiplier;
            PeriodType = periodType;
        }
        #endregion

        #region Public Members
        /// <summary>
        /// 
        /// </summary>
        private float _PeriodLength = 0.0f;
        /// <summary>
        /// Gets or sets the length of the period.
        /// </summary>
        /// <value>The length of the period.</value>
        /// <remarks></remarks>
        public float PeriodLength
        {
            get
            {
                return _PeriodLength;
            }
            set
            {
                _PeriodLength=value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _TimeStepCount = 0;
        /// <summary>
        /// Gets or sets the time step count.
        /// </summary>
        /// <value>The time step count.</value>
        /// <remarks></remarks>
        public int TimeStepCount
        {
            get
            {
                return _TimeStepCount;
            }
            set
            {
                _TimeStepCount=value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private float _TimeStepMultiplier = 0.0f;
        /// <summary>
        /// Gets or sets the time step multiplier.
        /// </summary>
        /// <value>The time step multiplier.</value>
        /// <remarks></remarks>
        public float TimeStepMultiplier
        {
            get
            {
                return _TimeStepMultiplier;
            }
            set
            {
                _TimeStepMultiplier=value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private StressPeriodType _PeriodType = StressPeriodType.SteadyState;
        /// <summary>
        /// Gets or sets the type of the period.
        /// </summary>
        /// <value>The type of the period.</value>
        /// <remarks></remarks>
        public StressPeriodType PeriodType
        {
            get
            {
                return _PeriodType;
            }
            set
            {
                _PeriodType=value;
            }
        }

        /// <summary>
        /// Gets the time step lengths.
        /// </summary>
        /// <returns>An array containing the computed time step lengths</returns>
        /// <remarks></remarks>
        public Array1d<float> GetTimeStepLengths()
        {
            Array1d<float> a = new Array1d<float>(this._TimeStepCount);
            if (this.TimeStepCount < 1)
            { return a; }

            float stepSize = GetInitialTimeStepLength();
            a[1] = stepSize;
            for (int i = 2; i < this.TimeStepCount + 1; i++)
            {
                a[i] *= this.TimeStepMultiplier;
            }
            return a;
        }
        /// <summary>
        /// Gets the values of period time for all time steps.
        /// </summary>
        /// <returns>An array of period time values</returns>
        /// <remarks></remarks>
        public Array1d<float> GetPeriodTimes()
        {
            Array1d<float> a = new Array1d<float>(this.TimeStepCount);
            if (this.TimeStepCount < 1)
            { return a; }

            Array1d<float> ts = GetTimeStepLengths();

            a[1] = ts[1];
            for (int i = 2; i < this.TimeStepCount + 1; i++)
            {
                if (i == this.TimeStepCount)
                {
                    a[i] = this.PeriodLength;
                }
                else
                {
                    a[i] += ts[i];
                }
            }
            return a;

        }

        #endregion

        #region Private Members
        /// <summary>
        /// Gets the initial length of the time step.
        /// </summary>
        /// <param name="period">The period.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private float GetInitialTimeStepLength()
        {
            if (this.TimeStepCount == 1)
            {
                return this.PeriodLength;
            }
            else if (this.TimeStepMultiplier == 1.0f)
            {
                return this.PeriodLength / Convert.ToSingle(this.TimeStepCount);
            }
            else
            {
                float power = Convert.ToSingle(Math.Pow(Convert.ToDouble(this.TimeStepMultiplier), Convert.ToDouble(this.TimeStepCount)));
                float t = this.PeriodLength * (this.TimeStepMultiplier - 1.0f) / (power - 1);
                return t;
            }
        }

        #endregion
    }
}
