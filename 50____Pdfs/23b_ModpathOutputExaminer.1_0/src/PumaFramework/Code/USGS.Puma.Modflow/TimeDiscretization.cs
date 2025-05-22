using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// This class manages a group of stress periods that define the time discretization for a MODFLOW simulation.
    /// </summary>
    /// <remarks></remarks>
    public class TimeDiscretization
    {
        #region Private Fields
        /// <summary>
        /// 
        /// </summary>
        private Collection<StressPeriod> _Periods;
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public TimeDiscretization()
        {
            _Periods = new Collection<StressPeriod>();
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="SimulationStressPeriods"/> class.
        /// </summary>
        /// <param name="periods">The periods.</param>
        /// <remarks></remarks>
        public TimeDiscretization(IList<StressPeriod> periods) : this()
        {
            if (periods != null)
            {
                for (int i = 0; i < periods.Count; i++)
                {
                    if (periods[i] == null)
                    {
                        throw new InvalidOperationException("The stress period list contains one or more undefined values.");
                    }
                    this.AddStressPeriod(periods[i]);
                }
            }
        }
        #endregion

        #region Public Members
        /// <summary>
        /// Adds a stress period with the properties of the specified <see cref="USGS.Puma.Modflow.StressPeriod"/> data item.
        /// </summary>
        /// <param name="period">The period.</param>
        /// <returns>An integer indicating the index of the stress period that was added.</returns>
        /// <remarks>A new instance of <see cref="USGS.Puma.Modflow.StressPeriod"/> is created from the input data item.</remarks>
        public int AddStressPeriod(StressPeriod period)
        {
            if (period == null)
            {
                throw new ArgumentNullException();
            }
            StressPeriod p = period;
            _Periods.Add(new StressPeriod(p.PeriodLength, p.TimeStepCount, p.TimeStepMultiplier, p.PeriodType));
            return _Periods.Count;
        }
        /// <summary>
        /// Adds a stress period with the specified properties.
        /// </summary>
        /// <param name="periodLength">Length of the stress period.</param>
        /// <param name="timeStepCount">The time step count.</param>
        /// <param name="timeStepMultiplier">The time step multiplier.</param>
        /// <param name="periodType">Type of stress period.</param>
        /// <returns>An integer indicating the index of the stress period that was added.</returns>
        /// <remarks></remarks>
        public int AddStressPeriod(float periodLength, int timeStepCount, float timeStepMultiplier, StressPeriodType periodType)
        {
            _Periods.Add(new StressPeriod(periodLength, timeStepCount, timeStepMultiplier, periodType));
            return _Periods.Count;
        }
        /// <summary>
        /// Inserts a stress period at specified the period number.
        /// </summary>
        /// <param name="periodNumber">The period number.</param>
        /// <param name="period">The period.</param>
        /// <remarks></remarks>
        public void InsertStressPeriod(int periodNumber, StressPeriod period)
        {
            _Periods.Insert(periodNumber - 1, period);
        }
        /// <summary>
        /// Removes the stress period.
        /// </summary>
        /// <param name="periodNumber">The period number.</param>
        /// <remarks></remarks>
        public void RemoveStressPeriod(int periodNumber)
        {
            if (periodNumber < 1 || periodNumber > _Periods.Count)
            {
                throw new ArgumentOutOfRangeException("periodNumber");
            }
            _Periods.RemoveAt(periodNumber - 1);
        }
        /// <summary>
        /// Removes all stress periods.
        /// </summary>
        /// <remarks></remarks>
        public void Clear()
        {
            _Periods.Clear();
        }
        /// <summary>
        /// Toes the array.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public StressPeriod[] ToArray()
        {
            return _Periods.ToArray<StressPeriod>();
        }
        /// <summary>
        /// Get the <see cref="USGS.Puma.Modflow.StressPeriod"/> for the specified stress period number.
        /// </summary>
        /// <param name="periodNumber">The period number.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public StressPeriod StressPeriod(int periodNumber)
        {
            if (periodNumber < 1 || periodNumber > _Periods.Count)
            {
                throw new ArgumentOutOfRangeException("periodNumber");
            }
            return _Periods[periodNumber - 1];
        }
        /// <summary>
        /// Builds a time step collection that includes all stress periods.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public TimeStepCollection BuildTimeStepCollection()
        {
            TimeStepCollection tsc = new TimeStepCollection();

            // Generate time steps and add them to the collection.
            float totalTime = 0.0f;
            for (int np = 1; np < _Periods.Count + 1; np++)
            {
                StressPeriod p = this.StressPeriod(np);
                CreatePeriodTimeSteps(np, totalTime, tsc);
                totalTime += p.PeriodLength;
            }

            return tsc;
        }
        /// <summary>
        /// Builds a time step collection for the specified stress period number.
        /// </summary>
        /// <param name="periodNumber">The period number.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public TimeStepCollection BuildTimeStepCollection(int periodNumber)
        {
            TimeStepCollection tsc = new TimeStepCollection();
            float totalTime = 0.0f;
            for (int i = 1; i < periodNumber; i++)
            {
                totalTime += this.StressPeriod(i).PeriodLength;
            }
            CreatePeriodTimeSteps(periodNumber, totalTime, tsc);
            return tsc;
        }
        /// <summary>
        /// Gets the number of stress periods
        /// </summary>
        /// <remarks></remarks>
        public int StressPeriodCount
        {
            get { return _Periods.Count; }
        }

        #endregion

        #region Private Members
        /// <summary>
        /// Creates the period time steps.
        /// </summary>
        /// <param name="periodNumber">The period number.</param>
        /// <param name="totalTime">The total time.</param>
        /// <param name="periodCollection">The period collection.</param>
        /// <remarks></remarks>
        private void CreatePeriodTimeSteps(int periodNumber, float totalTime, TimeStepCollection periodCollection)
        {
            StressPeriod p = this.StressPeriod(periodNumber);
            Array1d<float> periodTimes = p.GetTimeStepLengths();
            int np = periodNumber;

            if (p.TimeStepCount < 1)
            { return; }

            for (int ns = 1; ns < p.TimeStepCount + 1; ns++)
            {
                TimeStep ts = new TimeStep(np, ns);
                ts.PeriodTime = periodTimes[ns];
                ts.TotalTime = totalTime + ts.PeriodTime;
                periodCollection.Add(ts);
            }
        }


        #endregion

    }
}
