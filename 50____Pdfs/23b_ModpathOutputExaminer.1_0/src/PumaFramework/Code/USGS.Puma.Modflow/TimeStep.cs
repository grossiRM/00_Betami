using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class TimeStep
    {
        #region Public Static Methods
        public static string CreateKey(int period, int step)
        {
            string key = "P" + period.ToString() + "S" + step.ToString();
            return key;
        }
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="period"></param>
        /// <param name="step"></param>
        /// <param name="periodTime"></param>
        /// <param name="totalTime"></param>
        /// <param name="periodType"></param>
        public TimeStep(int period, int step,float periodTime, float totalTime, StressPeriodType periodType)
        {
            _Period = period;
            _Step = step;
            _Key = TimeStep.CreateKey(_Period, _Step);
            PeriodTime = periodTime;
            TotalTime = totalTime;
            periodType = periodType;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="period"></param>
        /// <param name="step"></param>
        /// <param name="periodTime"></param>
        /// <param name="totalTime"></param>
        public TimeStep(int period, int step, float periodTime, float totalTime) : this(period, step, periodTime, totalTime, StressPeriodType.Undefined) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="period"></param>
        /// <param name="step"></param>
        /// <param name="periodTime"></param>
        /// <param name="periodType"></param>
        public TimeStep(int period, int step, float periodTime, StressPeriodType periodType) : this(period, step, periodTime, 0.0f, periodType) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="period"></param>
        /// <param name="step"></param>
        public TimeStep(int period, int step) : this(period, step, 0.0f, 0.0f, StressPeriodType.Undefined) { }
        #endregion

        #region Public Properties
        private int _Period;
        /// <summary>
        /// 
        /// </summary>
        public int Period
        {
            get { return _Period; }
        }

        private int _Step;
        /// <summary>
        /// 
        /// </summary>
        public int Step
        {
            get { return _Step; }
        }

        private float _PeriodTime;
        /// <summary>
        /// 
        /// </summary>
        public float PeriodTime
        {
            get { return _PeriodTime; }
            set { _PeriodTime = value; }
        }

        private float _TotalTime;
        /// <summary>
        /// 
        /// </summary>
        public float TotalTime
        {
            get { return _TotalTime; }
            set { _TotalTime = value; }
        }

        private StressPeriodType _PeriodType;
        /// <summary>
        /// 
        /// </summary>
        public StressPeriodType PeriodType
        {
            get { return _PeriodType; }
            set { _PeriodType = value; }
        }

        private string _Key;
        /// <summary>
        /// 
        /// </summary>
        public string Key
        {
            get { return _Key; }
        }
        #endregion
    }
}
