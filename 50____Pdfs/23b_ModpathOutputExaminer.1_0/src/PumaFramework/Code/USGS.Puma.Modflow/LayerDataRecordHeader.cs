using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class LayerDataRecordHeader
    {
        #region Constructors
        public LayerDataRecordHeader()
        {
            Layer = 0;
            StressPeriod = 0;
            TimeStep = 0;
            Text = "";
            RowCount = 0;
            ColumnCount = 0;
            PeriodTime = 0f;
            TotalTime = 0f;
        }
        public LayerDataRecordHeader(int layer, int stressPeriod, int timeStep, string text, int columnCount, int rowCount, float periodTime, float totalTime)
        {
            Layer = layer;
            StressPeriod = stressPeriod;
            TimeStep = timeStep;
            Text = text;
            RowCount = rowCount;
            ColumnCount = columnCount;
            PeriodTime = periodTime;
            TotalTime = totalTime;
        }
        #endregion

        #region Public Members
        private int _Layer = 0;
        /// <summary>
        /// 
        /// </summary>
        public int Layer
        {
            get { return _Layer; }
            set { _Layer = value; }
        }

        private int _StressPeriod = 0;
        /// <summary>
        /// 
        /// </summary>
        public int StressPeriod
        {
            get { return _StressPeriod; }
            set { _StressPeriod = value; }
        }

        private int _TimeStep = 0;
        /// <summary>
        /// 
        /// </summary>
        public int TimeStep
        {
            get { return _TimeStep; }
            set { _TimeStep = value; }
        }

        private float _PeriodTime = 0;
        /// <summary>
        /// 
        /// </summary>
        public float PeriodTime
        {
            get { return _PeriodTime; }
            set { _PeriodTime = value; }
        }

        private float _TotalTime = 0;
        /// <summary>
        /// 
        /// </summary>
        public float TotalTime
        {
            get { return _TotalTime; }
            set { _TotalTime = value; }
        }

        private string _Text;
        /// <summary>
        /// 
        /// </summary>
        public string Text
        {
            get { return _Text; }
            set
            {
                _Text = "";
                if ( !string.IsNullOrEmpty(value)) _Text = value.ToUpper();
                if (_Text.Length > 16) _Text = _Text.Substring(0, 16);
            }
        }

        private int _RowCount;
        /// <summary>
        /// 
        /// </summary>
        public int RowCount
        {
            get { return _RowCount; }
            set { _RowCount = value; }
        }

        private int _ColumnCount;
        /// <summary>
        /// 
        /// </summary>
        public int ColumnCount
        {
            get { return _ColumnCount; }
            set { _ColumnCount = value; }
        }
        #endregion


    }
}
