using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.Modpath.IO;

namespace USGS.Puma.UI.Modpath
{
    public partial class ParticleRecordTextView : UserControl
    {
        
        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="recordType"></param>
        public ParticleRecordTextView()
        {
            InitializeComponent();

            _CurrentRecordIndex = -1;
            _Records = null;

        }
        #endregion

        #region Events and Event Dispatchers
        public event EventHandler<EventArgs> CurrentRecordIndexChanged;
        protected void OnCurrentRecordIndexChanged()
        {
            if (CurrentRecordIndexChanged != null)
            {
                CurrentRecordIndexChanged(this, new EventArgs());
            }
        }


        #endregion

        #region Event Handlers
        private void rtxRecords_SelectionChanged(object sender, EventArgs e)
        {
            if (RecordCount == 0)
            { 
                _CurrentRecordIndex = -1;
            }
            else
            {
                _CurrentRecordIndex = rtxRecords.GetLineFromCharIndex(rtxRecords.SelectionStart);
            }

            // Fire the event
            OnCurrentRecordIndexChanged();

        }
        #endregion

        #region Public Properties
        /// <summary>
        /// 
        /// </summary>
        public ParticleRecordTypes RecordType
        {
            get 
            {
                if (_Records == null)
                { return ParticleRecordTypes.Undefined; }

                if (_Records is List<EndpointRecord>)
                { return ParticleRecordTypes.Endpoint; }
                else if (_Records is List<PathlineRecord>)
                { return ParticleRecordTypes.Pathline; }
                else if (_Records is List<TimeseriesRecord>)
                { return ParticleRecordTypes.Timeseries; }
                else
                { return ParticleRecordTypes.Undefined; }

            }
        }

        private object _Records;
        /// <summary>
        /// 
        /// </summary>
        public object Records
        {
            get 
            { 
                return _Records; 
            }
            set 
            {
                if (value == null)
                {
                    Reset();
                }
                else
                {
                    if (value is List<EndpointRecord>)
                    {
                        _Records = value;
                        List<EndpointRecord> rec = _Records as List<EndpointRecord>;
                        if (rec.Count > 0)
                        { _CurrentRecordIndex = 0; }
                        else
                        { _CurrentRecordIndex = -1; }
                    }
                    else if (value is List<PathlineRecord>)
                    {
                        _Records = value;
                        List<PathlineRecord> rec = _Records as List<PathlineRecord>;
                        if (rec.Count > 0)
                        { _CurrentRecordIndex = 0; }
                        else
                        { _CurrentRecordIndex = -1; }
                    }
                    else if (value is List<TimeseriesRecord>)
                    {
                        _Records = value;
                        List<TimeseriesRecord> rec = _Records as List<TimeseriesRecord>;
                        if (rec.Count > 0)
                        { _CurrentRecordIndex = 0; }
                        else
                        { _CurrentRecordIndex = -1; }
                    }
                    else
                    { throw new ArgumentException("The specified records list is not a particle record list."); }

                    // Refresh the control
                    this.Refresh();

                    // Fire the CurrentRecordIndexChanged event
                    OnCurrentRecordIndexChanged();

                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// 
        private int _CurrentRecordIndex;
        /// <summary>
        /// 
        /// </summary>
        public int CurrentRecordIndex
        {
            get { return _CurrentRecordIndex; }
        }

        /// <summary>
        /// 
        /// </summary>
        public object CurrentRecord
        {
            get 
            {
                if (CurrentRecordIndex < 0)
                { return null; }
                else
                {
                    switch (RecordType)
                    {
                        case ParticleRecordTypes.Endpoint:
                            return (Records as List<EndpointRecord>)[CurrentRecordIndex];
                        case ParticleRecordTypes.Pathline:
                            return (Records as List<PathlineRecord>)[CurrentRecordIndex];
                        case ParticleRecordTypes.Timeseries:
                            return (Records as List<TimeseriesRecord>)[CurrentRecordIndex];
                        default:
                            return null;
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int RecordCount
        {
            get
            {
                if (Records == null)
                { return 0; }
                else
                {
                    switch (RecordType)
                    {
                        case ParticleRecordTypes.Endpoint:
                            return (Records as List<EndpointRecord>).Count;
                        case ParticleRecordTypes.Pathline:
                            return (Records as List<PathlineRecord>).Count;
                        case ParticleRecordTypes.Timeseries:
                            return (Records as List<TimeseriesRecord>).Count;
                        default:
                            return 0;
                    }

                }
            }
        }
        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        public void Refresh()
        {
            if (Records == null)
            {
                LoadDataSource(null);
            }
            else
            {
                string[] lines = CreateTextRecords(_Records);
                LoadDataSource(lines);
            }

            // Fire the CurrentRecordIndexChanged event
            OnCurrentRecordIndexChanged();

        }
        /// <summary>
        /// 
        /// </summary>
        public void Reset()
        {
            _CurrentRecordIndex = -1;
            _Records = null;
            Refresh();
        }
        /// <summary>
        /// 
        /// </summary>
        public virtual void CloseDataset()
        {
            // For now, just treat this as an alias for Reset.
            Reset();
        }
        #endregion

        #region Private and Protected Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="records"></param>
        protected void LoadDataSource(string[] lines)
        {
            if (lines == null)
            {
                _CurrentRecordIndex = -1;
                rtxRecords.Text = "";
                return;
            }
            if (lines.Length == 0)
            {
                _CurrentRecordIndex = -1;
                rtxRecords.Text = "";
                return;
            }
            rtxRecords.Lines = lines;
            _CurrentRecordIndex = 0;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="records"></param>
        /// <returns></returns>
        protected string[] CreateTextRecords(object records)
        {
            List<string> lines = new List<string>();
            switch (RecordType)
            {
                case ParticleRecordTypes.Endpoint:
                    List<EndpointRecord> epLines = records as List<EndpointRecord>;
                    EndpointFileWriter epWriter = new EndpointFileWriter();
                    for (int i = 0; i < epLines.Count; i++)
                    {
                        lines.Add(epWriter.CreateTextRecord(epLines[i]));
                    }
                    return lines.ToArray();

                case ParticleRecordTypes.Pathline:
                    List<PathlineRecord> plLines = records as List<PathlineRecord>;
                    PathlineFileWriter plWriter = new PathlineFileWriter();
                    for (int i = 0; i < plLines.Count; i++)
                    {
                        lines.Add(plWriter.CreateTextRecord(plLines[i]));
                    }
                    return lines.ToArray();

                case ParticleRecordTypes.Timeseries:
                    List<TimeseriesRecord> tsLines = records as List<TimeseriesRecord>;
                    TimeseriesFileWriter tsWriter = new TimeseriesFileWriter();
                    for (int i = 0; i < tsLines.Count; i++)
                    {
                        lines.Add(tsWriter.CreateTextRecord(tsLines[i]));
                    }
                    return lines.ToArray();

                default:
                    return null;
            }
        }
        #endregion




    }
}
