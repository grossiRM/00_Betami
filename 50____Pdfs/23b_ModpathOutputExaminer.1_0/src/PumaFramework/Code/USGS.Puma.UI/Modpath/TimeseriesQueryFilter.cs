using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.Modpath.IO;

namespace USGS.Puma.UI.Modpath
{
    public partial class TimeseriesQueryFilter : QueryFilter, ITimeseriesQueryFilter
    {
        public TimeseriesQueryFilter()
        {
            InitializeComponent();

            cboParticleIDOption.Items.Add("any value");
            cboParticleIDOption.Items.Add("equal to the specified value");

            cboTimePointOption.Items.Add("any value");
            cboTimePointOption.Items.Add("equal to the specified value");

            ResetData();

        
        }

        #region Event Handlers
        private void cboTimePointOption_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboTimePointOption.SelectedIndex == 0)
            {
                panelTimePoint.Visible = false;
            }
            else
            {
                panelTimePoint.Visible = true;
            }
        }
        private void cboParticleIDOption_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboParticleIDOption.SelectedIndex == 0)
            {
                panelParticleID.Visible = false;
            }
            else
            {
                panelParticleID.Visible = true;
            }
        }

        #endregion

        #region Public Methods
        private int _ParticleIDOption;
        public int ParticleIDOption
        {
            get { return _ParticleIDOption; }
            set { _ParticleIDOption = value; }
        }

        private int _ParticleID;
        public int ParticleID
        {
            get { return _ParticleID; }
            set { _ParticleID = value; }
        }

        private int _TimePointOption;
        public int TimePointOption
        {
            get { return _TimePointOption; }
            set { _TimePointOption = value; }
        }

        private int _TimePoint;
        public int TimePoint
        {
            get { return _TimePoint; }
            set { _TimePoint = value; }
        }

        private List<TimeseriesRecord> _DataSource;
        /// <summary>
        /// 
        /// </summary>
        public List<TimeseriesRecord> DataSource
        {
            get { return _DataSource; }
            set { _DataSource = value; }
        }

        #endregion

        #region Private and Protected Methods
        protected override void GetDialogData()
        {
            base.GetDialogData();
            ParticleID = int.Parse(txtParticleID.Text);
            ParticleIDOption = cboParticleIDOption.SelectedIndex;
            TimePoint = int.Parse(txtTimePoint.Text);
            TimePointOption = cboTimePointOption.SelectedIndex;
        }
        protected override void SetDialogData()
        {
            base.SetDialogData();
            txtParticleID.Text = ParticleID.ToString();
            cboParticleIDOption.SelectedIndex = ParticleIDOption;
            txtTimePoint.Text = TimePoint.ToString();
            cboTimePointOption.SelectedIndex = TimePointOption;
        }
        protected override void ResetData()
        {
            base.ResetData();
            ParticleID = 1;
            ParticleIDOption = 0;
            TimePoint = 0;
            TimePointOption = 0;
        }
        protected override string OutputQueryFilterSummary()
        {
            if (FilteringIsOn)
            {
                StringBuilder sb = new StringBuilder();

                // Time point condiditon
                if (TimePointOption == 0)
                {
                    sb.AppendLine("Show timeseries records for all time points.");
                }
                else
                {
                    sb.Append("Show only the timeseries records for time point ");
                    sb.Append(TimePoint).AppendLine();
                }

                // Particle ID condition
                sb.AppendLine();
                if (ParticleIDOption == 0)
                {
                    sb.AppendLine("Show timeseries records for all particle ID values.");
                }
                else
                {
                    sb.Append("Show only the timeseries records for particle ID = ");
                    sb.Append(ParticleID).AppendLine();
                }

                return sb.ToString(0, sb.Length);
            }
            else
            {
                return "No filter is defined. Show all endpoint records.";
            }

        }
        #endregion

        #region ITimeseriesQueryFilter Members
        public List<TimeseriesRecord> Execute()
        {
            if (DataSource == null)
            {
                throw new InvalidOperationException("The timeseries record data source is not defined.");
            }

            if (FilteringIsOn)
            {
                IEnumerable<TimeseriesRecord> recs = DataSource as IEnumerable<TimeseriesRecord>;
                // Query time point
                if (TimePointOption > 0)
                {
                    recs = TimeseriesQueryProcessor.FilterByTimePoint(recs, TimePoint);
                }

                // Query particle ID
                if (ParticleIDOption > 0)
                {
                    recs = TimeseriesQueryProcessor.FilterByID(recs, ParticleID);
                }

                // Execute the accumulated query
                if (recs != null)
                { return recs.ToList<TimeseriesRecord>(); }
                else
                { return null; }
            }
            else
            {
                return DataSource;
            }

        }

        #endregion

    }
}
