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
    public partial class PathlineQueryFilter : QueryFilter, IPathlineQueryFilter
    {
        public PathlineQueryFilter()
        {
            InitializeComponent();

            cboParticleIDOption.Items.Add("any value");
            cboParticleIDOption.Items.Add("equal to the specified particle ID");

            ResetData();

        }

        #region Event Handlers
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
        private int _ParticleID;
        public int ParticleID
        {
            get { return _ParticleID; }
            set { _ParticleID = value; }
        }

        private int _ParticleIDOption;
        public int ParticleIDOption
        {
            get { return _ParticleIDOption; }
            set { _ParticleIDOption = value; }
        }

        private List<PathlineRecord> _DataSource;
        /// <summary>
        /// 
        /// </summary>
        public List<PathlineRecord> DataSource
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
        }
        protected override void SetDialogData()
        {
            base.SetDialogData();
            txtParticleID.Text = ParticleID.ToString();
            cboParticleIDOption.SelectedIndex = ParticleIDOption;
        }
        protected override void ResetData()
        {
            base.ResetData();
            ParticleID = 1;
            ParticleIDOption = 0;
        }
        protected override string OutputQueryFilterSummary()
        {
            if (FilteringIsOn)
            {
                StringBuilder sb = new StringBuilder();

                // Particle ID condition
                if (ParticleIDOption == 0)
                {
                    sb.AppendLine("Show pathline records for all particle ID values.");
                }
                else
                {
                    sb.Append("Show only the pathline records for particle ID = ");
                    sb.Append(ParticleID).AppendLine();
                }

                return sb.ToString(0, sb.Length);
            }
            else
            {
                return "No filter is defined. Show all pathline records.";
            }
            
        }
        #endregion

        #region IPathlineQueryFilter Members
        public List<PathlineRecord> Execute()
        {
            if (DataSource == null)
            {
                throw new InvalidOperationException("The pathline record data source is not defined.");
            }

            if (FilteringIsOn)
            {
                IEnumerable<PathlineRecord> recs = DataSource as IEnumerable<PathlineRecord>;
                // Query particle ID
                if (ParticleIDOption > 0)
                {
                    recs = PathlineQueryProcessor.FilterByID(recs, ParticleID);
                }

                // Execute the accumulated query
                if (recs != null)
                { return recs.ToList<PathlineRecord>(); }
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
