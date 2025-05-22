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
    public partial class EndpointQueryFilter : QueryFilter, IEndpointQueryFilter
    {
        public EndpointQueryFilter()
        {
            InitializeComponent();

            cboInitialCell.Items.Add("any cell");
            cboInitialCell.Items.Add("the specified cell");

            cboFinalCell.Items.Add("any cell location");
            cboFinalCell.Items.Add("the specified cell");

            cboInitialZone.Items.Add("any zone");
            cboInitialZone.Items.Add("the specified zone");

            cboFinalZone.Items.Add("any zone");
            cboFinalZone.Items.Add("the specified zone");

            cboGroup.Items.Add("any group");
            cboGroup.Items.Add("the specified group");

            cboTimeFilter.Items.Add("any value");
            cboTimeFilter.Items.Add("less than the specified time");
            cboTimeFilter.Items.Add("greater than the specified time");

            cboInitialTime.Items.Add("any value");
            cboInitialTime.Items.Add("the specified value");

            ResetData();

        }

        #region Event Handlers
        private void cboTimeFilter_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboTimeFilter.SelectedIndex == 0)
            {
                panelTravelTime.Visible = false;
            }
            else
            {
                panelTravelTime.Visible = true;
            }
        }
        private void cboFinalCell_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboFinalCell.SelectedIndex == 0)
            {
                panelFinalCell.Visible = false;
            }
            else
            {
                panelFinalCell.Visible = true;
            }
        }
        private void cboInitialCell_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboInitialCell.SelectedIndex == 0)
            {
                panelInitialCell.Visible = false;
            }
            else
            {
                panelInitialCell.Visible = true;
            }

        }
        private void cboInitialZone_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboInitialZone.SelectedIndex == 0)
            {
                panelInitialZone.Visible = false;
            }
            else
            {
                panelInitialZone.Visible = true;
            }
        }
        private void cboGroup_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboGroup.SelectedIndex == 0)
            {
                panelGroup.Visible = false;
            }
            else
            {
                panelGroup.Visible = true;
            }

        }
        private void cboFinalZone_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboFinalZone.SelectedIndex == 0)
            {
                panelFinalZone.Visible = false;
            }
            else
            {
                panelFinalZone.Visible = true;
            }
        }

        private void cboInitialTime_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cboInitialTime.SelectedIndex == 0)
            {
                panelInitialTime.Visible = false;
            }
            else
            {
                panelInitialTime.Visible = true;
            }
        }

        #endregion

        #region Public Methods
        private int _InitialLayer;
        public int InitialLayer
        {
            get { return _InitialLayer; }
            set { _InitialLayer = value; }
        }

        private int _InitialRow;
        public int InitialRow
        {
            get { return _InitialRow; }
            set { _InitialRow = value; }
        }

        private int _InitialColumn;
        public int InitialColumn
        {
            get { return _InitialColumn; }
            set { _InitialColumn = value; }
        }

        private int _FinalLayer;
        public int FinalLayer
        {
            get { return _FinalLayer; }
            set { _FinalLayer = value; }
        }

        private int _FinalRow;
        public int FinalRow
        {
            get { return _FinalRow; }
            set { _FinalRow = value; }
        }

        private int _FinalColumn;
        public int FinalColumn
        {
            get { return _FinalColumn; }
            set { _FinalColumn = value; }
        }

        private int _InitialZone;
        public int InitialZone
        {
            get { return _InitialZone; }
            set { _InitialZone = value; }
        }

        private int _FinalZone;
        public int FinalZone
        {
            get { return _FinalZone; }
            set { _FinalZone = value; }
        }

        private int _Group;
        public int Group
        {
            get { return _Group; }
            set { _Group = value; }
        }

        private float _TravelTime;
        public float TravelTime
        {
            get { return _TravelTime; }
            set { _TravelTime = value; }
        }

        private float _InitialTime;
        /// <summary>
        /// 
        /// </summary>
        public float InitialTime
        {
            get { return _InitialTime; }
            set { _InitialTime = value; }
        }

        private int _InitialCellOption;
        public int InitialCellOption
        {
            get { return _InitialCellOption; }
            set { _InitialCellOption = value; }
        }

        private int _FinalCellOption;
        public int FinalCellOption
        {
            get { return _FinalCellOption; }
            set { _FinalCellOption = value; }
        }

        private int _InitialZoneOption;
        public int InitialZoneOption
        {
            get { return _InitialZoneOption; }
            set { _InitialZoneOption = value; }
        }

        private int _FinalZoneOption;
        public int FinalZoneOption
        {
            get { return _FinalZoneOption; }
            set { _FinalZoneOption = value; }
        }

        private int _GroupOption;
        public int GroupOption
        {
            get { return _GroupOption; }
            set { _GroupOption = value; }
        }

        private int _TravelTimeOption;
        public int TravelTimeOption
        {
            get { return _TravelTimeOption; }
            set { _TravelTimeOption = value; }
        }

        private int _InitialTimeOption;
        /// <summary>
        /// 
        /// </summary>
        public int InitialTimeOption
        {
            get { return _InitialTimeOption; }
            set { _InitialTimeOption = value; }
        }

        private List<EndpointRecord> _DataSource;
        /// <summary>
        /// 
        /// </summary>
        public List<EndpointRecord> DataSource
        {
            get { return _DataSource; }
            set { _DataSource = value; }
        }

        #endregion

        #region Private and Protected Methods
        protected override void GetDialogData()
        {
            base.GetDialogData();

            InitialLayer = int.Parse(txtInitialLayer.Text);
            InitialRow = int.Parse(txtInitialRow.Text);
            InitialColumn = int.Parse(txtInitialColumn.Text);

            FinalLayer = int.Parse(txtFinalLayer.Text);
            FinalRow = int.Parse(txtFinalRow.Text);
            FinalColumn = int.Parse(txtFinalColumn.Text);

            InitialZone = int.Parse(txtInitialZone.Text);

            FinalZone = int.Parse(txtFinalZone.Text);

            Group = int.Parse(txtGroup.Text);

            TravelTime = float.Parse(txtTravelTime.Text);

            InitialTime = 0.0f;
            if (cboInitialTimeValue.SelectedIndex > -1)
            {
                InitialTime = float.Parse(cboInitialTimeValue.Text);
            }

            InitialCellOption = cboInitialCell.SelectedIndex;
            FinalCellOption = cboFinalCell.SelectedIndex;
            InitialZoneOption = cboInitialZone.SelectedIndex;
            FinalZoneOption = cboFinalZone.SelectedIndex;
            GroupOption = cboGroup.SelectedIndex;
            TravelTimeOption = cboTimeFilter.SelectedIndex;
            InitialTimeOption = cboInitialTime.SelectedIndex;
        
        }
        protected override void SetDialogData()
        {
            base.SetDialogData();

            txtInitialLayer.Text = InitialLayer.ToString();
            txtInitialRow.Text = InitialRow.ToString();
            txtInitialColumn.Text = InitialColumn.ToString();
            txtFinalLayer.Text = FinalLayer.ToString();
            txtFinalRow.Text = FinalRow.ToString();
            txtFinalColumn.Text = FinalColumn.ToString();
            txtInitialZone.Text = InitialZone.ToString();
            txtFinalZone.Text = FinalZone.ToString();
            txtGroup.Text = Group.ToString();
            txtTravelTime.Text = TravelTime.ToString();

            cboInitialTimeValue.Items.Clear();
            if (DataSource != null)
            {
                List<float> timeValues = EndpointQueryProcessor.GetInitialTimeValues(DataSource as IEnumerable<EndpointRecord>);
                int index = 0;
                foreach (float timeValue in timeValues)
                {
                    cboInitialTimeValue.Items.Add(timeValue.ToString());
                    if (timeValue == InitialTime)
                    {
                        index = cboInitialTimeValue.Items.Count - 1;
                    }
                }
                if (cboInitialTimeValue.Items.Count > 0)
                {
                    cboInitialTimeValue.SelectedIndex = index;
                }
                
            }

            cboInitialCell.SelectedIndex = InitialCellOption;
            cboFinalCell.SelectedIndex = FinalCellOption;
            cboInitialZone.SelectedIndex = InitialZoneOption;
            cboFinalZone.SelectedIndex = FinalZoneOption;
            cboGroup.SelectedIndex = GroupOption;
            cboTimeFilter.SelectedIndex = TravelTimeOption;
            cboInitialTime.SelectedIndex = InitialTimeOption;

        }
        protected override void ResetData()
        {
            base.ResetData();
            InitialLayer = 1;
            InitialRow = 1;
            InitialColumn = 1;
            FinalLayer = 1;
            FinalRow = 1;
            FinalColumn = 1;
            InitialZone = 1;
            FinalZone = 1;
            Group = 1;
            TravelTime = 0.0f;
            InitialCellOption = 0;
            FinalCellOption = 0;
            InitialZoneOption = 0;
            FinalZoneOption = 0;
            GroupOption = 0;
            TravelTimeOption = 0;
            InitialTimeOption = 0;
            DataSource = null;
        }
        protected override string OutputQueryFilterSummary()
        {
            if (FilteringIsOn)
            {
                StringBuilder sb = new StringBuilder();

                // Initial cell condition
                if (InitialCellOption == 0)
                {
                    sb.AppendLine("Particles may start in any cell location.");
                }
                else
                {
                    sb.Append("Show only the particles that start in cell (layer, row, column): ");
                    sb.Append(InitialLayer).Append(", ");
                    sb.Append(InitialRow).Append(", ");
                    sb.Append(InitialColumn).AppendLine();
                }

                // Final cell condition
                if (FinalCellOption == 0)
                {
                    sb.AppendLine("Particles may terminate in any cell location.");
                }
                else
                {
                    sb.Append("Show only the particles that terminate in cell (layer, row, column): ");
                    sb.Append(FinalLayer).Append(", ");
                    sb.Append(FinalRow).Append(", ");
                    sb.Append(FinalColumn).AppendLine();
                }

                // Initial zone
                sb.AppendLine();
                if (InitialZoneOption == 0)
                {
                    sb.AppendLine("Particles may start in any zone.");
                }
                else
                {
                    sb.Append("Show only the particles that start in zone ");
                    sb.Append(InitialZone).AppendLine();
                }

                // Final Zone
                if (FinalZoneOption == 0)
                {
                    sb.AppendLine("Particles may terminate in any zone.");
                }
                else
                {
                    sb.Append("Show only the particles that terminate in zone ");
                    sb.Append(FinalZone).AppendLine();
                }

                // Travel time
                sb.AppendLine();
                if (TravelTimeOption == 0)
                {
                    sb.AppendLine("Particles may have any value of travel time");
                }
                else if (TravelTimeOption == 1)
                {
                    sb.Append("Show only the particles that have a travel time value less than ");
                    sb.Append(TravelTime);
                }
                else if (TravelTimeOption == 2)
                {
                    sb.Append("Show only the particles that have a travel time value greater than ");
                    sb.Append(TravelTime);

                }

                // Initial time
                sb.AppendLine();
                if (InitialTimeOption == 0)
                {
                    sb.AppendLine("Particles may have any value of initial time");
                }
                else
                {
                    sb.Append("Show only the particles that have an initial time value equal to ");
                    sb.Append(InitialTime);
                }

                return sb.ToString(0, sb.Length);
            }
            else
            {
                return "No filter is defined. Show all endpoint records.";
            }

        }
        #endregion

        #region IEndpointQueryFilter Members

        public List<EndpointRecord> Execute()
        {
            if (DataSource == null)
            {
                throw new InvalidOperationException("Endpoint record data source is not define.");
            }

            if (FilteringIsOn)
            {
                IEnumerable<EndpointRecord> recs = DataSource as IEnumerable<EndpointRecord>;
                // Query for initial cell
                if (InitialCellOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByCell(recs, InitialLayer, InitialRow, InitialColumn, EndpointLocationTypes.InitialPoint);
                }

                // Query for final cell
                if (FinalCellOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByCell(recs, FinalLayer, FinalRow, FinalColumn, EndpointLocationTypes.FinalPoint);
                }

                // Query for initial zone
                if (InitialZoneOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByZone(recs, InitialZone, EndpointLocationTypes.InitialPoint);
                }

                // Query for final zone
                if (FinalZoneOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByZone(recs, FinalZone, EndpointLocationTypes.FinalPoint);
                }

                // Query for group
                if (GroupOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByGroup(recs, Group);
                }

                // Query for travel time if a value is specified
                if (TravelTimeOption > 0)
                {
                    if (TravelTimeOption == 2)
                    { recs = EndpointQueryProcessor.FilterByTravelTime(recs, TravelTime, QueryEquality.GreaterThan); }
                    else
                    { recs = EndpointQueryProcessor.FilterByTravelTime(recs, TravelTime, QueryEquality.LessThan); }
                }

                if (InitialTimeOption > 0)
                {
                    recs = EndpointQueryProcessor.FilterByTime(recs, InitialTime, EndpointLocationTypes.InitialPoint, QueryEquality.Equal);
                }

                // Execute the accumulated query
                if (recs != null)
                { return recs.ToList<EndpointRecord>(); }
                else
                { return null; }
            }
            else
            {
                return DataSource;
            }

        }

        #endregion

        private void btnOK_Click(object sender, EventArgs e)
        {

        }


    }
}
