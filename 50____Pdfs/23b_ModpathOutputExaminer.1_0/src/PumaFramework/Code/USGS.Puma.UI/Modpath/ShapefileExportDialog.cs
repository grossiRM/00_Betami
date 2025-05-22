using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.IO;
using USGS.Puma.Modpath;
using USGS.Puma.Modpath.IO;
using USGS.Puma.NTS;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;
using USGS.Puma.NTS.IO;
using GeoAPI.Geometries;

namespace USGS.Puma.UI.Modpath
{
    public partial class ModpathShapefileExportDialog : Form
    {
        /// <summary>
        /// 
        /// </summary>
        public ModpathShapefileExportDialog()
        {
            InitializeComponent();

            cboEndpoint.Items.Add("Export initial locations");
            cboEndpoint.Items.Add("Export final locations");
            cboEndpoint.SelectedIndex = 0;

            txtStatus.Text = "";
            EndpointsShapefileName = "";
            PathlinesShapefileName = "";
            TimeseriesShapefileName = "";
            SimulationType = 1;

        }

        #region Event Handlers
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void chkExportPahlines_CheckedChanged(object sender, EventArgs e)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnExport_Click(object sender, EventArgs e)
        {
            txtStatus.Text = txtStatus.Text + "Exporting shapefiles ..." + Environment.NewLine;
            btnExport.Enabled = false;
            btnCancel.Enabled = false;
            this.Refresh();

            if (Export())
            {
                btnExport.Enabled = true;
                btnCancel.Enabled = true;
            }
            else
            {
                btnCancel.Enabled = true;
                txtStatus.Text = txtStatus.Text + "Error exporting shapefiles." + Environment.NewLine;
                MessageBox.Show("A problem occured while exporting shapefiles.");
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.OK;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ShapefileExportDialog_Shown(object sender, EventArgs e)
        {
            txtStatus.Text = "";
        }

        #endregion

        #region Public Methods
        private int _SimulationType;
        /// <summary>
        /// 
        /// </summary>
        public int SimulationType
        {
            get { return _SimulationType; }
            set 
            { 
                _SimulationType = value;
                chkExportEndpoints.Enabled = true;
                panelEndpoints.Enabled = true;
                chkExportEndpoints.Checked = false;
                chkExportPathlines.Checked = false;
                chkExportTimeseries.Checked = false;

                if (SimulationType == 2)
                {
                    chkExportPathlines.Enabled = true;
                    chkExportPathlines.Checked = true;
                    panelPathlines.Enabled = true;
                    chkExportTimeseries.Checked = false;
                    chkExportTimeseries.Enabled = false;
                    panelTimeseries.Enabled = false;
                }
                else if (SimulationType == 3)
                {
                    chkExportPathlines.Checked = false;
                    chkExportPathlines.Enabled = false;
                    panelPathlines.Enabled = false;
                    chkExportTimeseries.Enabled = true;
                    chkExportTimeseries.Checked = true;
                    panelTimeseries.Enabled = true;
                }
                else
                {
                    _SimulationType = 1;
                    chkExportEndpoints.Checked = true;
                    chkExportPathlines.Enabled = false;
                    panelPathlines.Enabled = false;
                    chkExportTimeseries.Enabled = false;
                    panelTimeseries.Enabled = false;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool ExportEndpointsSelected
        {
            get
            {
                return chkExportEndpoints.Checked;
            }
            set
            {
                chkExportEndpoints.Checked = value;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public bool ExportPathlinesSelected
        {
            get
            {
                return chkExportPathlines.Checked;
            }
            set
            {
                chkExportPathlines.Checked = value;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public bool ExportTimeseriesSelected
        {
            get
            {
                return chkExportTimeseries.Checked;
            }
            set
            {
                chkExportTimeseries.Checked = value;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public EndpointLocationTypes EndpointOption
        {
            get 
            { 
               if(cboEndpoint.SelectedIndex==0)
               {
                   return EndpointLocationTypes.InitialPoint;
               }
               else
               {
                   return EndpointLocationTypes.FinalPoint;
               }
            }
            set 
            {
                if (value == EndpointLocationTypes.InitialPoint)
                {
                    cboEndpoint.SelectedIndex = 0;
                }
                else
                {
                    cboEndpoint.SelectedIndex = 1;
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public string EndpointsShapefileName 
        {
            get
            {
                return txtEndpointsName.Text.Trim();
            }
            set
            {
                txtEndpointsName.Text = value.Trim();
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public string PathlinesShapefileName
        {
            get
            {
                return txtPathlinesName.Text;
            }
            set
            {
                txtPathlinesName.Text = value;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public string TimeseriesShapefileName
        {
            get
            {
                return txtTimeseriesName.Text;
            }
            set
            {
                txtTimeseriesName.Text = value;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        public string ExportFolder
        {
            get
            {
                return txtExportFolder.Text;
            }
            set
            {
                txtExportFolder.Text = value;
            }
        }

        private List<EndpointRecord> _EndpointData;
        /// <summary>
        /// 
        /// </summary>
        public List<EndpointRecord> EndpointData
        {
            get { return _EndpointData; }
            set { _EndpointData = value; }
        }

        private List<TimeseriesRecord> _TimeseriesData;
        /// <summary>
        /// 
        /// </summary>
        public List<TimeseriesRecord> TimeseriesData
        {
            get { return _TimeseriesData; }
            set { _TimeseriesData = value; }
        }

        private List<PathlineRecord> _PathlineData;
        /// <summary>
        /// 
        /// </summary>
        public List<PathlineRecord> PathlineData
        {
            get { return _PathlineData; }
            set { _PathlineData = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public bool Export()
        {
            if(ExportEndpointsSelected)
            {
                try
                {
                    FeatureCollection features = ParticleFeatures.CreateEndpointFeatures(EndpointData, EndpointOption);
                    EsriShapefileIO.Export(features, ExportFolder, EndpointsShapefileName);
                    //USGS.Puma.Modpath.IO.ParticleShapefileExporter.ExportEndpoints(ExportFolder, EndpointsShapefileName, EndpointData, EndpointOption);
                    txtStatus.Text = txtStatus.Text + "Shapefile exported: " + EndpointsShapefileName + Environment.NewLine;
                }
                catch (Exception e)
                {
                    txtStatus.Text = txtStatus.Text + "Problem exporting: " + EndpointsShapefileName + Environment.NewLine + e.Message;
                    return false;
                }
            }
            if(ExportPathlinesSelected)
            {
                try
                {
                    FeatureCollection features = ParticleFeatures.CreatePathlineFeatures(PathlineData, EndpointData, false);
                    EsriShapefileIO.Export(features, ExportFolder, PathlinesShapefileName);
                    //USGS.Puma.Modpath.IO.ParticleShapefileExporter.ExportPathlines(ExportFolder, PathlinesShapefileName, PathlineData, false);
                    txtStatus.Text = txtStatus.Text + "Shapefile exported: " + PathlinesShapefileName + Environment.NewLine;
                }
                catch (Exception e)
                {
                    txtStatus.Text = txtStatus.Text + "Problem exporting: " + PathlinesShapefileName + Environment.NewLine + e.Message;
                    return false;
                }
            }
            if(ExportTimeseriesSelected)
            {
                try
                {
                    FeatureCollection features = ParticleFeatures.CreateTimeseriesFeatures(TimeseriesData);
                    EsriShapefileIO.Export(features, ExportFolder, TimeseriesShapefileName);
                    //USGS.Puma.Modpath.IO.ParticleShapefileExporter.ExportTimeseries(ExportFolder, TimeseriesShapefileName, TimeseriesData);
                    txtStatus.Text = txtStatus.Text + "Shapefile exported: " + TimeseriesShapefileName + Environment.NewLine;
                }
                catch (Exception e)
                {
                    txtStatus.Text = txtStatus.Text + "Problem exporting: " + TimeseriesShapefileName + Environment.NewLine + e.Message;
                    return false;
                }
            }
            return true;
        }

        #endregion

        #region Private and Protected Methods


        #endregion

    }
}
