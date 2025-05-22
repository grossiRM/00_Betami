using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.Modflow;

namespace USGS.ModflowTrainingTools
{
    public partial class ModflowMetadaEditDialog : Form
    {

        #region Constructors
        public ModflowMetadaEditDialog()
        {
            InitializeComponent();
            _Metadata = null;
            CurrentBasemapFile = "";
        }
        #endregion

        #region Public Properties

        private ModflowMetadata _Metadata;
        /// <summary>
        /// 
        /// </summary>
        public ModflowMetadata Metadata
        {
            get { return _Metadata; }
            set { 
                _Metadata = value;
                SetData(_Metadata);
            }
        }

        private string _CurrentBasemapFile;
        /// <summary>
        /// 
        /// </summary>
        public string CurrentBasemapFile
        {
            get { return _CurrentBasemapFile; }
            set { 
                _CurrentBasemapFile = value;
                _CurrentBasemapFile = _CurrentBasemapFile.Trim();
                if (String.IsNullOrEmpty(_CurrentBasemapFile))
                {
                    btnUseCurrentBasemap.Enabled = false;
                }
                else
                {
                    btnUseCurrentBasemap.Enabled = true;
                }
            }
        }

        #endregion

        #region Event Handlers
        private void btnCalculateAngle_Click(object sender, EventArgs e)
        {
            BaselinePointEditDialog dialog = new BaselinePointEditDialog();
            dialog.OriginX = double.Parse(txtOriginX.Text);
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                double x = dialog.X;
                double y = dialog.Y;
                double originX = double.Parse(txtOriginX.Text);
                double originY = double.Parse(txtOriginY.Text);
                double a = Metadata.GridGeoReference.ComputeAngle(originX, originY, x, y);
                txtGeoReferenceGridAngle.Text = a.ToString();
            }
        }

        private void txtGeoReferenceGridAngle_Validating(object sender, CancelEventArgs e)
        {
            try
            {
                double a = double.Parse(txtGeoReferenceGridAngle.Text);
                if (a < -90 || a > 90)
                {
                    MessageBox.Show("Enter a number in the range -90 to 90.");
                    e.Cancel = true;
                }
            }
            catch
            {
                MessageBox.Show("Enter a number in the range -90 to 90.");
                e.Cancel = true;
            }
        }

        private void txtOriginX_Validating(object sender, CancelEventArgs e)
        {
            try
            {
                double a = double.Parse(txtOriginX.Text);
            }
            catch
            {
                MessageBox.Show("Enter a number.");
                e.Cancel = true;
            }
        }

        private void txtOriginY_Validating(object sender, CancelEventArgs e)
        {
            try
            {
                double a = double.Parse(txtOriginY.Text);
            }
            catch
            {
                MessageBox.Show("Enter a number.");
                e.Cancel = true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnSave_Click(object sender, EventArgs e)
        {
            GetData(Metadata);
            this.DialogResult = DialogResult.OK;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnBasemapBrowse_Click(object sender, EventArgs e)
        {
            if (Metadata != null)
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.InitialDirectory = Metadata.SourcefileDirectory;
                dialog.Filter = "*.pbm (basemap files)|*.pbm|*.* (all files)|*.*";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    txtDefaultBasemap.Text = dialog.FileName;
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnNoBasemap_Click(object sender, EventArgs e)
        {
            txtDefaultBasemap.Text = "";
        }

        #endregion

        #region Private Methods
        /// <summary>
        /// 
        /// </summary>
        private void SetData(ModflowMetadata metadata)
        {
            if (metadata != null)
            {
                txtGeoReferenceGridAngle.Text = metadata.GridGeoReference.Angle.ToString();
                txtOriginX.Text = metadata.GridGeoReference.OriginX.ToString();
                txtOriginY.Text = metadata.GridGeoReference.OriginY.ToString();
                txtDefaultBasemap.Text = metadata.BasemapFile;
            }
        }

        private void GetData(ModflowMetadata metadata)
        {
            if (metadata != null)
            {
                metadata.GridGeoReference.OriginX = double.Parse(txtOriginX.Text);
                metadata.GridGeoReference.OriginY = double.Parse(txtOriginY.Text);
                metadata.GridGeoReference.Angle = double.Parse(txtGeoReferenceGridAngle.Text);
                metadata.BasemapFile = txtDefaultBasemap.Text;
            }
        }
        #endregion

        private void btnUseCurrentBasemap_Click(object sender, EventArgs e)
        {
            txtDefaultBasemap.Text = CurrentBasemapFile;
        }


    }
}
