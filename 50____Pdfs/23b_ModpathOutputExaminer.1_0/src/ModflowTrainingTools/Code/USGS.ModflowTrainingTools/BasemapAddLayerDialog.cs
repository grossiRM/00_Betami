using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using USGS.Puma.UI.MapViewer;

namespace USGS.ModflowTrainingTools
{
    public partial class BasemapAddLayerDialog : Form
    {
        #region
        private Dictionary<string, FileInfo> _Shapefiles = null;

        #endregion

        public BasemapAddLayerDialog()
        {
            InitializeComponent();
            lvwShapefiles.View = View.List;
            lvwShapefiles.HideSelection = false;
            lvwShapefiles.MultiSelect = true;
            _Shapefiles = new Dictionary<string, FileInfo>();
        }

        public BasemapAddLayerDialog(string pathname)
            : this()
        {
            DirectoryInfo directory = new DirectoryInfo(pathname);
            if (directory.Exists)
            {
                FileInfo[] files = directory.GetFiles("*.shp");
                foreach (FileInfo file in files)
                {
                    lvwShapefiles.Items.Add(file.Name);
                    _Shapefiles.Add(file.Name, file);
                }
            }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            List<FileInfo> list = new List<FileInfo>();
            foreach (ListViewItem item in lvwShapefiles.SelectedItems)
            {
                if (_Shapefiles.ContainsKey(item.Text))
                {
                    list.Add(_Shapefiles[item.Text]);
                }
            }
            _SelectedFiles = list.ToArray();

            this.DialogResult =DialogResult.OK;
            Hide();
        }

        private void FBasemapAddLayer_Load(object sender, EventArgs e)
        {

        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            _Shapefiles = null;
            this.DialogResult = DialogResult.Cancel;
            Hide();
        }


        private FileInfo[] _SelectedFiles = null;
        public FileInfo[] SelectedFiles
        {
            get 
            {
                if (_SelectedFiles == null) _SelectedFiles = new FileInfo[0];
                return _SelectedFiles; 
            }
        }

    }
}
