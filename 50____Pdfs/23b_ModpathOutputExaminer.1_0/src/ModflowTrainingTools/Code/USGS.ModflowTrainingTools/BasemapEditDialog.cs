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
    public partial class BasemapEditDialog : Form
    {
        #region Private Fields
        private int _SelectedIndex = -1;
        private bool _UpdateLayerProp = true;
        #endregion

        #region Constructors
        public BasemapEditDialog()
        {
            InitializeComponent();
            Basemap = null;

            lblRGB.Text = "";
            lvwLayers.View = View.Details;
            lvwLayers.MultiSelect = false;
            lvwLayers.HideSelection = false;
            ColumnHeader column = new ColumnHeader();
            column.Text = "Layers";
            column.Width = lvwLayers.Width - 50;
            lvwLayers.Columns.Add(column);

            for (int i = 0; i < 10; i++)
            {
                cboSize.Items.Add(i + 1);
            }

            cboStyle.Items.Add("1 - Solid line, solid filled polygon, or circle marker");
            cboStyle.Items.Add("2 - Dashed line, unfilled polygon, or square marker");

            chkVisible.Checked = false;

        }
        public BasemapEditDialog(Basemap basemap)
            : this()
        {
            Basemap = basemap;
        }
        #endregion

        #region Event Handlers
        private void btnOK_Click(object sender, EventArgs e)
        {
            if (_SelectedIndex > -1) UpdateLayerProp();
            this.DialogResult = DialogResult.OK;
            Hide();
        }
        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
            Hide();
        }

        private void lvwLayers_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lvwLayers.SelectedItems.Count > 0)
            {
                if (_SelectedIndex > -1) UpdateLayerProp();
                _SelectedIndex = lvwLayers.SelectedIndices[0];
            }
            else
            {
                if (_SelectedIndex > -1 && _UpdateLayerProp) UpdateLayerProp();
                _SelectedIndex = -1;
            }
            UpdateLayerPropertyDialog(_SelectedIndex);
        }
        private void btnCustomColor_Click(object sender, EventArgs e)
        {
            Color c = picColor.BackColor;
            ColorDialog dialog = new ColorDialog();
            dialog.Color = c;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                picColor.BackColor = dialog.Color;
                lblRGB.Text = "RGB: "
                    + dialog.Color.R.ToString() + ", "
                    + dialog.Color.G.ToString() + ", "
                    + dialog.Color.B.ToString();
            }
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            AddLayer();
        }
        private void btnRemove_Click(object sender, EventArgs e)
        {
            if (lvwLayers.SelectedIndices.Count > 0)
            {
                int index = lvwLayers.SelectedIndices[0];
                _UpdateLayerProp = false;
                _Basemap.RemoveLayer(index);
                lvwLayers.Items.RemoveAt(index);
                _UpdateLayerProp = true;

            }
        }

        private void btnTop_Click(object sender, EventArgs e)
        {
            if (_SelectedIndex < 0) return;

            UpdateLayerProp();
            _Basemap.MoveToTop(_SelectedIndex);
            _SelectedIndex = -1;
            lvwLayers.Items.Clear();
            for (int i = 0; i < _Basemap.LayerCount; i++)
            {
                lvwLayers.Items.Add(_Basemap[i].LocalName);
            }
            if (lvwLayers.Items.Count > 0) lvwLayers.Items[0].Selected = true;

        }
        private void btnBottom_Click(object sender, EventArgs e)
        {
            if (_SelectedIndex < 0) return;

            UpdateLayerProp();
            _Basemap.MoveToBottom(_SelectedIndex);
            _SelectedIndex = -1;
            lvwLayers.Items.Clear();
            for (int i = 0; i < _Basemap.LayerCount; i++)
            {
                lvwLayers.Items.Add(_Basemap[i].LocalName);
            }
            if (lvwLayers.Items.Count > 0) lvwLayers.Items[_Basemap.LayerCount - 1].Selected = true;

        }
        private void btnUp_Click(object sender, EventArgs e)
        {
            if (_SelectedIndex < 1) return;

            UpdateLayerProp();
            _Basemap.MoveUp(_SelectedIndex);
            int newIndex = _SelectedIndex - 1;
            _SelectedIndex = -1;
            lvwLayers.Items.Clear();
            for (int i = 0; i < _Basemap.LayerCount; i++)
            {
                lvwLayers.Items.Add(_Basemap[i].LocalName);
            }
            if (lvwLayers.Items.Count > 0) lvwLayers.Items[newIndex].Selected = true;

        }
        private void btnDown_Click(object sender, EventArgs e)
        {
            if (_SelectedIndex < 0 || _SelectedIndex == _Basemap.LayerCount - 1) return;

            UpdateLayerProp();
            _Basemap.MoveDown(_SelectedIndex);
            int newIndex = _SelectedIndex + 1;
            _SelectedIndex = -1;
            lvwLayers.Items.Clear();
            for (int i = 0; i < _Basemap.LayerCount; i++)
            {
                lvwLayers.Items.Add(_Basemap[i].LocalName);
            }
            if (lvwLayers.Items.Count > 0) lvwLayers.Items[newIndex].Selected = true;

        }

        #endregion

        #region Public Properties
        private Basemap _Basemap = null;
        public Basemap Basemap
        {
            get
            { return _Basemap;}
            set
            { LoadBasemap(value); }
        }
        #endregion

        #region Private Methods
        private void AddLayer()
        {
            BasemapAddLayerDialog dialog = new BasemapAddLayerDialog(txtBasemap.Text);
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                StringBuilder sb = new StringBuilder();
                int count = 0;
                foreach (FileInfo file in dialog.SelectedFiles)
                {
                    int index = _Basemap.FindIndex(file.Name);
                    if (index < 0)
                    {
                        BasemapLayer layer = new BasemapLayer(file.Name, System.Drawing.Color.Black, 1, 1, file.Name + " data");
                        _Basemap.AddLayer(layer);
                        count++;
                    }
                }
                if (count > 0) LoadBasemap(_Basemap);
            }
        }
        private void LoadBasemap(Basemap basemap)
        {
            _Basemap = null;
            if (basemap == null)
            {
                txtBasemap.Text = "";
                cboStyle.SelectedItem = null;
                cboSize.SelectedItem = null;
                chkVisible.Checked = false;
                lvwLayers.Items.Clear();
            }
            else
            {
                _Basemap = basemap;
                string absoluteBasemapDir = Utilities.ConvertRelativePathToAbsolute(_Basemap.BasemapDirectory);
                txtBasemap.Text = absoluteBasemapDir;
                lvwLayers.Items.Clear();
                for (int i = 0; i < _Basemap.LayerCount; i++)
                {
                    lvwLayers.Items.Add(_Basemap[i].LocalName);
                }
                if (lvwLayers.Items.Count > 0) lvwLayers.Items[0].Selected = true;
            }
        }
        private void UpdateLayerProp()
        {
            _Basemap[_SelectedIndex].Description = txtDescription.Text;
            _Basemap[_SelectedIndex].Color = picColor.BackColor;
            _Basemap[_SelectedIndex].Size = cboSize.SelectedIndex + 1;
            _Basemap[_SelectedIndex].Style = cboStyle.SelectedIndex + 1;
            _Basemap[_SelectedIndex].IsVisible = chkVisible.Checked;
        }
        private void UpdateLayerPropertyDialog(int index)
        {
            if (index < 0)
            {
                gboxProperties.Text = "Properties";
                cboStyle.SelectedItem = null;
                cboSize.SelectedItem = null;
                chkVisible.Checked = false;
                txtDescription.Text = "";
                lblRGB.Text = "";
                picColor.BackColor = this.BackColor;
            }
            else
            {
                gboxProperties.Text = gboxProperties.Text + " -- " + lvwLayers.Items[index].Text;
                txtDescription.Text = _Basemap[_SelectedIndex].Description;
                cboSize.SelectedIndex = _Basemap[_SelectedIndex].Size - 1;
                cboStyle.SelectedIndex = _Basemap[_SelectedIndex].Style - 1;
                chkVisible.Checked = _Basemap[_SelectedIndex].IsVisible;
                picColor.BackColor = _Basemap[_SelectedIndex].Color;
                lblRGB.Text = "RGB: "
                    + picColor.BackColor.R.ToString() + ", "
                    + picColor.BackColor.G.ToString() + ", "
                    + picColor.BackColor.B.ToString();
            }
        }
        #endregion




    }
}
