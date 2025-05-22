using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using USGS.Puma.Core;
using USGS.Puma.Modflow;

namespace USGS.ModflowTrainingTools
{
    public partial class ModflowOutputContoursEditDialog : Form
    {
        public ModflowOutputContoursEditDialog()
        {
            InitializeComponent();

            cboContourLineWidth.Items.Clear();
            for (int i = 0; i < 5; i++)
            {
                cboContourLineWidth.Items.Add(i + 1);
            }
            cboContourLineWidth.SelectedIndex = 0;

            panelContourColor.BackColor = Color.Black;

        }

        private LayerDataRecordHeader _SelectedHeader = null;

        private ContourEngineData _ContourData = null;
        /// <summary>
        /// 
        /// </summary>
        public ContourEngineData ContourData
        {
            get { return _ContourData; }
            set 
            {
                if (value != null)
                {
                    InitializeData(value);
                }
                else
                {
                    _ContourData = null;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="headers"></param>
        private void InitializeData(ContourEngineData contourData)
        {
            string key = null;
            LayerDataRecordHeaderCollection headers = null;

            tvwContents.BeginUpdate();
            tvwContents.Nodes.Clear();
            tvwContents.EndUpdate();

            if (contourData == null)
            { return; }

            _ContourData = contourData;

            txtAddExcludedValue.Text = "";
            txtReferenceContour.Text = contourData.ReferenceContour.ToString();
            txtSpecifiedContourInterval.Text = contourData.ConstantContourInterval.ToString();
            panelContourColor.BackColor = contourData.ContourColor;
            cboContourLineWidth.SelectedIndex = contourData.ContourLineWidth - 1;

            switch (contourData.ContourIntervalOption)
            {
                case ContourIntervalOption.AutomaticConstantInterval:
                    radioButtonAutomaticContourInterval.Checked = true;
                    break;
                case ContourIntervalOption.SpecifiedConstantInterval:
                    radioButtonSpecifiedContourInterval.Checked = true;
                    break;
                default:
                    break;
            }

            headers = null;
            if (contourData.ContourSourceFile != null)
            {
                headers = contourData.ContourSourceFile.ReadAllRecordHeaders();
            }

            if (headers == null)
            {
                _SelectedHeader = null;
                BuildSelectedDataLayerLabel(_SelectedHeader);
            }

            // Turn off treeview drawing while building node structure
            if (contourData.ContourSourceFile != null)
            {
                tvwContents.BeginUpdate();

                string rootNodeText = "Layer data";
                if (headers.Count > 0)
                {
                    rootNodeText = headers[0].Text.TrimStart(' ');
                }
                TreeNode contentsRootNode = tvwContents.Nodes.Add(rootNodeText);
                contentsRootNode.ImageIndex = 0;
                contentsRootNode.SelectedImageIndex = 0;
                TreeNode node = null;
                string keyLayer = "";

                int count = 0;
                foreach (LayerDataRecordHeader header in headers)
                {
                    count += 1;
                    key = header.StressPeriod.ToString();
                    if (contentsRootNode.Nodes.ContainsKey(key))
                    { node = contentsRootNode.Nodes[key]; }
                    else
                    {
                        node = contentsRootNode.Nodes.Add(key, "Period " + header.StressPeriod.ToString());
                        node.ImageIndex = 0;
                        node.SelectedImageIndex = 0;
                    }

                    key = key + "," + header.TimeStep.ToString();
                    if (node.Nodes.ContainsKey(key))
                    { node = node.Nodes[key]; }
                    else
                    {
                        node = node.Nodes.Add(key, "Step " + header.TimeStep.ToString());
                        node.ImageIndex = 0;
                        node.SelectedImageIndex = 0;
                    }

                    keyLayer = header.Layer.ToString();
                    key = key + "," + keyLayer;
                    if (node.Nodes.ContainsKey(key))
                    { node = node.Nodes[key]; }
                    else
                    {
                        node = node.Nodes.Add(key, "Layer " + header.Layer.ToString());
                        node.ImageIndex = 2;
                        node.SelectedImageIndex = 2;
                    }
                    node.Tag = header;
                    if (count == 1)
                    {
                        tvwContents.SelectedNode = node;
                    }

                }

                if (contentsRootNode.Nodes.Count == 1)
                {
                    contentsRootNode.Expand();
                    contentsRootNode.Nodes[0].Expand();
                    if (contentsRootNode.Nodes[0].Nodes.Count == 1)
                    { contentsRootNode.Nodes[0].Nodes[0].Expand(); }
                }

                // Re-establish treeview drawing mode.
                tvwContents.EndUpdate();
            }
            else
            {
                tabData.TabPages.Remove(tabPageDataLayer);
            }

            // Set no flow values
            lbxExcludedValues.Items.Clear();
            for (int i = 0; i < contourData.ExcludedValues.Count; i++)
            {
                if (!lbxExcludedValues.Items.Contains(contourData.ExcludedValues[i]))
                {
                    lbxExcludedValues.Items.Add(contourData.ExcludedValues[i]);
                }
            }

            if (headers != null)
            {
                if (contourData.SelectedDataLayer == null)
                {
                    _SelectedHeader = headers[0];
                }
                else
                {
                    _SelectedHeader = contourData.SelectedDataLayer;
                }
            }
            else
            {
                _SelectedHeader = null;
            }

            lblSelectedDataLayer.Text = BuildSelectedDataLayerLabel(_SelectedHeader);

        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            UpdateContourData();
            this.DialogResult = DialogResult.OK;

        }

        private void UpdateContourData()
        {
            ContourData.SelectedDataLayer = _SelectedHeader;
            ContourData.ReferenceContour = float.Parse(txtReferenceContour.Text);
            ContourData.ConstantContourInterval = float.Parse(txtSpecifiedContourInterval.Text);
            ContourData.ContourColor = panelContourColor.BackColor;
            ContourData.ContourLineWidth = cboContourLineWidth.SelectedIndex + 1;

            if (radioButtonAutomaticContourInterval.Checked)
            {
                ContourData.ContourIntervalOption = ContourIntervalOption.AutomaticConstantInterval;
            }
            else if (radioButtonSpecifiedContourInterval.Checked)
            {
                ContourData.ContourIntervalOption = ContourIntervalOption.SpecifiedConstantInterval;
            }

            ContourData.ExcludedValues.Clear();
            for (int i = 0; i < lbxExcludedValues.Items.Count; i++)
            {
                float excludedValue = (float)lbxExcludedValues.Items[i];
                ContourData.ExcludedValues.Add(excludedValue);
            }

        }

        private void tvwContents_NodeMouseClick(object sender, TreeNodeMouseClickEventArgs e)
        {
            if (e.Node.Tag != null)
            {
                _SelectedHeader = e.Node.Tag as LayerDataRecordHeader;
                lblSelectedDataLayer.Text = BuildSelectedDataLayerLabel(_SelectedHeader);
            }
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }

        private string BuildSelectedDataLayerLabel(LayerDataRecordHeader header)
        {
            string label = "Selected data layer:";
            if (header != null)
            {
                label = label + "  Stress period = " + header.StressPeriod.ToString() + "  Time step = "
                  + header.TimeStep.ToString() + "  Model layer = " + header.Layer.ToString();
            }
            else
            {
                label = label + " <none>";
            }
            return label;
        }

        private void txtReferenceContour_TextChanged(object sender, EventArgs e)
        {

        }

        private void btnSelectContourColor_Click(object sender, EventArgs e)
        {
            ColorDialog dialog = new ColorDialog();
            dialog.Color = panelContourColor.BackColor;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                panelContourColor.BackColor = dialog.Color;
            }
        }

        private void btnAddExcludedValue_Click(object sender, EventArgs e)
        {
            float excludedValue = float.Parse(txtAddExcludedValue.Text);
            if (!lbxExcludedValues.Items.Contains(excludedValue))
            {
                lbxExcludedValues.Items.Add(excludedValue);
            }

        }

        private void btnRemoveExcludedValue_Click(object sender, EventArgs e)
        {
            if (lbxExcludedValues.SelectedIndex >= 0)
            {
                lbxExcludedValues.Items.RemoveAt(lbxExcludedValues.SelectedIndex);
            }
        }

    }
}
