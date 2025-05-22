using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.IO;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using GeoAPI.Geometries;
using USGS.Puma.IO;
using USGS.Puma.NTS.Features;
using USGS.Puma.NTS.Geometries;

namespace ModpathOutputExaminer
{
    public partial class ImportShapefilesDialog : Form
    {
        private FeatureCollection features = null;

        public ImportShapefilesDialog()
        {
            InitializeComponent();
            chkLimitDisplay.Checked = true;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.OK;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            string filename = BrowseFile();
            filename = filename.Trim();
            if (!string.IsNullOrEmpty(filename))
            {
                OpenFile(filename);
            }
        }

        private string BrowseFile()
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Shapefiles (*.shp)|*.shp|All files (*.*)|*.*";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                this.Refresh();
                return dialog.FileName;
            }
            return "";
        }

        private void OpenFile(string filename)
        {
            try
            {
                features = null;
                features = EsriShapefileIO.Import(filename);
                if (features != null)
                {
                    txtShapefile.Text = filename;
                    rtxSummary.Text = GenerateSummaryText(chkLimitDisplay.Checked);
                }
                else
                {
                    throw new Exception("Error importing shapefile:" + Environment.NewLine + filename);
                }
            }
            catch (Exception e)
            {
                rtxSummary.Text = e.Message + Environment.NewLine + e.InnerException.Message;
                features = null;
            }
        }

        private string GenerateSummaryText(bool limitDisplay)
        {
            if (features == null)
                return "";

            StringBuilder sb = new StringBuilder();
            sb.Append(features.Count);
            sb.Append(" features were imported.").AppendLine().AppendLine();
            for (int n = 0; n < features.Count; n++)
            {
                IGeometry g = features[n].Geometry;

                sb.Append(g.GeometryType.ToString()).Append(" ").Append(n);
                sb.Append("  (").Append((g as Geometry).VerticalEnvelopeInternal.MinZ).Append(" ,  ").Append((g as Geometry).VerticalEnvelopeInternal.MaxZ).Append(")").AppendLine();
                ICoordinate[] coords = g.Coordinates;
                for (int i = 0; i < coords.Length; i++)
                {
                    sb.AppendLine(coords[i].ToString());
                }
                sb.AppendLine();
            }
            return sb.ToString(0, sb.Length);
        }

        private void chkLimitDisplay_CheckedChanged(object sender, EventArgs e)
        {
            rtxSummary.Text = GenerateSummaryText(chkLimitDisplay.Checked);
        }
    }
}
