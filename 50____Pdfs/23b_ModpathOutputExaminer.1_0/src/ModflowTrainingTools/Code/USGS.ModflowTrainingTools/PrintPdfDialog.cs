using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace USGS.ModflowTrainingTools
{
    public partial class PrintPdfDialog : Form
    {
        public PrintPdfDialog()
        {
            InitializeComponent();

            txtFilename.Text = "MapOutput.pdf";
        }

        public string Filename
        {
            get { return txtFilename.Text.Trim(); }
            set 
            { 
                txtFilename.Text = value; 
            }
        }

        public string Title
        {
            get { return txtTitle.Text.Trim(); }
            set { txtTitle.Text = value; }
        }

        private string _Description;
        public string Description
        {
            get { return txtDescription.Text.Trim(); }
            set { txtDescription.Text = value; }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.OK;
            this.Hide();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
            this.Hide();
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.Title = "Save Map PDF Output";
            dialog.FileName = txtFilename.Text;
            dialog.Filter = "PDF files (*.pdf)|*.pdf|All files (*.*)|*.*)";
            dialog.AddExtension = true;
            dialog.DefaultExt = ".pdf";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                txtFilename.Text = dialog.FileName;
            }
        }

    }
}
