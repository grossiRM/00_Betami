using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace ModpathOutputExaminer
{
    public partial class AboutBoxModpathOutputExaminer : Form
    {
        public AboutBoxModpathOutputExaminer()
        {
            InitializeComponent();
            //lblVersion.Text = "Version " + Application.ProductVersion;
            lblVersion.Text = "Version 1.0.00";
            string exeFile = Application.ExecutablePath;
            //DateTime date = System.IO.File.GetLastWriteTime(exeFile);
            //lblDate.Text = date.ToLongDateString();
            lblDate.Text = "July 16, 2012";
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Hide();
        }

        private void AboutBoxModflowOutputViewer_Load(object sender, EventArgs e)
        {

        }

        private void lblVersion_Click(object sender, EventArgs e)
        {

        }
    }
}
