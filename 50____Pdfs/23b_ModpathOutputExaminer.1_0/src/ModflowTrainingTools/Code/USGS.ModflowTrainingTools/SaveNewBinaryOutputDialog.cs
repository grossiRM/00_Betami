using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.IO;
using System.Windows.Forms;
using USGS.Puma.IO;
using USGS.Puma.Modflow;

namespace USGS.ModflowTrainingTools
{
    public partial class SaveNewBinaryOutputDialog : Form
    {
        private string _DefaultName = "";

        public SaveNewBinaryOutputDialog()
        {
            InitializeComponent();
        }

        #region Event Handlers
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonBrowse_Click(object sender, EventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.Filter = "Head and Drawdown files (*.hed; *.ddn)|*.hed;*.ddn|All Files (*.*)|*.*";
            dialog.InitialDirectory = Path.GetDirectoryName(textboxFilename.Text);
            dialog.FileName = textboxFilename.Text;
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                textboxFilename.Text = dialog.FileName;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonCancel_Click(object sender, EventArgs e)
        {
            this.Filename = "";
            this.DialogResult = DialogResult.Cancel;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonSave_Click(object sender, EventArgs e)
        {
            try
            {
                string filename = textboxFilename.Text;

                if (File.Exists(filename))
                {
                    DialogResult result = MessageBox.Show("The file already exists. Do you want to overwrite it?", "File Exists", MessageBoxButtons.YesNo);
                    if (result == DialogResult.No)
                    {
                        return; 
                    }
                    File.Delete(filename);
                }
                
                if (radioButtonPrecisionSingle.Checked)
                {
                    BinaryLayerWriter.WriteFile(this.FileReader, filename, OutputPrecisionType.Single);
                }
                else
                {
                    BinaryLayerWriter.WriteFile(this.FileReader, filename, OutputPrecisionType.Double);
                }

                this.Filename = filename;
                this.DialogResult = DialogResult.OK;
            }
            catch (Exception error)
            {
                MessageBox.Show(error.Message);
                return;
            }
        }
        #endregion

        #region Public Properties and Methods
        private string _Filename;
        /// <summary>
        /// 
        /// </summary>
        public string Filename
        {
            get { return _Filename; }
            set { _Filename = value; }
        }

        private BinaryLayerReader _FileReader;
        /// <summary>
        /// 
        /// </summary>
        public BinaryLayerReader FileReader
        {
            get { return _FileReader; }
            set { _FileReader = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        public void InitializeDialog(BinaryLayerReader reader)
        {
            FileReader = reader;
            InitializeDialog();
        }
        /// <summary>
        /// 
        /// </summary>
        public void InitializeDialog()
        {
            if (this.FileReader == null)
            {
                throw new Exception("FileReader has not been initialized.");
            }

            string directory = Path.GetDirectoryName(this.FileReader.Filename);
            string localname = Path.GetFileName(this.FileReader.Filename);
            localname = "copy_of_" + localname;
            _DefaultName = Path.Combine(directory, localname);
            textboxFilename.Text = _DefaultName;
            if (this.FileReader.OutputPrecision == OutputPrecisionType.Double)
            {
                radioButtonPrecisionDouble.Checked = true;
            }
            else
            {
                radioButtonPrecisionSingle.Checked = true;
            }
        }
        #endregion


    }

}
