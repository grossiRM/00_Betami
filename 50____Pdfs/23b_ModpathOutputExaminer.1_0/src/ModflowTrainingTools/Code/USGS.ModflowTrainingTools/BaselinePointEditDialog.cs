using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace USGS.ModflowTrainingTools
{
    public partial class BaselinePointEditDialog : Form
    {
        public BaselinePointEditDialog()
        {
            InitializeComponent();
        }

        private double _X;
        /// <summary>
        /// 
        /// </summary>
        public double X
        {
            get { return _X; }
            set { _X = value; }
        }

        private double _Y;
        /// <summary>
        /// 
        /// </summary>
        public double Y
        {
            get { return _Y; }
            set { _Y = value; }
        }

        private double _OriginX;
        /// <summary>
        /// 
        /// </summary>
        public double OriginX
        {
            get { return _OriginX; }
            set { _OriginX = value; }
        }


        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            double result;
            if (!double.TryParse(txtX.Text, out result))
            {
                MessageBox.Show("Enter a numberic value for x and y.");
            }
            else if (!double.TryParse(txtY.Text, out result))
            {
                MessageBox.Show("Enter a numberic value for x and y.");
            }
            else
            {
                X = double.Parse(txtX.Text);
                Y = double.Parse(txtY.Text);
                if (X < OriginX)
                {
                    MessageBox.Show("The x value must be greater than or equal to " + OriginX.ToString());
                }
                else
                {
                    this.DialogResult = DialogResult.OK;
                }
            }

        }
    }
}
