namespace USGS.ModflowTrainingTools
{
    partial class SaveNewBinaryOutputDialog
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.textboxFilename = new System.Windows.Forms.TextBox();
            this.buttonBrowse = new System.Windows.Forms.Button();
            this.labelFilename = new System.Windows.Forms.Label();
            this.radioButtonPrecisionSingle = new System.Windows.Forms.RadioButton();
            this.radioButtonPrecisionDouble = new System.Windows.Forms.RadioButton();
            this.buttonSave = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // textboxFilename
            // 
            this.textboxFilename.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.textboxFilename.Location = new System.Drawing.Point(5, 29);
            this.textboxFilename.Name = "textboxFilename";
            this.textboxFilename.Size = new System.Drawing.Size(421, 20);
            this.textboxFilename.TabIndex = 0;
            // 
            // buttonBrowse
            // 
            this.buttonBrowse.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonBrowse.Location = new System.Drawing.Point(432, 27);
            this.buttonBrowse.Name = "buttonBrowse";
            this.buttonBrowse.Size = new System.Drawing.Size(75, 23);
            this.buttonBrowse.TabIndex = 1;
            this.buttonBrowse.Text = "Browse";
            this.buttonBrowse.UseVisualStyleBackColor = true;
            this.buttonBrowse.Click += new System.EventHandler(this.buttonBrowse_Click);
            // 
            // labelFilename
            // 
            this.labelFilename.AutoSize = true;
            this.labelFilename.Location = new System.Drawing.Point(2, 13);
            this.labelFilename.Name = "labelFilename";
            this.labelFilename.Size = new System.Drawing.Size(52, 13);
            this.labelFilename.TabIndex = 2;
            this.labelFilename.Text = "Filename:";
            // 
            // radioButtonPrecisionSingle
            // 
            this.radioButtonPrecisionSingle.AutoSize = true;
            this.radioButtonPrecisionSingle.Location = new System.Drawing.Point(5, 55);
            this.radioButtonPrecisionSingle.Name = "radioButtonPrecisionSingle";
            this.radioButtonPrecisionSingle.Size = new System.Drawing.Size(139, 17);
            this.radioButtonPrecisionSingle.TabIndex = 3;
            this.radioButtonPrecisionSingle.TabStop = true;
            this.radioButtonPrecisionSingle.Text = "Save as single precision";
            this.radioButtonPrecisionSingle.UseVisualStyleBackColor = true;
            // 
            // radioButtonPrecisionDouble
            // 
            this.radioButtonPrecisionDouble.AutoSize = true;
            this.radioButtonPrecisionDouble.Location = new System.Drawing.Point(5, 78);
            this.radioButtonPrecisionDouble.Name = "radioButtonPrecisionDouble";
            this.radioButtonPrecisionDouble.Size = new System.Drawing.Size(144, 17);
            this.radioButtonPrecisionDouble.TabIndex = 4;
            this.radioButtonPrecisionDouble.TabStop = true;
            this.radioButtonPrecisionDouble.Text = "Save as double precision";
            this.radioButtonPrecisionDouble.UseVisualStyleBackColor = true;
            // 
            // buttonSave
            // 
            this.buttonSave.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonSave.Location = new System.Drawing.Point(351, 108);
            this.buttonSave.Name = "buttonSave";
            this.buttonSave.Size = new System.Drawing.Size(75, 23);
            this.buttonSave.TabIndex = 5;
            this.buttonSave.Text = "Save";
            this.buttonSave.UseVisualStyleBackColor = true;
            this.buttonSave.Click += new System.EventHandler(this.buttonSave_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonCancel.Location = new System.Drawing.Point(432, 108);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(75, 23);
            this.buttonCancel.TabIndex = 6;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // SaveNewBinaryOutputDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(511, 139);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonSave);
            this.Controls.Add(this.radioButtonPrecisionDouble);
            this.Controls.Add(this.radioButtonPrecisionSingle);
            this.Controls.Add(this.labelFilename);
            this.Controls.Add(this.buttonBrowse);
            this.Controls.Add(this.textboxFilename);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "SaveNewBinaryOutputDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Save New Binary Output";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox textboxFilename;
        private System.Windows.Forms.Button buttonBrowse;
        private System.Windows.Forms.Label labelFilename;
        private System.Windows.Forms.RadioButton radioButtonPrecisionSingle;
        private System.Windows.Forms.RadioButton radioButtonPrecisionDouble;
        private System.Windows.Forms.Button buttonSave;
        private System.Windows.Forms.Button buttonCancel;
    }
}