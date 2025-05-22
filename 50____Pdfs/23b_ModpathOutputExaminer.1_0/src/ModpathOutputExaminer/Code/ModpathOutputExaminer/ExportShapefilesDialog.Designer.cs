namespace ModpathOutputExaminer
{
    partial class ExportShapefilesDialog
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
            this.btnExport = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.chkGridOutline = new System.Windows.Forms.CheckBox();
            this.chkGridlines = new System.Windows.Forms.CheckBox();
            this.chkContours = new System.Windows.Forms.CheckBox();
            this.chkParticleOutput = new System.Windows.Forms.CheckBox();
            this.gboxFeatureData = new System.Windows.Forms.GroupBox();
            this.lblStatus = new System.Windows.Forms.Label();
            this.textboxStatus = new System.Windows.Forms.TextBox();
            this.txtParticleOutput = new System.Windows.Forms.TextBox();
            this.txtContourLines = new System.Windows.Forms.TextBox();
            this.lblLocalName = new System.Windows.Forms.Label();
            this.txtModelGridlines = new System.Windows.Forms.TextBox();
            this.txtModelGridOutline = new System.Windows.Forms.TextBox();
            this.lblExportDirectory = new System.Windows.Forms.Label();
            this.txtExportDirectory = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.gboxFeatureData.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnExport
            // 
            this.btnExport.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnExport.Location = new System.Drawing.Point(271, 362);
            this.btnExport.Name = "btnExport";
            this.btnExport.Size = new System.Drawing.Size(75, 23);
            this.btnExport.TabIndex = 0;
            this.btnExport.Text = "Export";
            this.btnExport.UseVisualStyleBackColor = true;
            this.btnExport.Click += new System.EventHandler(this.btnExport_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(352, 362);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 1;
            this.btnCancel.Text = "Close";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // chkGridOutline
            // 
            this.chkGridOutline.AutoSize = true;
            this.chkGridOutline.Location = new System.Drawing.Point(23, 40);
            this.chkGridOutline.Name = "chkGridOutline";
            this.chkGridOutline.Size = new System.Drawing.Size(109, 17);
            this.chkGridOutline.TabIndex = 2;
            this.chkGridOutline.Text = "Model grid outline";
            this.chkGridOutline.UseVisualStyleBackColor = true;
            // 
            // chkGridlines
            // 
            this.chkGridlines.AutoSize = true;
            this.chkGridlines.Location = new System.Drawing.Point(23, 80);
            this.chkGridlines.Name = "chkGridlines";
            this.chkGridlines.Size = new System.Drawing.Size(99, 17);
            this.chkGridlines.TabIndex = 3;
            this.chkGridlines.Text = "Model grid lines";
            this.chkGridlines.UseVisualStyleBackColor = true;
            // 
            // chkContours
            // 
            this.chkContours.AutoSize = true;
            this.chkContours.Location = new System.Drawing.Point(23, 120);
            this.chkContours.Name = "chkContours";
            this.chkContours.Size = new System.Drawing.Size(87, 17);
            this.chkContours.TabIndex = 4;
            this.chkContours.Text = "Contour lines";
            this.chkContours.UseVisualStyleBackColor = true;
            // 
            // chkParticleOutput
            // 
            this.chkParticleOutput.AutoSize = true;
            this.chkParticleOutput.Location = new System.Drawing.Point(23, 160);
            this.chkParticleOutput.Name = "chkParticleOutput";
            this.chkParticleOutput.Size = new System.Drawing.Size(94, 17);
            this.chkParticleOutput.TabIndex = 5;
            this.chkParticleOutput.Text = "Particle output";
            this.chkParticleOutput.UseVisualStyleBackColor = true;
            // 
            // gboxFeatureData
            // 
            this.gboxFeatureData.Controls.Add(this.lblStatus);
            this.gboxFeatureData.Controls.Add(this.textboxStatus);
            this.gboxFeatureData.Controls.Add(this.txtParticleOutput);
            this.gboxFeatureData.Controls.Add(this.txtContourLines);
            this.gboxFeatureData.Controls.Add(this.lblLocalName);
            this.gboxFeatureData.Controls.Add(this.txtModelGridlines);
            this.gboxFeatureData.Controls.Add(this.txtModelGridOutline);
            this.gboxFeatureData.Controls.Add(this.chkGridOutline);
            this.gboxFeatureData.Controls.Add(this.chkGridlines);
            this.gboxFeatureData.Controls.Add(this.chkParticleOutput);
            this.gboxFeatureData.Controls.Add(this.chkContours);
            this.gboxFeatureData.Location = new System.Drawing.Point(12, 64);
            this.gboxFeatureData.Name = "gboxFeatureData";
            this.gboxFeatureData.Size = new System.Drawing.Size(415, 293);
            this.gboxFeatureData.TabIndex = 7;
            this.gboxFeatureData.TabStop = false;
            this.gboxFeatureData.Text = "Feature data";
            // 
            // lblStatus
            // 
            this.lblStatus.AutoSize = true;
            this.lblStatus.Location = new System.Drawing.Point(20, 195);
            this.lblStatus.Name = "lblStatus";
            this.lblStatus.Size = new System.Drawing.Size(40, 13);
            this.lblStatus.TabIndex = 11;
            this.lblStatus.Text = "Status:";
            // 
            // textboxStatus
            // 
            this.textboxStatus.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.textboxStatus.Location = new System.Drawing.Point(23, 211);
            this.textboxStatus.Multiline = true;
            this.textboxStatus.Name = "textboxStatus";
            this.textboxStatus.ReadOnly = true;
            this.textboxStatus.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textboxStatus.Size = new System.Drawing.Size(385, 74);
            this.textboxStatus.TabIndex = 12;
            // 
            // txtParticleOutput
            // 
            this.txtParticleOutput.Location = new System.Drawing.Point(156, 158);
            this.txtParticleOutput.Name = "txtParticleOutput";
            this.txtParticleOutput.Size = new System.Drawing.Size(252, 20);
            this.txtParticleOutput.TabIndex = 11;
            // 
            // txtContourLines
            // 
            this.txtContourLines.Location = new System.Drawing.Point(156, 117);
            this.txtContourLines.Name = "txtContourLines";
            this.txtContourLines.Size = new System.Drawing.Size(252, 20);
            this.txtContourLines.TabIndex = 10;
            // 
            // lblLocalName
            // 
            this.lblLocalName.AutoSize = true;
            this.lblLocalName.Location = new System.Drawing.Point(153, 16);
            this.lblLocalName.Name = "lblLocalName";
            this.lblLocalName.Size = new System.Drawing.Size(195, 13);
            this.lblLocalName.TabIndex = 9;
            this.lblLocalName.Text = "Local shapefile name  (no file extension)";
            // 
            // txtModelGridlines
            // 
            this.txtModelGridlines.Location = new System.Drawing.Point(156, 77);
            this.txtModelGridlines.Name = "txtModelGridlines";
            this.txtModelGridlines.Size = new System.Drawing.Size(252, 20);
            this.txtModelGridlines.TabIndex = 8;
            // 
            // txtModelGridOutline
            // 
            this.txtModelGridOutline.Location = new System.Drawing.Point(156, 37);
            this.txtModelGridOutline.Name = "txtModelGridOutline";
            this.txtModelGridOutline.Size = new System.Drawing.Size(253, 20);
            this.txtModelGridOutline.TabIndex = 7;
            // 
            // lblExportDirectory
            // 
            this.lblExportDirectory.AutoSize = true;
            this.lblExportDirectory.Location = new System.Drawing.Point(9, 9);
            this.lblExportDirectory.Name = "lblExportDirectory";
            this.lblExportDirectory.Size = new System.Drawing.Size(83, 13);
            this.lblExportDirectory.TabIndex = 8;
            this.lblExportDirectory.Text = "Export directory:";
            // 
            // txtExportDirectory
            // 
            this.txtExportDirectory.Location = new System.Drawing.Point(12, 25);
            this.txtExportDirectory.Name = "txtExportDirectory";
            this.txtExportDirectory.ReadOnly = true;
            this.txtExportDirectory.Size = new System.Drawing.Size(334, 20);
            this.txtExportDirectory.TabIndex = 9;
            // 
            // btnBrowse
            // 
            this.btnBrowse.Location = new System.Drawing.Point(352, 23);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(75, 23);
            this.btnBrowse.TabIndex = 10;
            this.btnBrowse.Text = "Browse";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // ExportShapefilesDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(439, 399);
            this.Controls.Add(this.btnBrowse);
            this.Controls.Add(this.txtExportDirectory);
            this.Controls.Add(this.lblExportDirectory);
            this.Controls.Add(this.gboxFeatureData);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnExport);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ExportShapefilesDialog";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Export Shapefiles";
            this.gboxFeatureData.ResumeLayout(false);
            this.gboxFeatureData.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnExport;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.CheckBox chkGridOutline;
        private System.Windows.Forms.CheckBox chkGridlines;
        private System.Windows.Forms.CheckBox chkContours;
        private System.Windows.Forms.CheckBox chkParticleOutput;
        private System.Windows.Forms.GroupBox gboxFeatureData;
        private System.Windows.Forms.Label lblLocalName;
        private System.Windows.Forms.TextBox txtModelGridlines;
        private System.Windows.Forms.TextBox txtModelGridOutline;
        private System.Windows.Forms.TextBox txtParticleOutput;
        private System.Windows.Forms.TextBox txtContourLines;
        private System.Windows.Forms.Label lblExportDirectory;
        private System.Windows.Forms.TextBox txtExportDirectory;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.Label lblStatus;
        private System.Windows.Forms.TextBox textboxStatus;
    }
}