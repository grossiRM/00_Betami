namespace USGS.Puma.UI.Modpath
{
    partial class ModpathShapefileExportDialog
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
            this.chkExportEndpoints = new System.Windows.Forms.CheckBox();
            this.chkExportPathlines = new System.Windows.Forms.CheckBox();
            this.chkExportTimeseries = new System.Windows.Forms.CheckBox();
            this.txtEndpointsName = new System.Windows.Forms.TextBox();
            this.lblEndpointsName = new System.Windows.Forms.Label();
            this.lblNameNote = new System.Windows.Forms.Label();
            this.cboEndpoint = new System.Windows.Forms.ComboBox();
            this.panelEndpoints = new System.Windows.Forms.Panel();
            this.panelPathlines = new System.Windows.Forms.Panel();
            this.txtPathlinesName = new System.Windows.Forms.TextBox();
            this.lblPathlinesName = new System.Windows.Forms.Label();
            this.panelTimeseries = new System.Windows.Forms.Panel();
            this.txtTimeseriesName = new System.Windows.Forms.TextBox();
            this.lblTimeseriesName = new System.Windows.Forms.Label();
            this.lblWorkingDirectory = new System.Windows.Forms.Label();
            this.txtExportFolder = new System.Windows.Forms.TextBox();
            this.btnExport = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.gboxStatus = new System.Windows.Forms.GroupBox();
            this.txtStatus = new System.Windows.Forms.TextBox();
            this.panelEndpoints.SuspendLayout();
            this.panelPathlines.SuspendLayout();
            this.panelTimeseries.SuspendLayout();
            this.gboxStatus.SuspendLayout();
            this.SuspendLayout();
            // 
            // chkExportEndpoints
            // 
            this.chkExportEndpoints.AutoSize = true;
            this.chkExportEndpoints.Location = new System.Drawing.Point(25, 107);
            this.chkExportEndpoints.Name = "chkExportEndpoints";
            this.chkExportEndpoints.Size = new System.Drawing.Size(145, 17);
            this.chkExportEndpoints.TabIndex = 0;
            this.chkExportEndpoints.Text = "Export endpoint shapefile";
            this.chkExportEndpoints.UseVisualStyleBackColor = true;
            // 
            // chkExportPathlines
            // 
            this.chkExportPathlines.AutoSize = true;
            this.chkExportPathlines.Location = new System.Drawing.Point(25, 229);
            this.chkExportPathlines.Name = "chkExportPathlines";
            this.chkExportPathlines.Size = new System.Drawing.Size(141, 17);
            this.chkExportPathlines.TabIndex = 1;
            this.chkExportPathlines.Text = "Export pathline shapefile";
            this.chkExportPathlines.UseVisualStyleBackColor = true;
            this.chkExportPathlines.CheckedChanged += new System.EventHandler(this.chkExportPahlines_CheckedChanged);
            // 
            // chkExportTimeseries
            // 
            this.chkExportTimeseries.AutoSize = true;
            this.chkExportTimeseries.Location = new System.Drawing.Point(25, 320);
            this.chkExportTimeseries.Name = "chkExportTimeseries";
            this.chkExportTimeseries.Size = new System.Drawing.Size(150, 17);
            this.chkExportTimeseries.TabIndex = 2;
            this.chkExportTimeseries.Text = "Export timeseries shapefile";
            this.chkExportTimeseries.UseVisualStyleBackColor = true;
            // 
            // txtEndpointsName
            // 
            this.txtEndpointsName.Location = new System.Drawing.Point(0, 25);
            this.txtEndpointsName.Name = "txtEndpointsName";
            this.txtEndpointsName.Size = new System.Drawing.Size(370, 20);
            this.txtEndpointsName.TabIndex = 3;
            // 
            // lblEndpointsName
            // 
            this.lblEndpointsName.AutoSize = true;
            this.lblEndpointsName.Location = new System.Drawing.Point(-3, 9);
            this.lblEndpointsName.Name = "lblEndpointsName";
            this.lblEndpointsName.Size = new System.Drawing.Size(38, 13);
            this.lblEndpointsName.TabIndex = 4;
            this.lblEndpointsName.Text = "Name:";
            // 
            // lblNameNote
            // 
            this.lblNameNote.AutoSize = true;
            this.lblNameNote.Location = new System.Drawing.Point(22, 73);
            this.lblNameNote.Name = "lblNameNote";
            this.lblNameNote.Size = new System.Drawing.Size(246, 13);
            this.lblNameNote.TabIndex = 5;
            this.lblNameNote.Text = "Specify the basename for each exported shapefile.";
            // 
            // cboEndpoint
            // 
            this.cboEndpoint.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboEndpoint.FormattingEnabled = true;
            this.cboEndpoint.Location = new System.Drawing.Point(0, 51);
            this.cboEndpoint.Name = "cboEndpoint";
            this.cboEndpoint.Size = new System.Drawing.Size(195, 21);
            this.cboEndpoint.TabIndex = 6;
            // 
            // panelEndpoints
            // 
            this.panelEndpoints.Controls.Add(this.cboEndpoint);
            this.panelEndpoints.Controls.Add(this.txtEndpointsName);
            this.panelEndpoints.Controls.Add(this.lblEndpointsName);
            this.panelEndpoints.Location = new System.Drawing.Point(44, 130);
            this.panelEndpoints.Name = "panelEndpoints";
            this.panelEndpoints.Size = new System.Drawing.Size(373, 73);
            this.panelEndpoints.TabIndex = 7;
            // 
            // panelPathlines
            // 
            this.panelPathlines.Controls.Add(this.txtPathlinesName);
            this.panelPathlines.Controls.Add(this.lblPathlinesName);
            this.panelPathlines.Location = new System.Drawing.Point(44, 252);
            this.panelPathlines.Name = "panelPathlines";
            this.panelPathlines.Size = new System.Drawing.Size(373, 45);
            this.panelPathlines.TabIndex = 8;
            // 
            // txtPathlinesName
            // 
            this.txtPathlinesName.Location = new System.Drawing.Point(0, 25);
            this.txtPathlinesName.Name = "txtPathlinesName";
            this.txtPathlinesName.Size = new System.Drawing.Size(370, 20);
            this.txtPathlinesName.TabIndex = 1;
            // 
            // lblPathlinesName
            // 
            this.lblPathlinesName.AutoSize = true;
            this.lblPathlinesName.Location = new System.Drawing.Point(-3, 9);
            this.lblPathlinesName.Name = "lblPathlinesName";
            this.lblPathlinesName.Size = new System.Drawing.Size(35, 13);
            this.lblPathlinesName.TabIndex = 0;
            this.lblPathlinesName.Text = "Name";
            // 
            // panelTimeseries
            // 
            this.panelTimeseries.Controls.Add(this.txtTimeseriesName);
            this.panelTimeseries.Controls.Add(this.lblTimeseriesName);
            this.panelTimeseries.Location = new System.Drawing.Point(44, 343);
            this.panelTimeseries.Name = "panelTimeseries";
            this.panelTimeseries.Size = new System.Drawing.Size(371, 45);
            this.panelTimeseries.TabIndex = 9;
            // 
            // txtTimeseriesName
            // 
            this.txtTimeseriesName.Location = new System.Drawing.Point(0, 25);
            this.txtTimeseriesName.Name = "txtTimeseriesName";
            this.txtTimeseriesName.Size = new System.Drawing.Size(368, 20);
            this.txtTimeseriesName.TabIndex = 1;
            // 
            // lblTimeseriesName
            // 
            this.lblTimeseriesName.AutoSize = true;
            this.lblTimeseriesName.Location = new System.Drawing.Point(-3, 9);
            this.lblTimeseriesName.Name = "lblTimeseriesName";
            this.lblTimeseriesName.Size = new System.Drawing.Size(35, 13);
            this.lblTimeseriesName.TabIndex = 0;
            this.lblTimeseriesName.Text = "Name";
            // 
            // lblWorkingDirectory
            // 
            this.lblWorkingDirectory.AutoSize = true;
            this.lblWorkingDirectory.Location = new System.Drawing.Point(22, 20);
            this.lblWorkingDirectory.Name = "lblWorkingDirectory";
            this.lblWorkingDirectory.Size = new System.Drawing.Size(66, 13);
            this.lblWorkingDirectory.TabIndex = 10;
            this.lblWorkingDirectory.Text = "Export folder";
            // 
            // txtExportFolder
            // 
            this.txtExportFolder.BackColor = System.Drawing.SystemColors.Window;
            this.txtExportFolder.Location = new System.Drawing.Point(24, 36);
            this.txtExportFolder.Name = "txtExportFolder";
            this.txtExportFolder.ReadOnly = true;
            this.txtExportFolder.Size = new System.Drawing.Size(390, 20);
            this.txtExportFolder.TabIndex = 11;
            // 
            // btnExport
            // 
            this.btnExport.Location = new System.Drawing.Point(265, 482);
            this.btnExport.Name = "btnExport";
            this.btnExport.Size = new System.Drawing.Size(75, 23);
            this.btnExport.TabIndex = 12;
            this.btnExport.Text = "Export";
            this.btnExport.UseVisualStyleBackColor = true;
            this.btnExport.Click += new System.EventHandler(this.btnExport_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(346, 482);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 13;
            this.btnCancel.Text = "Close";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // gboxStatus
            // 
            this.gboxStatus.Controls.Add(this.txtStatus);
            this.gboxStatus.Location = new System.Drawing.Point(36, 399);
            this.gboxStatus.Name = "gboxStatus";
            this.gboxStatus.Size = new System.Drawing.Size(385, 77);
            this.gboxStatus.TabIndex = 15;
            this.gboxStatus.TabStop = false;
            this.gboxStatus.Text = "Status";
            // 
            // txtStatus
            // 
            this.txtStatus.BackColor = System.Drawing.SystemColors.Window;
            this.txtStatus.Location = new System.Drawing.Point(8, 18);
            this.txtStatus.Multiline = true;
            this.txtStatus.Name = "txtStatus";
            this.txtStatus.ReadOnly = true;
            this.txtStatus.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.txtStatus.Size = new System.Drawing.Size(368, 53);
            this.txtStatus.TabIndex = 0;
            // 
            // ModpathShapefileExportDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(433, 517);
            this.Controls.Add(this.gboxStatus);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnExport);
            this.Controls.Add(this.txtExportFolder);
            this.Controls.Add(this.lblWorkingDirectory);
            this.Controls.Add(this.panelTimeseries);
            this.Controls.Add(this.panelPathlines);
            this.Controls.Add(this.panelEndpoints);
            this.Controls.Add(this.lblNameNote);
            this.Controls.Add(this.chkExportTimeseries);
            this.Controls.Add(this.chkExportPathlines);
            this.Controls.Add(this.chkExportEndpoints);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ModpathShapefileExportDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Export Shapefiles";
            this.panelEndpoints.ResumeLayout(false);
            this.panelEndpoints.PerformLayout();
            this.panelPathlines.ResumeLayout(false);
            this.panelPathlines.PerformLayout();
            this.panelTimeseries.ResumeLayout(false);
            this.panelTimeseries.PerformLayout();
            this.gboxStatus.ResumeLayout(false);
            this.gboxStatus.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.CheckBox chkExportEndpoints;
        private System.Windows.Forms.CheckBox chkExportPathlines;
        private System.Windows.Forms.CheckBox chkExportTimeseries;
        private System.Windows.Forms.TextBox txtEndpointsName;
        private System.Windows.Forms.Label lblEndpointsName;
        private System.Windows.Forms.Label lblNameNote;
        private System.Windows.Forms.ComboBox cboEndpoint;
        private System.Windows.Forms.Panel panelEndpoints;
        private System.Windows.Forms.Panel panelPathlines;
        private System.Windows.Forms.Panel panelTimeseries;
        private System.Windows.Forms.Label lblWorkingDirectory;
        private System.Windows.Forms.TextBox txtExportFolder;
        private System.Windows.Forms.TextBox txtPathlinesName;
        private System.Windows.Forms.Label lblPathlinesName;
        private System.Windows.Forms.Label lblTimeseriesName;
        private System.Windows.Forms.TextBox txtTimeseriesName;
        private System.Windows.Forms.Button btnExport;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.GroupBox gboxStatus;
        private System.Windows.Forms.TextBox txtStatus;
    }
}