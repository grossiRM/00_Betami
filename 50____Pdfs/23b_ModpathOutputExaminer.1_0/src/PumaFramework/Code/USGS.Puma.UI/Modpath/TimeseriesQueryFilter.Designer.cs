namespace USGS.Puma.UI.Modpath
{
    partial class TimeseriesQueryFilter
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
            this.panelParticleID = new System.Windows.Forms.Panel();
            this.lblParticleIDValue = new System.Windows.Forms.Label();
            this.txtParticleID = new System.Windows.Forms.TextBox();
            this.cboParticleIDOption = new System.Windows.Forms.ComboBox();
            this.lblParticleID = new System.Windows.Forms.Label();
            this.panelTimePoint = new System.Windows.Forms.Panel();
            this.lblTimePointValue = new System.Windows.Forms.Label();
            this.txtTimePoint = new System.Windows.Forms.TextBox();
            this.cboTimePointOption = new System.Windows.Forms.ComboBox();
            this.lblTimePoint = new System.Windows.Forms.Label();
            this.gboxQueryDef.SuspendLayout();
            this.panelParticleID.SuspendLayout();
            this.panelTimePoint.SuspendLayout();
            this.SuspendLayout();
            // 
            // radioBtnFilteringIsOff
            // 
            this.radioBtnFilteringIsOff.Size = new System.Drawing.Size(152, 17);
            this.radioBtnFilteringIsOff.Text = "Show all timeseries records";
            // 
            // radioBtnFilteringIsOn
            // 
            this.radioBtnFilteringIsOn.Size = new System.Drawing.Size(336, 17);
            this.radioBtnFilteringIsOn.Text = "Show only those timeseries records that meet the following criteria:";
            // 
            // gboxQueryDef
            // 
            this.gboxQueryDef.Controls.Add(this.lblTimePoint);
            this.gboxQueryDef.Controls.Add(this.cboTimePointOption);
            this.gboxQueryDef.Controls.Add(this.panelTimePoint);
            this.gboxQueryDef.Controls.Add(this.lblParticleID);
            this.gboxQueryDef.Controls.Add(this.cboParticleIDOption);
            this.gboxQueryDef.Controls.Add(this.panelParticleID);
            this.gboxQueryDef.Size = new System.Drawing.Size(318, 177);
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(196, 241);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(277, 241);
            // 
            // panelParticleID
            // 
            this.panelParticleID.Controls.Add(this.lblParticleIDValue);
            this.panelParticleID.Controls.Add(this.txtParticleID);
            this.panelParticleID.Location = new System.Drawing.Point(229, 97);
            this.panelParticleID.Name = "panelParticleID";
            this.panelParticleID.Size = new System.Drawing.Size(85, 39);
            this.panelParticleID.TabIndex = 0;
            // 
            // lblParticleIDValue
            // 
            this.lblParticleIDValue.AutoSize = true;
            this.lblParticleIDValue.Location = new System.Drawing.Point(0, 3);
            this.lblParticleIDValue.Name = "lblParticleIDValue";
            this.lblParticleIDValue.Size = new System.Drawing.Size(16, 13);
            this.lblParticleIDValue.TabIndex = 1;
            this.lblParticleIDValue.Text = "Id";
            // 
            // txtParticleID
            // 
            this.txtParticleID.Location = new System.Drawing.Point(0, 19);
            this.txtParticleID.Name = "txtParticleID";
            this.txtParticleID.Size = new System.Drawing.Size(84, 20);
            this.txtParticleID.TabIndex = 0;
            // 
            // cboParticleIDOption
            // 
            this.cboParticleIDOption.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboParticleIDOption.FormattingEnabled = true;
            this.cboParticleIDOption.Location = new System.Drawing.Point(21, 115);
            this.cboParticleIDOption.Name = "cboParticleIDOption";
            this.cboParticleIDOption.Size = new System.Drawing.Size(199, 21);
            this.cboParticleIDOption.TabIndex = 1;
            this.cboParticleIDOption.SelectedIndexChanged += new System.EventHandler(this.cboParticleIDOption_SelectedIndexChanged);
            // 
            // lblParticleID
            // 
            this.lblParticleID.AutoSize = true;
            this.lblParticleID.Location = new System.Drawing.Point(18, 99);
            this.lblParticleID.Name = "lblParticleID";
            this.lblParticleID.Size = new System.Drawing.Size(67, 13);
            this.lblParticleID.TabIndex = 2;
            this.lblParticleID.Text = "Particle Id is:";
            // 
            // panelTimePoint
            // 
            this.panelTimePoint.Controls.Add(this.lblTimePointValue);
            this.panelTimePoint.Controls.Add(this.txtTimePoint);
            this.panelTimePoint.Location = new System.Drawing.Point(229, 42);
            this.panelTimePoint.Name = "panelTimePoint";
            this.panelTimePoint.Size = new System.Drawing.Size(85, 39);
            this.panelTimePoint.TabIndex = 3;
            // 
            // lblTimePointValue
            // 
            this.lblTimePointValue.AutoSize = true;
            this.lblTimePointValue.Location = new System.Drawing.Point(0, 3);
            this.lblTimePointValue.Name = "lblTimePointValue";
            this.lblTimePointValue.Size = new System.Drawing.Size(56, 13);
            this.lblTimePointValue.TabIndex = 1;
            this.lblTimePointValue.Text = "Time point";
            // 
            // txtTimePoint
            // 
            this.txtTimePoint.Location = new System.Drawing.Point(0, 19);
            this.txtTimePoint.Name = "txtTimePoint";
            this.txtTimePoint.Size = new System.Drawing.Size(84, 20);
            this.txtTimePoint.TabIndex = 0;
            // 
            // cboTimePointOption
            // 
            this.cboTimePointOption.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboTimePointOption.FormattingEnabled = true;
            this.cboTimePointOption.Location = new System.Drawing.Point(21, 61);
            this.cboTimePointOption.Name = "cboTimePointOption";
            this.cboTimePointOption.Size = new System.Drawing.Size(199, 21);
            this.cboTimePointOption.TabIndex = 4;
            this.cboTimePointOption.SelectedIndexChanged += new System.EventHandler(this.cboTimePointOption_SelectedIndexChanged);
            // 
            // lblTimePoint
            // 
            this.lblTimePoint.AutoSize = true;
            this.lblTimePoint.Location = new System.Drawing.Point(18, 45);
            this.lblTimePoint.Name = "lblTimePoint";
            this.lblTimePoint.Size = new System.Drawing.Size(69, 13);
            this.lblTimePoint.TabIndex = 5;
            this.lblTimePoint.Text = "Time point is:";
            // 
            // TimeseriesQueryFilter
            // 
            this.ClientSize = new System.Drawing.Size(364, 271);
            this.Name = "TimeseriesQueryFilter";
            this.Text = "Timeseries Query Filter";
            this.gboxQueryDef.ResumeLayout(false);
            this.gboxQueryDef.PerformLayout();
            this.panelParticleID.ResumeLayout(false);
            this.panelParticleID.PerformLayout();
            this.panelTimePoint.ResumeLayout(false);
            this.panelTimePoint.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Panel panelParticleID;
        private System.Windows.Forms.Label lblParticleIDValue;
        private System.Windows.Forms.TextBox txtParticleID;
        private System.Windows.Forms.Label lblParticleID;
        private System.Windows.Forms.ComboBox cboParticleIDOption;
        private System.Windows.Forms.Panel panelTimePoint;
        private System.Windows.Forms.TextBox txtTimePoint;
        private System.Windows.Forms.ComboBox cboTimePointOption;
        private System.Windows.Forms.Label lblTimePointValue;
        private System.Windows.Forms.Label lblTimePoint;
    }
}
