namespace USGS.Puma.UI.Modpath
{
    partial class PathlineQueryFilter
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
            this.lblParticleID = new System.Windows.Forms.Label();
            this.cboParticleIDOption = new System.Windows.Forms.ComboBox();
            this.panelParticleID = new System.Windows.Forms.Panel();
            this.lblParticleIDValue = new System.Windows.Forms.Label();
            this.txtParticleID = new System.Windows.Forms.TextBox();
            this.gboxQueryDef.SuspendLayout();
            this.panelParticleID.SuspendLayout();
            this.SuspendLayout();
            // 
            // gboxQueryDef
            // 
            this.gboxQueryDef.Controls.Add(this.panelParticleID);
            this.gboxQueryDef.Controls.Add(this.cboParticleIDOption);
            this.gboxQueryDef.Controls.Add(this.lblParticleID);
            this.gboxQueryDef.Size = new System.Drawing.Size(350, 99);
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(228, 172);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(309, 172);
            // 
            // lblParticleID
            // 
            this.lblParticleID.AutoSize = true;
            this.lblParticleID.Location = new System.Drawing.Point(21, 24);
            this.lblParticleID.Name = "lblParticleID";
            this.lblParticleID.Size = new System.Drawing.Size(67, 13);
            this.lblParticleID.TabIndex = 0;
            this.lblParticleID.Text = "Particle Id is:";
            // 
            // cboParticleIDOption
            // 
            this.cboParticleIDOption.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboParticleIDOption.FormattingEnabled = true;
            this.cboParticleIDOption.Location = new System.Drawing.Point(24, 40);
            this.cboParticleIDOption.Name = "cboParticleIDOption";
            this.cboParticleIDOption.Size = new System.Drawing.Size(229, 21);
            this.cboParticleIDOption.TabIndex = 1;
            this.cboParticleIDOption.SelectedIndexChanged += new System.EventHandler(this.cboParticleIDOption_SelectedIndexChanged);
            // 
            // panelParticleID
            // 
            this.panelParticleID.Controls.Add(this.lblParticleIDValue);
            this.panelParticleID.Controls.Add(this.txtParticleID);
            this.panelParticleID.Location = new System.Drawing.Point(259, 20);
            this.panelParticleID.Name = "panelParticleID";
            this.panelParticleID.Size = new System.Drawing.Size(87, 41);
            this.panelParticleID.TabIndex = 2;
            // 
            // lblParticleIDValue
            // 
            this.lblParticleIDValue.AutoSize = true;
            this.lblParticleIDValue.Location = new System.Drawing.Point(0, 4);
            this.lblParticleIDValue.Name = "lblParticleIDValue";
            this.lblParticleIDValue.Size = new System.Drawing.Size(16, 13);
            this.lblParticleIDValue.TabIndex = 1;
            this.lblParticleIDValue.Text = "Id";
            // 
            // txtParticleID
            // 
            this.txtParticleID.Location = new System.Drawing.Point(0, 20);
            this.txtParticleID.Name = "txtParticleID";
            this.txtParticleID.Size = new System.Drawing.Size(84, 20);
            this.txtParticleID.TabIndex = 0;
            // 
            // PathlineQueryFilter
            // 
            this.ClientSize = new System.Drawing.Size(396, 207);
            this.Name = "PathlineQueryFilter";
            this.Text = "Pathline Query Filter";
            this.gboxQueryDef.ResumeLayout(false);
            this.gboxQueryDef.PerformLayout();
            this.panelParticleID.ResumeLayout(false);
            this.panelParticleID.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ComboBox cboParticleIDOption;
        private System.Windows.Forms.Label lblParticleID;
        private System.Windows.Forms.Panel panelParticleID;
        private System.Windows.Forms.Label lblParticleIDValue;
        private System.Windows.Forms.TextBox txtParticleID;
    }
}
