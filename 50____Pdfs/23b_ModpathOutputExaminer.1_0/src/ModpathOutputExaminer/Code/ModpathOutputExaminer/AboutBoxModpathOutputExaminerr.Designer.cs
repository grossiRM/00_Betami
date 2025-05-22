namespace ModpathOutputExaminer
{
    partial class AboutBoxModpathOutputExaminer
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
            this.lblMainHeader = new System.Windows.Forms.Label();
            this.lblAppNameHeader = new System.Windows.Forms.Label();
            this.panelMainHeader = new System.Windows.Forms.Panel();
            this.lblVersion = new System.Windows.Forms.Label();
            this.lblDate = new System.Windows.Forms.Label();
            this.btnClose = new System.Windows.Forms.Button();
            this.picboxUsgsVisualID = new System.Windows.Forms.PictureBox();
            this.panelMainHeader.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picboxUsgsVisualID)).BeginInit();
            this.SuspendLayout();
            // 
            // lblMainHeader
            // 
            this.lblMainHeader.AutoSize = true;
            this.lblMainHeader.Font = new System.Drawing.Font("Palatino Linotype", 24F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblMainHeader.ForeColor = System.Drawing.Color.Navy;
            this.lblMainHeader.Location = new System.Drawing.Point(49, 9);
            this.lblMainHeader.Name = "lblMainHeader";
            this.lblMainHeader.Size = new System.Drawing.Size(416, 43);
            this.lblMainHeader.TabIndex = 0;
            this.lblMainHeader.Text = "MODFLOW Training Tools";
            // 
            // lblAppNameHeader
            // 
            this.lblAppNameHeader.AutoSize = true;
            this.lblAppNameHeader.Font = new System.Drawing.Font("Palatino Linotype", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblAppNameHeader.Location = new System.Drawing.Point(100, 103);
            this.lblAppNameHeader.Name = "lblAppNameHeader";
            this.lblAppNameHeader.Size = new System.Drawing.Size(280, 28);
            this.lblAppNameHeader.TabIndex = 1;
            this.lblAppNameHeader.Text = "Modpath Output Examiner";
            // 
            // panelMainHeader
            // 
            this.panelMainHeader.BackColor = System.Drawing.Color.White;
            this.panelMainHeader.Controls.Add(this.lblMainHeader);
            this.panelMainHeader.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelMainHeader.Location = new System.Drawing.Point(0, 0);
            this.panelMainHeader.Name = "panelMainHeader";
            this.panelMainHeader.Size = new System.Drawing.Size(515, 68);
            this.panelMainHeader.TabIndex = 2;
            // 
            // lblVersion
            // 
            this.lblVersion.AutoSize = true;
            this.lblVersion.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblVersion.Location = new System.Drawing.Point(12, 281);
            this.lblVersion.Name = "lblVersion";
            this.lblVersion.Size = new System.Drawing.Size(69, 13);
            this.lblVersion.TabIndex = 3;
            this.lblVersion.Text = "Version 1.0.1";
            this.lblVersion.Click += new System.EventHandler(this.lblVersion_Click);
            // 
            // lblDate
            // 
            this.lblDate.AutoSize = true;
            this.lblDate.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDate.Location = new System.Drawing.Point(12, 303);
            this.lblDate.Name = "lblDate";
            this.lblDate.Size = new System.Drawing.Size(70, 13);
            this.lblDate.TabIndex = 4;
            this.lblDate.Text = "July 16, 2012";
            // 
            // btnClose
            // 
            this.btnClose.Location = new System.Drawing.Point(428, 303);
            this.btnClose.Name = "btnClose";
            this.btnClose.Size = new System.Drawing.Size(75, 23);
            this.btnClose.TabIndex = 5;
            this.btnClose.Text = "Close";
            this.btnClose.UseVisualStyleBackColor = true;
            this.btnClose.Click += new System.EventHandler(this.btnClose_Click);
            // 
            // picboxUsgsVisualID
            // 
            this.picboxUsgsVisualID.Image = global::ModpathOutputExaminer.Properties.Resources.USGS_visual_identity;
            this.picboxUsgsVisualID.Location = new System.Drawing.Point(167, 159);
            this.picboxUsgsVisualID.Name = "picboxUsgsVisualID";
            this.picboxUsgsVisualID.Size = new System.Drawing.Size(178, 72);
            this.picboxUsgsVisualID.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.picboxUsgsVisualID.TabIndex = 6;
            this.picboxUsgsVisualID.TabStop = false;
            // 
            // AboutBoxModpathOutputExaminer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(515, 332);
            this.Controls.Add(this.picboxUsgsVisualID);
            this.Controls.Add(this.btnClose);
            this.Controls.Add(this.lblDate);
            this.Controls.Add(this.lblVersion);
            this.Controls.Add(this.panelMainHeader);
            this.Controls.Add(this.lblAppNameHeader);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "AboutBoxModpathOutputExaminer";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Load += new System.EventHandler(this.AboutBoxModflowOutputViewer_Load);
            this.panelMainHeader.ResumeLayout(false);
            this.panelMainHeader.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picboxUsgsVisualID)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblMainHeader;
        private System.Windows.Forms.Label lblAppNameHeader;
        private System.Windows.Forms.Panel panelMainHeader;
        private System.Windows.Forms.Label lblVersion;
        private System.Windows.Forms.Label lblDate;
        private System.Windows.Forms.Button btnClose;
        private System.Windows.Forms.PictureBox picboxUsgsVisualID;
    }
}