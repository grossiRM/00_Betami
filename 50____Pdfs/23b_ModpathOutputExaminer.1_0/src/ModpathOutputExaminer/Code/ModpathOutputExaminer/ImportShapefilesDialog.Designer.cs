namespace ModpathOutputExaminer
{
    partial class ImportShapefilesDialog
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
            this.lblShapefile = new System.Windows.Forms.Label();
            this.txtShapefile = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.rtxSummary = new System.Windows.Forms.RichTextBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.chkLimitDisplay = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // lblShapefile
            // 
            this.lblShapefile.AutoSize = true;
            this.lblShapefile.Location = new System.Drawing.Point(12, 23);
            this.lblShapefile.Name = "lblShapefile";
            this.lblShapefile.Size = new System.Drawing.Size(51, 13);
            this.lblShapefile.TabIndex = 0;
            this.lblShapefile.Text = "Shapefile";
            // 
            // txtShapefile
            // 
            this.txtShapefile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.txtShapefile.Location = new System.Drawing.Point(15, 39);
            this.txtShapefile.Name = "txtShapefile";
            this.txtShapefile.Size = new System.Drawing.Size(350, 20);
            this.txtShapefile.TabIndex = 1;
            // 
            // btnBrowse
            // 
            this.btnBrowse.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnBrowse.Location = new System.Drawing.Point(371, 37);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(75, 23);
            this.btnBrowse.TabIndex = 2;
            this.btnBrowse.Text = "Browse";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // rtxSummary
            // 
            this.rtxSummary.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.rtxSummary.BackColor = System.Drawing.SystemColors.Window;
            this.rtxSummary.Location = new System.Drawing.Point(15, 85);
            this.rtxSummary.Name = "rtxSummary";
            this.rtxSummary.ReadOnly = true;
            this.rtxSummary.Size = new System.Drawing.Size(430, 180);
            this.rtxSummary.TabIndex = 3;
            this.rtxSummary.Text = "";
            this.rtxSummary.WordWrap = false;
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.Location = new System.Drawing.Point(292, 271);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 4;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(373, 271);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 5;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // chkLimitDisplay
            // 
            this.chkLimitDisplay.AutoSize = true;
            this.chkLimitDisplay.Location = new System.Drawing.Point(15, 65);
            this.chkLimitDisplay.Name = "chkLimitDisplay";
            this.chkLimitDisplay.Size = new System.Drawing.Size(211, 17);
            this.chkLimitDisplay.TabIndex = 6;
            this.chkLimitDisplay.Text = "Limit the display to the first 500 features";
            this.chkLimitDisplay.UseVisualStyleBackColor = true;
            this.chkLimitDisplay.CheckedChanged += new System.EventHandler(this.chkLimitDisplay_CheckedChanged);
            // 
            // ImportShapefilesDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(457, 302);
            this.Controls.Add(this.chkLimitDisplay);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.rtxSummary);
            this.Controls.Add(this.btnBrowse);
            this.Controls.Add(this.txtShapefile);
            this.Controls.Add(this.lblShapefile);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ImportShapefilesDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Import Shapefiles";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblShapefile;
        private System.Windows.Forms.TextBox txtShapefile;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.RichTextBox rtxSummary;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.CheckBox chkLimitDisplay;
    }
}