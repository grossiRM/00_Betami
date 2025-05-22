namespace USGS.Puma.UI.Modpath
{
    partial class QueryFilter
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
            this.radioBtnFilteringIsOff = new System.Windows.Forms.RadioButton();
            this.radioBtnFilteringIsOn = new System.Windows.Forms.RadioButton();
            this.gboxQueryDef = new System.Windows.Forms.GroupBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // radioBtnFilteringIsOff
            // 
            this.radioBtnFilteringIsOff.AutoSize = true;
            this.radioBtnFilteringIsOff.Checked = true;
            this.radioBtnFilteringIsOff.Location = new System.Drawing.Point(12, 12);
            this.radioBtnFilteringIsOff.Name = "radioBtnFilteringIsOff";
            this.radioBtnFilteringIsOff.Size = new System.Drawing.Size(127, 17);
            this.radioBtnFilteringIsOff.TabIndex = 0;
            this.radioBtnFilteringIsOff.TabStop = true;
            this.radioBtnFilteringIsOff.Text = "Show all data records";
            this.radioBtnFilteringIsOff.UseVisualStyleBackColor = true;
            // 
            // radioBtnFilteringIsOn
            // 
            this.radioBtnFilteringIsOn.AutoSize = true;
            this.radioBtnFilteringIsOn.Location = new System.Drawing.Point(12, 35);
            this.radioBtnFilteringIsOn.Name = "radioBtnFilteringIsOn";
            this.radioBtnFilteringIsOn.Size = new System.Drawing.Size(311, 17);
            this.radioBtnFilteringIsOn.TabIndex = 1;
            this.radioBtnFilteringIsOn.Text = "Show only those data records that meet the following criteria:";
            this.radioBtnFilteringIsOn.UseVisualStyleBackColor = true;
            this.radioBtnFilteringIsOn.CheckedChanged += new System.EventHandler(this.radioBtnFilteringIsOn_CheckedChanged);
            // 
            // gboxQueryDef
            // 
            this.gboxQueryDef.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.gboxQueryDef.Enabled = false;
            this.gboxQueryDef.Location = new System.Drawing.Point(34, 58);
            this.gboxQueryDef.Name = "gboxQueryDef";
            this.gboxQueryDef.Size = new System.Drawing.Size(406, 177);
            this.gboxQueryDef.TabIndex = 2;
            this.gboxQueryDef.TabStop = false;
            this.gboxQueryDef.Text = "Query Definition";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.Location = new System.Drawing.Point(284, 241);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 3;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(365, 241);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 4;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // QueryFilter
            // 
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;
            this.ClientSize = new System.Drawing.Size(452, 271);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.gboxQueryDef);
            this.Controls.Add(this.radioBtnFilteringIsOn);
            this.Controls.Add(this.radioBtnFilteringIsOff);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "QueryFilter";
            this.Text = "QueryFilter";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        protected System.Windows.Forms.RadioButton radioBtnFilteringIsOff;
        protected System.Windows.Forms.RadioButton radioBtnFilteringIsOn;
        protected System.Windows.Forms.GroupBox gboxQueryDef;
        protected System.Windows.Forms.Button btnOK;
        protected System.Windows.Forms.Button btnCancel;
    }
}