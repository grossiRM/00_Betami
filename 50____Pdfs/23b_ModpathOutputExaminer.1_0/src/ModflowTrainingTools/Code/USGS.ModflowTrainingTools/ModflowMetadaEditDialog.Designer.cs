namespace USGS.ModflowTrainingTools
{
    partial class ModflowMetadaEditDialog
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
            this.tabMetadata = new System.Windows.Forms.TabControl();
            this.tabPageGeoReference = new System.Windows.Forms.TabPage();
            this.btnCalculateAngle = new System.Windows.Forms.Button();
            this.txtGeoReferenceGridAngle = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.lblOriginY = new System.Windows.Forms.Label();
            this.lblOriginX = new System.Windows.Forms.Label();
            this.txtOriginY = new System.Windows.Forms.TextBox();
            this.txtOriginX = new System.Windows.Forms.TextBox();
            this.labelOrigin = new System.Windows.Forms.Label();
            this.tabPageBasemap = new System.Windows.Forms.TabPage();
            this.btnUseCurrentBasemap = new System.Windows.Forms.Button();
            this.btnNoBasemap = new System.Windows.Forms.Button();
            this.btnBasemapBrowse = new System.Windows.Forms.Button();
            this.txtDefaultBasemap = new System.Windows.Forms.TextBox();
            this.lblDefaultBasemap = new System.Windows.Forms.Label();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.tabMetadata.SuspendLayout();
            this.tabPageGeoReference.SuspendLayout();
            this.tabPageBasemap.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabMetadata
            // 
            this.tabMetadata.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.tabMetadata.Controls.Add(this.tabPageGeoReference);
            this.tabMetadata.Controls.Add(this.tabPageBasemap);
            this.tabMetadata.Location = new System.Drawing.Point(12, 12);
            this.tabMetadata.Name = "tabMetadata";
            this.tabMetadata.SelectedIndex = 0;
            this.tabMetadata.Size = new System.Drawing.Size(485, 283);
            this.tabMetadata.TabIndex = 0;
            // 
            // tabPageGeoReference
            // 
            this.tabPageGeoReference.Controls.Add(this.btnCalculateAngle);
            this.tabPageGeoReference.Controls.Add(this.txtGeoReferenceGridAngle);
            this.tabPageGeoReference.Controls.Add(this.label1);
            this.tabPageGeoReference.Controls.Add(this.lblOriginY);
            this.tabPageGeoReference.Controls.Add(this.lblOriginX);
            this.tabPageGeoReference.Controls.Add(this.txtOriginY);
            this.tabPageGeoReference.Controls.Add(this.txtOriginX);
            this.tabPageGeoReference.Controls.Add(this.labelOrigin);
            this.tabPageGeoReference.Location = new System.Drawing.Point(4, 22);
            this.tabPageGeoReference.Name = "tabPageGeoReference";
            this.tabPageGeoReference.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageGeoReference.Size = new System.Drawing.Size(477, 257);
            this.tabPageGeoReference.TabIndex = 0;
            this.tabPageGeoReference.Text = "Grid GeoReference";
            this.tabPageGeoReference.UseVisualStyleBackColor = true;
            // 
            // btnCalculateAngle
            // 
            this.btnCalculateAngle.Location = new System.Drawing.Point(294, 119);
            this.btnCalculateAngle.Name = "btnCalculateAngle";
            this.btnCalculateAngle.Size = new System.Drawing.Size(167, 23);
            this.btnCalculateAngle.TabIndex = 8;
            this.btnCalculateAngle.Text = "Calculate Grid Angle";
            this.btnCalculateAngle.UseVisualStyleBackColor = true;
            this.btnCalculateAngle.Click += new System.EventHandler(this.btnCalculateAngle_Click);
            // 
            // txtGeoReferenceGridAngle
            // 
            this.txtGeoReferenceGridAngle.Location = new System.Drawing.Point(84, 121);
            this.txtGeoReferenceGridAngle.Name = "txtGeoReferenceGridAngle";
            this.txtGeoReferenceGridAngle.Size = new System.Drawing.Size(204, 20);
            this.txtGeoReferenceGridAngle.TabIndex = 7;
            this.txtGeoReferenceGridAngle.Validating += new System.ComponentModel.CancelEventHandler(this.txtGeoReferenceGridAngle_Validating);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(19, 124);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(58, 13);
            this.label1.TabIndex = 6;
            this.label1.Text = "Grid angle:";
            // 
            // lblOriginY
            // 
            this.lblOriginY.AutoSize = true;
            this.lblOriginY.Location = new System.Drawing.Point(276, 53);
            this.lblOriginY.Name = "lblOriginY";
            this.lblOriginY.Size = new System.Drawing.Size(12, 13);
            this.lblOriginY.TabIndex = 4;
            this.lblOriginY.Text = "y";
            // 
            // lblOriginX
            // 
            this.lblOriginX.AutoSize = true;
            this.lblOriginX.Location = new System.Drawing.Point(66, 53);
            this.lblOriginX.Name = "lblOriginX";
            this.lblOriginX.Size = new System.Drawing.Size(12, 13);
            this.lblOriginX.TabIndex = 3;
            this.lblOriginX.Text = "x";
            // 
            // txtOriginY
            // 
            this.txtOriginY.Location = new System.Drawing.Point(294, 50);
            this.txtOriginY.Name = "txtOriginY";
            this.txtOriginY.Size = new System.Drawing.Size(167, 20);
            this.txtOriginY.TabIndex = 2;
            this.txtOriginY.Validating += new System.ComponentModel.CancelEventHandler(this.txtOriginY_Validating);
            // 
            // txtOriginX
            // 
            this.txtOriginX.Location = new System.Drawing.Point(84, 50);
            this.txtOriginX.Name = "txtOriginX";
            this.txtOriginX.Size = new System.Drawing.Size(167, 20);
            this.txtOriginX.TabIndex = 1;
            this.txtOriginX.Validating += new System.ComponentModel.CancelEventHandler(this.txtOriginX_Validating);
            // 
            // labelOrigin
            // 
            this.labelOrigin.AutoSize = true;
            this.labelOrigin.Location = new System.Drawing.Point(19, 53);
            this.labelOrigin.Name = "labelOrigin";
            this.labelOrigin.Size = new System.Drawing.Size(37, 13);
            this.labelOrigin.TabIndex = 0;
            this.labelOrigin.Text = "Origin:";
            // 
            // tabPageBasemap
            // 
            this.tabPageBasemap.Controls.Add(this.btnUseCurrentBasemap);
            this.tabPageBasemap.Controls.Add(this.btnNoBasemap);
            this.tabPageBasemap.Controls.Add(this.btnBasemapBrowse);
            this.tabPageBasemap.Controls.Add(this.txtDefaultBasemap);
            this.tabPageBasemap.Controls.Add(this.lblDefaultBasemap);
            this.tabPageBasemap.Location = new System.Drawing.Point(4, 22);
            this.tabPageBasemap.Name = "tabPageBasemap";
            this.tabPageBasemap.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageBasemap.Size = new System.Drawing.Size(477, 257);
            this.tabPageBasemap.TabIndex = 1;
            this.tabPageBasemap.Text = "Default Basemap";
            this.tabPageBasemap.UseVisualStyleBackColor = true;
            // 
            // btnUseCurrentBasemap
            // 
            this.btnUseCurrentBasemap.Location = new System.Drawing.Point(334, 146);
            this.btnUseCurrentBasemap.Name = "btnUseCurrentBasemap";
            this.btnUseCurrentBasemap.Size = new System.Drawing.Size(137, 23);
            this.btnUseCurrentBasemap.TabIndex = 4;
            this.btnUseCurrentBasemap.Text = "Use Current Basemap";
            this.btnUseCurrentBasemap.UseVisualStyleBackColor = true;
            this.btnUseCurrentBasemap.Click += new System.EventHandler(this.btnUseCurrentBasemap_Click);
            // 
            // btnNoBasemap
            // 
            this.btnNoBasemap.Location = new System.Drawing.Point(334, 117);
            this.btnNoBasemap.Name = "btnNoBasemap";
            this.btnNoBasemap.Size = new System.Drawing.Size(137, 23);
            this.btnNoBasemap.TabIndex = 3;
            this.btnNoBasemap.Text = "No Default Basemap";
            this.btnNoBasemap.UseVisualStyleBackColor = true;
            this.btnNoBasemap.Click += new System.EventHandler(this.btnNoBasemap_Click);
            // 
            // btnBasemapBrowse
            // 
            this.btnBasemapBrowse.Location = new System.Drawing.Point(331, 88);
            this.btnBasemapBrowse.Name = "btnBasemapBrowse";
            this.btnBasemapBrowse.Size = new System.Drawing.Size(140, 23);
            this.btnBasemapBrowse.TabIndex = 2;
            this.btnBasemapBrowse.Text = "Browse";
            this.btnBasemapBrowse.UseVisualStyleBackColor = true;
            this.btnBasemapBrowse.Click += new System.EventHandler(this.btnBasemapBrowse_Click);
            // 
            // txtDefaultBasemap
            // 
            this.txtDefaultBasemap.Location = new System.Drawing.Point(6, 53);
            this.txtDefaultBasemap.Name = "txtDefaultBasemap";
            this.txtDefaultBasemap.ReadOnly = true;
            this.txtDefaultBasemap.Size = new System.Drawing.Size(465, 20);
            this.txtDefaultBasemap.TabIndex = 1;
            // 
            // lblDefaultBasemap
            // 
            this.lblDefaultBasemap.AutoSize = true;
            this.lblDefaultBasemap.Location = new System.Drawing.Point(3, 37);
            this.lblDefaultBasemap.Name = "lblDefaultBasemap";
            this.lblDefaultBasemap.Size = new System.Drawing.Size(91, 13);
            this.lblDefaultBasemap.TabIndex = 0;
            this.lblDefaultBasemap.Text = "Default Basemap:";
            // 
            // btnSave
            // 
            this.btnSave.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnSave.Location = new System.Drawing.Point(341, 301);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(75, 23);
            this.btnSave.TabIndex = 1;
            this.btnSave.Text = "Save";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(422, 301);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 2;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // ModflowMetadaEditDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(509, 329);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.tabMetadata);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ModflowMetadaEditDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Edit MODFLOW Metadata";
            this.tabMetadata.ResumeLayout(false);
            this.tabPageGeoReference.ResumeLayout(false);
            this.tabPageGeoReference.PerformLayout();
            this.tabPageBasemap.ResumeLayout(false);
            this.tabPageBasemap.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabMetadata;
        private System.Windows.Forms.TabPage tabPageGeoReference;
        private System.Windows.Forms.TabPage tabPageBasemap;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label labelOrigin;
        private System.Windows.Forms.Label lblOriginY;
        private System.Windows.Forms.Label lblOriginX;
        private System.Windows.Forms.TextBox txtOriginY;
        private System.Windows.Forms.TextBox txtOriginX;
        private System.Windows.Forms.TextBox txtGeoReferenceGridAngle;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnCalculateAngle;
        private System.Windows.Forms.Button btnBasemapBrowse;
        private System.Windows.Forms.TextBox txtDefaultBasemap;
        private System.Windows.Forms.Label lblDefaultBasemap;
        private System.Windows.Forms.Button btnNoBasemap;
        private System.Windows.Forms.Button btnUseCurrentBasemap;
    }
}