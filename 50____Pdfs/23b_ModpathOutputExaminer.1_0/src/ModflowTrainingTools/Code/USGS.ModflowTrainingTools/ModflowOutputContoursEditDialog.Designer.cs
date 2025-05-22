namespace USGS.ModflowTrainingTools
{
    partial class ModflowOutputContoursEditDialog
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ModflowOutputContoursEditDialog));
            this.tabData = new System.Windows.Forms.TabControl();
            this.tabPageDataLayer = new System.Windows.Forms.TabPage();
            this.lblSelectedDataLayer = new System.Windows.Forms.Label();
            this.tvwContents = new System.Windows.Forms.TreeView();
            this.imageListContourProperties = new System.Windows.Forms.ImageList(this.components);
            this.tabPageContours = new System.Windows.Forms.TabPage();
            this.txtReferenceContour = new System.Windows.Forms.TextBox();
            this.lblReferenceContour = new System.Windows.Forms.Label();
            this.gboxExcludedValues = new System.Windows.Forms.GroupBox();
            this.txtAddExcludedValue = new System.Windows.Forms.TextBox();
            this.lbxExcludedValues = new System.Windows.Forms.ListBox();
            this.btnAddExcludedValue = new System.Windows.Forms.Button();
            this.btnRemoveExcludedValue = new System.Windows.Forms.Button();
            this.gboxContourIntervalOption = new System.Windows.Forms.GroupBox();
            this.txtSpecifiedContourInterval = new System.Windows.Forms.TextBox();
            this.radioButtonSpecifiedContourInterval = new System.Windows.Forms.RadioButton();
            this.radioButtonAutomaticContourInterval = new System.Windows.Forms.RadioButton();
            this.tabPageSymbology = new System.Windows.Forms.TabPage();
            this.cboContourLineWidth = new System.Windows.Forms.ComboBox();
            this.lblContourLineWidth = new System.Windows.Forms.Label();
            this.btnSelectContourColor = new System.Windows.Forms.Button();
            this.panelContourColor = new System.Windows.Forms.Panel();
            this.lblColor = new System.Windows.Forms.Label();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.tabData.SuspendLayout();
            this.tabPageDataLayer.SuspendLayout();
            this.tabPageContours.SuspendLayout();
            this.gboxExcludedValues.SuspendLayout();
            this.gboxContourIntervalOption.SuspendLayout();
            this.tabPageSymbology.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabData
            // 
            this.tabData.Controls.Add(this.tabPageDataLayer);
            this.tabData.Controls.Add(this.tabPageContours);
            this.tabData.Controls.Add(this.tabPageSymbology);
            this.tabData.Location = new System.Drawing.Point(12, 12);
            this.tabData.Name = "tabData";
            this.tabData.SelectedIndex = 0;
            this.tabData.Size = new System.Drawing.Size(559, 324);
            this.tabData.TabIndex = 0;
            // 
            // tabPageDataLayer
            // 
            this.tabPageDataLayer.Controls.Add(this.lblSelectedDataLayer);
            this.tabPageDataLayer.Controls.Add(this.tvwContents);
            this.tabPageDataLayer.Location = new System.Drawing.Point(4, 22);
            this.tabPageDataLayer.Name = "tabPageDataLayer";
            this.tabPageDataLayer.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageDataLayer.Size = new System.Drawing.Size(551, 298);
            this.tabPageDataLayer.TabIndex = 0;
            this.tabPageDataLayer.Text = "Data Layer";
            this.tabPageDataLayer.UseVisualStyleBackColor = true;
            // 
            // lblSelectedDataLayer
            // 
            this.lblSelectedDataLayer.AutoSize = true;
            this.lblSelectedDataLayer.Location = new System.Drawing.Point(6, 21);
            this.lblSelectedDataLayer.Name = "lblSelectedDataLayer";
            this.lblSelectedDataLayer.Size = new System.Drawing.Size(101, 13);
            this.lblSelectedDataLayer.TabIndex = 1;
            this.lblSelectedDataLayer.Text = "Selected data layer:";
            // 
            // tvwContents
            // 
            this.tvwContents.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.tvwContents.ImageIndex = 0;
            this.tvwContents.ImageList = this.imageListContourProperties;
            this.tvwContents.Location = new System.Drawing.Point(6, 48);
            this.tvwContents.Name = "tvwContents";
            this.tvwContents.SelectedImageIndex = 0;
            this.tvwContents.Size = new System.Drawing.Size(539, 244);
            this.tvwContents.TabIndex = 0;
            this.tvwContents.NodeMouseClick += new System.Windows.Forms.TreeNodeMouseClickEventHandler(this.tvwContents_NodeMouseClick);
            // 
            // imageListContourProperties
            // 
            this.imageListContourProperties.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageListContourProperties.ImageStream")));
            this.imageListContourProperties.TransparentColor = System.Drawing.Color.Magenta;
            this.imageListContourProperties.Images.SetKeyName(0, "folder_closed.bmp");
            this.imageListContourProperties.Images.SetKeyName(1, "folder_open.bmp");
            this.imageListContourProperties.Images.SetKeyName(2, "LayerArray.bmp");
            // 
            // tabPageContours
            // 
            this.tabPageContours.Controls.Add(this.txtReferenceContour);
            this.tabPageContours.Controls.Add(this.lblReferenceContour);
            this.tabPageContours.Controls.Add(this.gboxExcludedValues);
            this.tabPageContours.Controls.Add(this.gboxContourIntervalOption);
            this.tabPageContours.Location = new System.Drawing.Point(4, 22);
            this.tabPageContours.Name = "tabPageContours";
            this.tabPageContours.Size = new System.Drawing.Size(551, 298);
            this.tabPageContours.TabIndex = 2;
            this.tabPageContours.Text = " Contours";
            this.tabPageContours.UseVisualStyleBackColor = true;
            // 
            // txtReferenceContour
            // 
            this.txtReferenceContour.Location = new System.Drawing.Point(130, 19);
            this.txtReferenceContour.Name = "txtReferenceContour";
            this.txtReferenceContour.Size = new System.Drawing.Size(100, 20);
            this.txtReferenceContour.TabIndex = 11;
            this.txtReferenceContour.TextChanged += new System.EventHandler(this.txtReferenceContour_TextChanged);
            // 
            // lblReferenceContour
            // 
            this.lblReferenceContour.AutoSize = true;
            this.lblReferenceContour.Location = new System.Drawing.Point(25, 22);
            this.lblReferenceContour.Name = "lblReferenceContour";
            this.lblReferenceContour.Size = new System.Drawing.Size(99, 13);
            this.lblReferenceContour.TabIndex = 10;
            this.lblReferenceContour.Text = "Reference contour:";
            // 
            // gboxExcludedValues
            // 
            this.gboxExcludedValues.Controls.Add(this.txtAddExcludedValue);
            this.gboxExcludedValues.Controls.Add(this.lbxExcludedValues);
            this.gboxExcludedValues.Controls.Add(this.btnAddExcludedValue);
            this.gboxExcludedValues.Controls.Add(this.btnRemoveExcludedValue);
            this.gboxExcludedValues.Location = new System.Drawing.Point(28, 170);
            this.gboxExcludedValues.Name = "gboxExcludedValues";
            this.gboxExcludedValues.Size = new System.Drawing.Size(500, 115);
            this.gboxExcludedValues.TabIndex = 9;
            this.gboxExcludedValues.TabStop = false;
            this.gboxExcludedValues.Text = "Excluded Values";
            // 
            // txtAddExcludedValue
            // 
            this.txtAddExcludedValue.Location = new System.Drawing.Point(419, 38);
            this.txtAddExcludedValue.Name = "txtAddExcludedValue";
            this.txtAddExcludedValue.Size = new System.Drawing.Size(60, 20);
            this.txtAddExcludedValue.TabIndex = 8;
            // 
            // lbxExcludedValues
            // 
            this.lbxExcludedValues.FormattingEnabled = true;
            this.lbxExcludedValues.Location = new System.Drawing.Point(16, 19);
            this.lbxExcludedValues.Name = "lbxExcludedValues";
            this.lbxExcludedValues.Size = new System.Drawing.Size(231, 82);
            this.lbxExcludedValues.TabIndex = 5;
            // 
            // btnAddExcludedValue
            // 
            this.btnAddExcludedValue.Location = new System.Drawing.Point(275, 36);
            this.btnAddExcludedValue.Name = "btnAddExcludedValue";
            this.btnAddExcludedValue.Size = new System.Drawing.Size(138, 23);
            this.btnAddExcludedValue.TabIndex = 6;
            this.btnAddExcludedValue.Text = "  Add Excluded Value >>";
            this.btnAddExcludedValue.UseVisualStyleBackColor = true;
            this.btnAddExcludedValue.Click += new System.EventHandler(this.btnAddExcludedValue_Click);
            // 
            // btnRemoveExcludedValue
            // 
            this.btnRemoveExcludedValue.Location = new System.Drawing.Point(275, 65);
            this.btnRemoveExcludedValue.Name = "btnRemoveExcludedValue";
            this.btnRemoveExcludedValue.Size = new System.Drawing.Size(138, 23);
            this.btnRemoveExcludedValue.TabIndex = 7;
            this.btnRemoveExcludedValue.Text = "  Remove Selected Value";
            this.btnRemoveExcludedValue.UseVisualStyleBackColor = true;
            this.btnRemoveExcludedValue.Click += new System.EventHandler(this.btnRemoveExcludedValue_Click);
            // 
            // gboxContourIntervalOption
            // 
            this.gboxContourIntervalOption.Controls.Add(this.txtSpecifiedContourInterval);
            this.gboxContourIntervalOption.Controls.Add(this.radioButtonSpecifiedContourInterval);
            this.gboxContourIntervalOption.Controls.Add(this.radioButtonAutomaticContourInterval);
            this.gboxContourIntervalOption.Location = new System.Drawing.Point(28, 56);
            this.gboxContourIntervalOption.Name = "gboxContourIntervalOption";
            this.gboxContourIntervalOption.Size = new System.Drawing.Size(500, 97);
            this.gboxContourIntervalOption.TabIndex = 8;
            this.gboxContourIntervalOption.TabStop = false;
            this.gboxContourIntervalOption.Text = "Contour Interval Option";
            // 
            // txtSpecifiedContourInterval
            // 
            this.txtSpecifiedContourInterval.Location = new System.Drawing.Point(325, 61);
            this.txtSpecifiedContourInterval.Name = "txtSpecifiedContourInterval";
            this.txtSpecifiedContourInterval.Size = new System.Drawing.Size(75, 20);
            this.txtSpecifiedContourInterval.TabIndex = 2;
            // 
            // radioButtonSpecifiedContourInterval
            // 
            this.radioButtonSpecifiedContourInterval.AutoSize = true;
            this.radioButtonSpecifiedContourInterval.Location = new System.Drawing.Point(16, 62);
            this.radioButtonSpecifiedContourInterval.Name = "radioButtonSpecifiedContourInterval";
            this.radioButtonSpecifiedContourInterval.Size = new System.Drawing.Size(306, 17);
            this.radioButtonSpecifiedContourInterval.TabIndex = 1;
            this.radioButtonSpecifiedContourInterval.TabStop = true;
            this.radioButtonSpecifiedContourInterval.Text = "Specify a constant contour interval to use for all data layers:";
            this.radioButtonSpecifiedContourInterval.UseVisualStyleBackColor = true;
            // 
            // radioButtonAutomaticContourInterval
            // 
            this.radioButtonAutomaticContourInterval.AutoSize = true;
            this.radioButtonAutomaticContourInterval.Location = new System.Drawing.Point(16, 28);
            this.radioButtonAutomaticContourInterval.Name = "radioButtonAutomaticContourInterval";
            this.radioButtonAutomaticContourInterval.Size = new System.Drawing.Size(414, 17);
            this.radioButtonAutomaticContourInterval.TabIndex = 0;
            this.radioButtonAutomaticContourInterval.TabStop = true;
            this.radioButtonAutomaticContourInterval.Text = "Automatically generate an appropriate constant contour interval for each data lay" +
                "er";
            this.radioButtonAutomaticContourInterval.UseVisualStyleBackColor = true;
            // 
            // tabPageSymbology
            // 
            this.tabPageSymbology.Controls.Add(this.cboContourLineWidth);
            this.tabPageSymbology.Controls.Add(this.lblContourLineWidth);
            this.tabPageSymbology.Controls.Add(this.btnSelectContourColor);
            this.tabPageSymbology.Controls.Add(this.panelContourColor);
            this.tabPageSymbology.Controls.Add(this.lblColor);
            this.tabPageSymbology.Location = new System.Drawing.Point(4, 22);
            this.tabPageSymbology.Name = "tabPageSymbology";
            this.tabPageSymbology.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageSymbology.Size = new System.Drawing.Size(551, 298);
            this.tabPageSymbology.TabIndex = 1;
            this.tabPageSymbology.Text = "Symbology";
            this.tabPageSymbology.UseVisualStyleBackColor = true;
            // 
            // cboContourLineWidth
            // 
            this.cboContourLineWidth.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboContourLineWidth.FormattingEnabled = true;
            this.cboContourLineWidth.Location = new System.Drawing.Point(382, 60);
            this.cboContourLineWidth.Name = "cboContourLineWidth";
            this.cboContourLineWidth.Size = new System.Drawing.Size(54, 21);
            this.cboContourLineWidth.TabIndex = 4;
            // 
            // lblContourLineWidth
            // 
            this.lblContourLineWidth.AutoSize = true;
            this.lblContourLineWidth.Location = new System.Drawing.Point(318, 63);
            this.lblContourLineWidth.Name = "lblContourLineWidth";
            this.lblContourLineWidth.Size = new System.Drawing.Size(58, 13);
            this.lblContourLineWidth.TabIndex = 3;
            this.lblContourLineWidth.Text = "Line width:";
            // 
            // btnSelectContourColor
            // 
            this.btnSelectContourColor.Location = new System.Drawing.Point(163, 58);
            this.btnSelectContourColor.Name = "btnSelectContourColor";
            this.btnSelectContourColor.Size = new System.Drawing.Size(98, 23);
            this.btnSelectContourColor.TabIndex = 2;
            this.btnSelectContourColor.Text = "Select Color";
            this.btnSelectContourColor.UseVisualStyleBackColor = true;
            this.btnSelectContourColor.Click += new System.EventHandler(this.btnSelectContourColor_Click);
            // 
            // panelContourColor
            // 
            this.panelContourColor.BackColor = System.Drawing.Color.Black;
            this.panelContourColor.Location = new System.Drawing.Point(116, 63);
            this.panelContourColor.Name = "panelContourColor";
            this.panelContourColor.Size = new System.Drawing.Size(31, 13);
            this.panelContourColor.TabIndex = 1;
            // 
            // lblColor
            // 
            this.lblColor.AutoSize = true;
            this.lblColor.Location = new System.Drawing.Point(76, 63);
            this.lblColor.Name = "lblColor";
            this.lblColor.Size = new System.Drawing.Size(34, 13);
            this.lblColor.TabIndex = 0;
            this.lblColor.Text = "Color:";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.Location = new System.Drawing.Point(417, 342);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 1;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.Location = new System.Drawing.Point(496, 342);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 2;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // ModflowOutputContoursEditDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(583, 368);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.tabData);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ModflowOutputContoursEditDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Contour Properties";
            this.tabData.ResumeLayout(false);
            this.tabPageDataLayer.ResumeLayout(false);
            this.tabPageDataLayer.PerformLayout();
            this.tabPageContours.ResumeLayout(false);
            this.tabPageContours.PerformLayout();
            this.gboxExcludedValues.ResumeLayout(false);
            this.gboxExcludedValues.PerformLayout();
            this.gboxContourIntervalOption.ResumeLayout(false);
            this.gboxContourIntervalOption.PerformLayout();
            this.tabPageSymbology.ResumeLayout(false);
            this.tabPageSymbology.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabData;
        private System.Windows.Forms.TabPage tabPageDataLayer;
        private System.Windows.Forms.TabPage tabPageSymbology;
        private System.Windows.Forms.TreeView tvwContents;
        private System.Windows.Forms.Label lblSelectedDataLayer;
        private System.Windows.Forms.TabPage tabPageContours;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ImageList imageListContourProperties;
        private System.Windows.Forms.ListBox lbxExcludedValues;
        private System.Windows.Forms.Button btnRemoveExcludedValue;
        private System.Windows.Forms.Button btnAddExcludedValue;
        private System.Windows.Forms.GroupBox gboxContourIntervalOption;
        private System.Windows.Forms.GroupBox gboxExcludedValues;
        private System.Windows.Forms.TextBox txtSpecifiedContourInterval;
        private System.Windows.Forms.RadioButton radioButtonSpecifiedContourInterval;
        private System.Windows.Forms.RadioButton radioButtonAutomaticContourInterval;
        private System.Windows.Forms.TextBox txtReferenceContour;
        private System.Windows.Forms.Label lblReferenceContour;
        private System.Windows.Forms.TextBox txtAddExcludedValue;
        private System.Windows.Forms.ComboBox cboContourLineWidth;
        private System.Windows.Forms.Label lblContourLineWidth;
        private System.Windows.Forms.Button btnSelectContourColor;
        private System.Windows.Forms.Panel panelContourColor;
        private System.Windows.Forms.Label lblColor;
    }
}