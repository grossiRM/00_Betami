namespace ModpathOutputExaminer
{
    partial class TimeseriesSymbologyDialog
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
            this.lblSize = new System.Windows.Forms.Label();
            this.radioBtnSingleSymbol = new System.Windows.Forms.RadioButton();
            this.panelSingleSymbolColor = new System.Windows.Forms.Panel();
            this.buttonSelectColor = new System.Windows.Forms.Button();
            this.buttonOK = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.radioBtnUniqueValues = new System.Windows.Forms.RadioButton();
            this.radioBtnContinuousValues = new System.Windows.Forms.RadioButton();
            this.panelColorRamps = new System.Windows.Forms.Panel();
            this.cboContinuousRenderField = new System.Windows.Forms.ComboBox();
            this.labelContinuousRenderField = new System.Windows.Forms.Label();
            this.radioBtnBlueToGreen = new System.Windows.Forms.RadioButton();
            this.radioBtnRedToBlue = new System.Windows.Forms.RadioButton();
            this.radioBtnRedToGreen = new System.Windows.Forms.RadioButton();
            this.colorBarBlueToGreen = new USGS.Puma.UI.MapViewer.ColorBar();
            this.colorBarRedToBlue = new USGS.Puma.UI.MapViewer.ColorBar();
            this.colorBarRedToGreen = new USGS.Puma.UI.MapViewer.ColorBar();
            this.colorBarRainbow5 = new USGS.Puma.UI.MapViewer.ColorBar();
            this.radioBtnRainbow5 = new System.Windows.Forms.RadioButton();
            this.colorBarRainbow7 = new USGS.Puma.UI.MapViewer.ColorBar();
            this.radioBtnRainbow7 = new System.Windows.Forms.RadioButton();
            this.panelUniqueValues = new System.Windows.Forms.Panel();
            this.radioBtnUvPalette3 = new System.Windows.Forms.RadioButton();
            this.radioBtnUvPalette2 = new System.Windows.Forms.RadioButton();
            this.radioBtnUvPalette1 = new System.Windows.Forms.RadioButton();
            this.colorBarUvPalette3 = new USGS.Puma.UI.MapViewer.ColorBar();
            this.colorBarUvPalette2 = new USGS.Puma.UI.MapViewer.ColorBar();
            this.colorBarUvPalette1 = new USGS.Puma.UI.MapViewer.ColorBar();
            this.cboUniqueValuesRenderField = new System.Windows.Forms.ComboBox();
            this.labelUniqueValuesRenderField = new System.Windows.Forms.Label();
            this.panelSingleSymbol = new System.Windows.Forms.Panel();
            this.cboSymbolType = new System.Windows.Forms.ComboBox();
            this.cboSize = new System.Windows.Forms.ComboBox();
            this.panelColorRamps.SuspendLayout();
            this.panelUniqueValues.SuspendLayout();
            this.panelSingleSymbol.SuspendLayout();
            this.SuspendLayout();
            // 
            // lblSize
            // 
            this.lblSize.AutoSize = true;
            this.lblSize.Location = new System.Drawing.Point(134, 15);
            this.lblSize.Name = "lblSize";
            this.lblSize.Size = new System.Drawing.Size(30, 13);
            this.lblSize.TabIndex = 0;
            this.lblSize.Text = "Size:";
            // 
            // radioBtnSingleSymbol
            // 
            this.radioBtnSingleSymbol.AutoSize = true;
            this.radioBtnSingleSymbol.Location = new System.Drawing.Point(31, 59);
            this.radioBtnSingleSymbol.Name = "radioBtnSingleSymbol";
            this.radioBtnSingleSymbol.Size = new System.Drawing.Size(89, 17);
            this.radioBtnSingleSymbol.TabIndex = 2;
            this.radioBtnSingleSymbol.TabStop = true;
            this.radioBtnSingleSymbol.Text = "Single symbol";
            this.radioBtnSingleSymbol.UseVisualStyleBackColor = true;
            // 
            // panelSingleSymbolColor
            // 
            this.panelSingleSymbolColor.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelSingleSymbolColor.Location = new System.Drawing.Point(1, 2);
            this.panelSingleSymbolColor.Name = "panelSingleSymbolColor";
            this.panelSingleSymbolColor.Size = new System.Drawing.Size(24, 24);
            this.panelSingleSymbolColor.TabIndex = 3;
            // 
            // buttonSelectColor
            // 
            this.buttonSelectColor.Location = new System.Drawing.Point(31, 3);
            this.buttonSelectColor.Name = "buttonSelectColor";
            this.buttonSelectColor.Size = new System.Drawing.Size(75, 23);
            this.buttonSelectColor.TabIndex = 4;
            this.buttonSelectColor.Text = "Select Color";
            this.buttonSelectColor.UseVisualStyleBackColor = true;
            this.buttonSelectColor.Click += new System.EventHandler(this.buttonSelectColor_Click);
            // 
            // buttonOK
            // 
            this.buttonOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonOK.Location = new System.Drawing.Point(191, 515);
            this.buttonOK.Name = "buttonOK";
            this.buttonOK.Size = new System.Drawing.Size(75, 23);
            this.buttonOK.TabIndex = 5;
            this.buttonOK.Text = "OK";
            this.buttonOK.UseVisualStyleBackColor = true;
            this.buttonOK.Click += new System.EventHandler(this.buttonOK_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonCancel.Location = new System.Drawing.Point(272, 515);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(75, 23);
            this.buttonCancel.TabIndex = 6;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // radioBtnUniqueValues
            // 
            this.radioBtnUniqueValues.AutoSize = true;
            this.radioBtnUniqueValues.Location = new System.Drawing.Point(31, 124);
            this.radioBtnUniqueValues.Name = "radioBtnUniqueValues";
            this.radioBtnUniqueValues.Size = new System.Drawing.Size(93, 17);
            this.radioBtnUniqueValues.TabIndex = 7;
            this.radioBtnUniqueValues.TabStop = true;
            this.radioBtnUniqueValues.Text = "Unique values";
            this.radioBtnUniqueValues.UseVisualStyleBackColor = true;
            // 
            // radioBtnContinuousValues
            // 
            this.radioBtnContinuousValues.AutoSize = true;
            this.radioBtnContinuousValues.Location = new System.Drawing.Point(31, 307);
            this.radioBtnContinuousValues.Name = "radioBtnContinuousValues";
            this.radioBtnContinuousValues.Size = new System.Drawing.Size(130, 17);
            this.radioBtnContinuousValues.TabIndex = 8;
            this.radioBtnContinuousValues.TabStop = true;
            this.radioBtnContinuousValues.Text = "Continuous color ramp";
            this.radioBtnContinuousValues.UseVisualStyleBackColor = true;
            // 
            // panelColorRamps
            // 
            this.panelColorRamps.Controls.Add(this.cboContinuousRenderField);
            this.panelColorRamps.Controls.Add(this.labelContinuousRenderField);
            this.panelColorRamps.Controls.Add(this.radioBtnBlueToGreen);
            this.panelColorRamps.Controls.Add(this.radioBtnRedToBlue);
            this.panelColorRamps.Controls.Add(this.radioBtnRedToGreen);
            this.panelColorRamps.Controls.Add(this.colorBarBlueToGreen);
            this.panelColorRamps.Controls.Add(this.colorBarRedToBlue);
            this.panelColorRamps.Controls.Add(this.colorBarRedToGreen);
            this.panelColorRamps.Controls.Add(this.colorBarRainbow5);
            this.panelColorRamps.Controls.Add(this.radioBtnRainbow5);
            this.panelColorRamps.Controls.Add(this.colorBarRainbow7);
            this.panelColorRamps.Controls.Add(this.radioBtnRainbow7);
            this.panelColorRamps.Location = new System.Drawing.Point(52, 330);
            this.panelColorRamps.Name = "panelColorRamps";
            this.panelColorRamps.Size = new System.Drawing.Size(291, 165);
            this.panelColorRamps.TabIndex = 9;
            // 
            // cboContinuousRenderField
            // 
            this.cboContinuousRenderField.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboContinuousRenderField.FormattingEnabled = true;
            this.cboContinuousRenderField.Location = new System.Drawing.Point(3, 16);
            this.cboContinuousRenderField.Name = "cboContinuousRenderField";
            this.cboContinuousRenderField.Size = new System.Drawing.Size(261, 21);
            this.cboContinuousRenderField.TabIndex = 11;
            // 
            // labelContinuousRenderField
            // 
            this.labelContinuousRenderField.AutoSize = true;
            this.labelContinuousRenderField.Location = new System.Drawing.Point(3, 0);
            this.labelContinuousRenderField.Name = "labelContinuousRenderField";
            this.labelContinuousRenderField.Size = new System.Drawing.Size(70, 13);
            this.labelContinuousRenderField.TabIndex = 10;
            this.labelContinuousRenderField.Text = "Render Field:";
            // 
            // radioBtnBlueToGreen
            // 
            this.radioBtnBlueToGreen.AutoSize = true;
            this.radioBtnBlueToGreen.Location = new System.Drawing.Point(3, 137);
            this.radioBtnBlueToGreen.Name = "radioBtnBlueToGreen";
            this.radioBtnBlueToGreen.Size = new System.Drawing.Size(14, 13);
            this.radioBtnBlueToGreen.TabIndex = 9;
            this.radioBtnBlueToGreen.TabStop = true;
            this.radioBtnBlueToGreen.UseVisualStyleBackColor = true;
            // 
            // radioBtnRedToBlue
            // 
            this.radioBtnRedToBlue.AutoSize = true;
            this.radioBtnRedToBlue.Location = new System.Drawing.Point(3, 114);
            this.radioBtnRedToBlue.Name = "radioBtnRedToBlue";
            this.radioBtnRedToBlue.Size = new System.Drawing.Size(14, 13);
            this.radioBtnRedToBlue.TabIndex = 8;
            this.radioBtnRedToBlue.TabStop = true;
            this.radioBtnRedToBlue.UseVisualStyleBackColor = true;
            // 
            // radioBtnRedToGreen
            // 
            this.radioBtnRedToGreen.AutoSize = true;
            this.radioBtnRedToGreen.Location = new System.Drawing.Point(3, 91);
            this.radioBtnRedToGreen.Name = "radioBtnRedToGreen";
            this.radioBtnRedToGreen.Size = new System.Drawing.Size(14, 13);
            this.radioBtnRedToGreen.TabIndex = 7;
            this.radioBtnRedToGreen.TabStop = true;
            this.radioBtnRedToGreen.UseVisualStyleBackColor = true;
            // 
            // colorBarBlueToGreen
            // 
            this.colorBarBlueToGreen.Colors = null;
            this.colorBarBlueToGreen.Location = new System.Drawing.Point(23, 135);
            this.colorBarBlueToGreen.Name = "colorBarBlueToGreen";
            this.colorBarBlueToGreen.Size = new System.Drawing.Size(241, 17);
            this.colorBarBlueToGreen.TabIndex = 6;
            this.colorBarBlueToGreen.Click += new System.EventHandler(this.colorBarBlueToGreen_Click);
            // 
            // colorBarRedToBlue
            // 
            this.colorBarRedToBlue.Colors = null;
            this.colorBarRedToBlue.Location = new System.Drawing.Point(23, 112);
            this.colorBarRedToBlue.Name = "colorBarRedToBlue";
            this.colorBarRedToBlue.Size = new System.Drawing.Size(241, 17);
            this.colorBarRedToBlue.TabIndex = 5;
            this.colorBarRedToBlue.Click += new System.EventHandler(this.colorBarRedToBlue_Click);
            // 
            // colorBarRedToGreen
            // 
            this.colorBarRedToGreen.Colors = null;
            this.colorBarRedToGreen.Location = new System.Drawing.Point(23, 89);
            this.colorBarRedToGreen.Name = "colorBarRedToGreen";
            this.colorBarRedToGreen.Size = new System.Drawing.Size(241, 17);
            this.colorBarRedToGreen.TabIndex = 4;
            this.colorBarRedToGreen.Click += new System.EventHandler(this.colorBarRedToGreen_Click);
            // 
            // colorBarRainbow5
            // 
            this.colorBarRainbow5.Colors = null;
            this.colorBarRainbow5.Location = new System.Drawing.Point(23, 66);
            this.colorBarRainbow5.Name = "colorBarRainbow5";
            this.colorBarRainbow5.Size = new System.Drawing.Size(241, 17);
            this.colorBarRainbow5.TabIndex = 3;
            this.colorBarRainbow5.Click += new System.EventHandler(this.colorBarRainbow5_Click);
            // 
            // radioBtnRainbow5
            // 
            this.radioBtnRainbow5.AutoSize = true;
            this.radioBtnRainbow5.Location = new System.Drawing.Point(3, 68);
            this.radioBtnRainbow5.Name = "radioBtnRainbow5";
            this.radioBtnRainbow5.Size = new System.Drawing.Size(14, 13);
            this.radioBtnRainbow5.TabIndex = 2;
            this.radioBtnRainbow5.TabStop = true;
            this.radioBtnRainbow5.Tag = "rainbow5";
            this.radioBtnRainbow5.UseVisualStyleBackColor = true;
            // 
            // colorBarRainbow7
            // 
            this.colorBarRainbow7.Colors = null;
            this.colorBarRainbow7.Location = new System.Drawing.Point(23, 42);
            this.colorBarRainbow7.Name = "colorBarRainbow7";
            this.colorBarRainbow7.Size = new System.Drawing.Size(241, 17);
            this.colorBarRainbow7.TabIndex = 1;
            this.colorBarRainbow7.Click += new System.EventHandler(this.colorBarRainbow7_Click);
            // 
            // radioBtnRainbow7
            // 
            this.radioBtnRainbow7.AutoSize = true;
            this.radioBtnRainbow7.Location = new System.Drawing.Point(3, 44);
            this.radioBtnRainbow7.Name = "radioBtnRainbow7";
            this.radioBtnRainbow7.Size = new System.Drawing.Size(14, 13);
            this.radioBtnRainbow7.TabIndex = 0;
            this.radioBtnRainbow7.TabStop = true;
            this.radioBtnRainbow7.Tag = "rainbow7";
            this.radioBtnRainbow7.UseVisualStyleBackColor = true;
            // 
            // panelUniqueValues
            // 
            this.panelUniqueValues.Controls.Add(this.radioBtnUvPalette3);
            this.panelUniqueValues.Controls.Add(this.radioBtnUvPalette2);
            this.panelUniqueValues.Controls.Add(this.radioBtnUvPalette1);
            this.panelUniqueValues.Controls.Add(this.colorBarUvPalette3);
            this.panelUniqueValues.Controls.Add(this.colorBarUvPalette2);
            this.panelUniqueValues.Controls.Add(this.colorBarUvPalette1);
            this.panelUniqueValues.Controls.Add(this.cboUniqueValuesRenderField);
            this.panelUniqueValues.Controls.Add(this.labelUniqueValuesRenderField);
            this.panelUniqueValues.Location = new System.Drawing.Point(52, 147);
            this.panelUniqueValues.Name = "panelUniqueValues";
            this.panelUniqueValues.Size = new System.Drawing.Size(289, 142);
            this.panelUniqueValues.TabIndex = 10;
            // 
            // radioBtnUvPalette3
            // 
            this.radioBtnUvPalette3.AutoSize = true;
            this.radioBtnUvPalette3.Location = new System.Drawing.Point(3, 98);
            this.radioBtnUvPalette3.Name = "radioBtnUvPalette3";
            this.radioBtnUvPalette3.Size = new System.Drawing.Size(14, 13);
            this.radioBtnUvPalette3.TabIndex = 8;
            this.radioBtnUvPalette3.TabStop = true;
            this.radioBtnUvPalette3.UseVisualStyleBackColor = true;
            // 
            // radioBtnUvPalette2
            // 
            this.radioBtnUvPalette2.AutoSize = true;
            this.radioBtnUvPalette2.Location = new System.Drawing.Point(3, 73);
            this.radioBtnUvPalette2.Name = "radioBtnUvPalette2";
            this.radioBtnUvPalette2.Size = new System.Drawing.Size(14, 13);
            this.radioBtnUvPalette2.TabIndex = 7;
            this.radioBtnUvPalette2.TabStop = true;
            this.radioBtnUvPalette2.UseVisualStyleBackColor = true;
            // 
            // radioBtnUvPalette1
            // 
            this.radioBtnUvPalette1.AutoSize = true;
            this.radioBtnUvPalette1.Location = new System.Drawing.Point(3, 47);
            this.radioBtnUvPalette1.Name = "radioBtnUvPalette1";
            this.radioBtnUvPalette1.Size = new System.Drawing.Size(14, 13);
            this.radioBtnUvPalette1.TabIndex = 6;
            this.radioBtnUvPalette1.TabStop = true;
            this.radioBtnUvPalette1.UseVisualStyleBackColor = true;
            // 
            // colorBarUvPalette3
            // 
            this.colorBarUvPalette3.Colors = null;
            this.colorBarUvPalette3.Location = new System.Drawing.Point(23, 96);
            this.colorBarUvPalette3.Name = "colorBarUvPalette3";
            this.colorBarUvPalette3.Size = new System.Drawing.Size(242, 17);
            this.colorBarUvPalette3.TabIndex = 5;
            this.colorBarUvPalette3.Click += new System.EventHandler(this.colorBarUvPalette3_Click);
            // 
            // colorBarUvPalette2
            // 
            this.colorBarUvPalette2.Colors = null;
            this.colorBarUvPalette2.Location = new System.Drawing.Point(23, 71);
            this.colorBarUvPalette2.Name = "colorBarUvPalette2";
            this.colorBarUvPalette2.Size = new System.Drawing.Size(242, 17);
            this.colorBarUvPalette2.TabIndex = 4;
            this.colorBarUvPalette2.Click += new System.EventHandler(this.colorBarUvPalette2_Click);
            // 
            // colorBarUvPalette1
            // 
            this.colorBarUvPalette1.Colors = null;
            this.colorBarUvPalette1.Location = new System.Drawing.Point(23, 45);
            this.colorBarUvPalette1.Name = "colorBarUvPalette1";
            this.colorBarUvPalette1.Size = new System.Drawing.Size(242, 17);
            this.colorBarUvPalette1.TabIndex = 3;
            this.colorBarUvPalette1.Click += new System.EventHandler(this.colorBarUvPalette1_Click);
            // 
            // cboUniqueValuesRenderField
            // 
            this.cboUniqueValuesRenderField.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboUniqueValuesRenderField.FormattingEnabled = true;
            this.cboUniqueValuesRenderField.Location = new System.Drawing.Point(4, 16);
            this.cboUniqueValuesRenderField.Name = "cboUniqueValuesRenderField";
            this.cboUniqueValuesRenderField.Size = new System.Drawing.Size(261, 21);
            this.cboUniqueValuesRenderField.TabIndex = 2;
            // 
            // labelUniqueValuesRenderField
            // 
            this.labelUniqueValuesRenderField.AutoSize = true;
            this.labelUniqueValuesRenderField.Location = new System.Drawing.Point(4, 0);
            this.labelUniqueValuesRenderField.Name = "labelUniqueValuesRenderField";
            this.labelUniqueValuesRenderField.Size = new System.Drawing.Size(70, 13);
            this.labelUniqueValuesRenderField.TabIndex = 1;
            this.labelUniqueValuesRenderField.Text = "Render Field:";
            // 
            // panelSingleSymbol
            // 
            this.panelSingleSymbol.Controls.Add(this.buttonSelectColor);
            this.panelSingleSymbol.Controls.Add(this.panelSingleSymbolColor);
            this.panelSingleSymbol.Location = new System.Drawing.Point(52, 82);
            this.panelSingleSymbol.Name = "panelSingleSymbol";
            this.panelSingleSymbol.Size = new System.Drawing.Size(289, 36);
            this.panelSingleSymbol.TabIndex = 11;
            // 
            // cboSymbolType
            // 
            this.cboSymbolType.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboSymbolType.FormattingEnabled = true;
            this.cboSymbolType.Location = new System.Drawing.Point(31, 12);
            this.cboSymbolType.Name = "cboSymbolType";
            this.cboSymbolType.Size = new System.Drawing.Size(97, 21);
            this.cboSymbolType.TabIndex = 12;
            // 
            // cboSize
            // 
            this.cboSize.FormattingEnabled = true;
            this.cboSize.Location = new System.Drawing.Point(162, 12);
            this.cboSize.Name = "cboSize";
            this.cboSize.Size = new System.Drawing.Size(45, 21);
            this.cboSize.TabIndex = 14;
            // 
            // TimeseriesSymbologyDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(357, 553);
            this.Controls.Add(this.cboSize);
            this.Controls.Add(this.cboSymbolType);
            this.Controls.Add(this.panelSingleSymbol);
            this.Controls.Add(this.panelUniqueValues);
            this.Controls.Add(this.panelColorRamps);
            this.Controls.Add(this.radioBtnContinuousValues);
            this.Controls.Add(this.radioBtnUniqueValues);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOK);
            this.Controls.Add(this.radioBtnSingleSymbol);
            this.Controls.Add(this.lblSize);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "TimeseriesSymbologyDialog";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Timeseries Map Symbols";
            this.panelColorRamps.ResumeLayout(false);
            this.panelColorRamps.PerformLayout();
            this.panelUniqueValues.ResumeLayout(false);
            this.panelUniqueValues.PerformLayout();
            this.panelSingleSymbol.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblSize;
        private System.Windows.Forms.RadioButton radioBtnSingleSymbol;
        private System.Windows.Forms.Panel panelSingleSymbolColor;
        private System.Windows.Forms.Button buttonSelectColor;
        private System.Windows.Forms.Button buttonOK;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.RadioButton radioBtnUniqueValues;
        private System.Windows.Forms.RadioButton radioBtnContinuousValues;
        private System.Windows.Forms.Panel panelColorRamps;
        private System.Windows.Forms.RadioButton radioBtnRainbow7;
        private System.Windows.Forms.Panel panelUniqueValues;
        private System.Windows.Forms.Panel panelSingleSymbol;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarRainbow7;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarRainbow5;
        private System.Windows.Forms.RadioButton radioBtnRainbow5;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarBlueToGreen;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarRedToBlue;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarRedToGreen;
        private System.Windows.Forms.RadioButton radioBtnBlueToGreen;
        private System.Windows.Forms.RadioButton radioBtnRedToBlue;
        private System.Windows.Forms.RadioButton radioBtnRedToGreen;
        private System.Windows.Forms.ComboBox cboContinuousRenderField;
        private System.Windows.Forms.Label labelContinuousRenderField;
        private System.Windows.Forms.ComboBox cboUniqueValuesRenderField;
        private System.Windows.Forms.Label labelUniqueValuesRenderField;
        private System.Windows.Forms.ComboBox cboSymbolType;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarUvPalette2;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarUvPalette1;
        private USGS.Puma.UI.MapViewer.ColorBar colorBarUvPalette3;
        private System.Windows.Forms.RadioButton radioBtnUvPalette3;
        private System.Windows.Forms.RadioButton radioBtnUvPalette2;
        private System.Windows.Forms.RadioButton radioBtnUvPalette1;
        private System.Windows.Forms.ComboBox cboSize;
    }
}