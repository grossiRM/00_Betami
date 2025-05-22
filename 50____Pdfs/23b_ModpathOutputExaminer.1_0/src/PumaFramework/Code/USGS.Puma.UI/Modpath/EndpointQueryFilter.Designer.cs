namespace USGS.Puma.UI.Modpath
{
    partial class EndpointQueryFilter
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
            this.panelInitialCell = new System.Windows.Forms.Panel();
            this.lblInitialColumn = new System.Windows.Forms.Label();
            this.lblInitialRow = new System.Windows.Forms.Label();
            this.lblInitialLayer = new System.Windows.Forms.Label();
            this.txtInitialColumn = new System.Windows.Forms.TextBox();
            this.txtInitialRow = new System.Windows.Forms.TextBox();
            this.txtInitialLayer = new System.Windows.Forms.TextBox();
            this.panelFinalCell = new System.Windows.Forms.Panel();
            this.lblFinalColumn = new System.Windows.Forms.Label();
            this.lblFinalRow = new System.Windows.Forms.Label();
            this.lblFinalLayer = new System.Windows.Forms.Label();
            this.txtFinalColumn = new System.Windows.Forms.TextBox();
            this.txtFinalRow = new System.Windows.Forms.TextBox();
            this.txtFinalLayer = new System.Windows.Forms.TextBox();
            this.panelInitialZone = new System.Windows.Forms.Panel();
            this.lblInitialZoneValue = new System.Windows.Forms.Label();
            this.txtInitialZone = new System.Windows.Forms.TextBox();
            this.panelFinalZone = new System.Windows.Forms.Panel();
            this.lblFinalZoneValue = new System.Windows.Forms.Label();
            this.txtFinalZone = new System.Windows.Forms.TextBox();
            this.panelTravelTime = new System.Windows.Forms.Panel();
            this.lblTravelTimeValue = new System.Windows.Forms.Label();
            this.txtTravelTime = new System.Windows.Forms.TextBox();
            this.cboInitialCell = new System.Windows.Forms.ComboBox();
            this.cboFinalCell = new System.Windows.Forms.ComboBox();
            this.cboInitialZone = new System.Windows.Forms.ComboBox();
            this.cboFinalZone = new System.Windows.Forms.ComboBox();
            this.cboTimeFilter = new System.Windows.Forms.ComboBox();
            this.lblTravelTime = new System.Windows.Forms.Label();
            this.lblFinalZone = new System.Windows.Forms.Label();
            this.lblInitialZone = new System.Windows.Forms.Label();
            this.lblFinalCell = new System.Windows.Forms.Label();
            this.lblInitialCell = new System.Windows.Forms.Label();
            this.lblInitialTime = new System.Windows.Forms.Label();
            this.cboInitialTime = new System.Windows.Forms.ComboBox();
            this.panelInitialTime = new System.Windows.Forms.Panel();
            this.cboInitialTimeValue = new System.Windows.Forms.ComboBox();
            this.lblInitialTimeValue = new System.Windows.Forms.Label();
            this.panelGroup = new System.Windows.Forms.Panel();
            this.txtGroup = new System.Windows.Forms.TextBox();
            this.lblGroupValue = new System.Windows.Forms.Label();
            this.cboGroup = new System.Windows.Forms.ComboBox();
            this.lblGroup = new System.Windows.Forms.Label();
            this.gboxQueryDef.SuspendLayout();
            this.panelInitialCell.SuspendLayout();
            this.panelFinalCell.SuspendLayout();
            this.panelInitialZone.SuspendLayout();
            this.panelFinalZone.SuspendLayout();
            this.panelTravelTime.SuspendLayout();
            this.panelInitialTime.SuspendLayout();
            this.panelGroup.SuspendLayout();
            this.SuspendLayout();
            // 
            // radioBtnFilteringIsOff
            // 
            this.radioBtnFilteringIsOff.Size = new System.Drawing.Size(147, 17);
            this.radioBtnFilteringIsOff.Text = "Show all endpoint records";
            // 
            // radioBtnFilteringIsOn
            // 
            this.radioBtnFilteringIsOn.Size = new System.Drawing.Size(331, 17);
            this.radioBtnFilteringIsOn.Text = "Show only those endpoint records that meet the following criteria:";
            // 
            // gboxQueryDef
            // 
            this.gboxQueryDef.Controls.Add(this.lblGroup);
            this.gboxQueryDef.Controls.Add(this.cboGroup);
            this.gboxQueryDef.Controls.Add(this.panelGroup);
            this.gboxQueryDef.Controls.Add(this.panelInitialTime);
            this.gboxQueryDef.Controls.Add(this.cboInitialTime);
            this.gboxQueryDef.Controls.Add(this.lblInitialTime);
            this.gboxQueryDef.Controls.Add(this.lblInitialCell);
            this.gboxQueryDef.Controls.Add(this.lblFinalCell);
            this.gboxQueryDef.Controls.Add(this.lblInitialZone);
            this.gboxQueryDef.Controls.Add(this.lblFinalZone);
            this.gboxQueryDef.Controls.Add(this.lblTravelTime);
            this.gboxQueryDef.Controls.Add(this.cboTimeFilter);
            this.gboxQueryDef.Controls.Add(this.cboFinalZone);
            this.gboxQueryDef.Controls.Add(this.cboInitialZone);
            this.gboxQueryDef.Controls.Add(this.cboFinalCell);
            this.gboxQueryDef.Controls.Add(this.cboInitialCell);
            this.gboxQueryDef.Controls.Add(this.panelTravelTime);
            this.gboxQueryDef.Controls.Add(this.panelFinalZone);
            this.gboxQueryDef.Controls.Add(this.panelInitialZone);
            this.gboxQueryDef.Controls.Add(this.panelFinalCell);
            this.gboxQueryDef.Controls.Add(this.panelInitialCell);
            this.gboxQueryDef.Size = new System.Drawing.Size(357, 373);
            // 
            // btnOK
            // 
            this.btnOK.Location = new System.Drawing.Point(235, 437);
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(316, 437);
            // 
            // panelInitialCell
            // 
            this.panelInitialCell.Controls.Add(this.lblInitialColumn);
            this.panelInitialCell.Controls.Add(this.lblInitialRow);
            this.panelInitialCell.Controls.Add(this.lblInitialLayer);
            this.panelInitialCell.Controls.Add(this.txtInitialColumn);
            this.panelInitialCell.Controls.Add(this.txtInitialRow);
            this.panelInitialCell.Controls.Add(this.txtInitialLayer);
            this.panelInitialCell.Location = new System.Drawing.Point(203, 16);
            this.panelInitialCell.Name = "panelInitialCell";
            this.panelInitialCell.Size = new System.Drawing.Size(144, 44);
            this.panelInitialCell.TabIndex = 0;
            // 
            // lblInitialColumn
            // 
            this.lblInitialColumn.AutoSize = true;
            this.lblInitialColumn.Location = new System.Drawing.Point(91, 7);
            this.lblInitialColumn.Name = "lblInitialColumn";
            this.lblInitialColumn.Size = new System.Drawing.Size(42, 13);
            this.lblInitialColumn.TabIndex = 5;
            this.lblInitialColumn.Text = "Column";
            // 
            // lblInitialRow
            // 
            this.lblInitialRow.AutoSize = true;
            this.lblInitialRow.Location = new System.Drawing.Point(44, 7);
            this.lblInitialRow.Name = "lblInitialRow";
            this.lblInitialRow.Size = new System.Drawing.Size(29, 13);
            this.lblInitialRow.TabIndex = 4;
            this.lblInitialRow.Text = "Row";
            // 
            // lblInitialLayer
            // 
            this.lblInitialLayer.AutoSize = true;
            this.lblInitialLayer.Location = new System.Drawing.Point(0, 7);
            this.lblInitialLayer.Name = "lblInitialLayer";
            this.lblInitialLayer.Size = new System.Drawing.Size(33, 13);
            this.lblInitialLayer.TabIndex = 3;
            this.lblInitialLayer.Text = "Layer";
            // 
            // txtInitialColumn
            // 
            this.txtInitialColumn.Location = new System.Drawing.Point(94, 23);
            this.txtInitialColumn.Name = "txtInitialColumn";
            this.txtInitialColumn.Size = new System.Drawing.Size(41, 20);
            this.txtInitialColumn.TabIndex = 2;
            // 
            // txtInitialRow
            // 
            this.txtInitialRow.Location = new System.Drawing.Point(47, 23);
            this.txtInitialRow.Name = "txtInitialRow";
            this.txtInitialRow.Size = new System.Drawing.Size(41, 20);
            this.txtInitialRow.TabIndex = 1;
            // 
            // txtInitialLayer
            // 
            this.txtInitialLayer.Location = new System.Drawing.Point(0, 23);
            this.txtInitialLayer.Name = "txtInitialLayer";
            this.txtInitialLayer.Size = new System.Drawing.Size(41, 20);
            this.txtInitialLayer.TabIndex = 0;
            // 
            // panelFinalCell
            // 
            this.panelFinalCell.Controls.Add(this.lblFinalColumn);
            this.panelFinalCell.Controls.Add(this.lblFinalRow);
            this.panelFinalCell.Controls.Add(this.lblFinalLayer);
            this.panelFinalCell.Controls.Add(this.txtFinalColumn);
            this.panelFinalCell.Controls.Add(this.txtFinalRow);
            this.panelFinalCell.Controls.Add(this.txtFinalLayer);
            this.panelFinalCell.Location = new System.Drawing.Point(203, 66);
            this.panelFinalCell.Name = "panelFinalCell";
            this.panelFinalCell.Size = new System.Drawing.Size(144, 45);
            this.panelFinalCell.TabIndex = 1;
            // 
            // lblFinalColumn
            // 
            this.lblFinalColumn.AutoSize = true;
            this.lblFinalColumn.Location = new System.Drawing.Point(91, 8);
            this.lblFinalColumn.Name = "lblFinalColumn";
            this.lblFinalColumn.Size = new System.Drawing.Size(42, 13);
            this.lblFinalColumn.TabIndex = 5;
            this.lblFinalColumn.Text = "Column";
            // 
            // lblFinalRow
            // 
            this.lblFinalRow.AutoSize = true;
            this.lblFinalRow.Location = new System.Drawing.Point(44, 8);
            this.lblFinalRow.Name = "lblFinalRow";
            this.lblFinalRow.Size = new System.Drawing.Size(29, 13);
            this.lblFinalRow.TabIndex = 4;
            this.lblFinalRow.Text = "Row";
            // 
            // lblFinalLayer
            // 
            this.lblFinalLayer.AutoSize = true;
            this.lblFinalLayer.Location = new System.Drawing.Point(0, 8);
            this.lblFinalLayer.Name = "lblFinalLayer";
            this.lblFinalLayer.Size = new System.Drawing.Size(33, 13);
            this.lblFinalLayer.TabIndex = 3;
            this.lblFinalLayer.Text = "Layer";
            // 
            // txtFinalColumn
            // 
            this.txtFinalColumn.Location = new System.Drawing.Point(94, 24);
            this.txtFinalColumn.Name = "txtFinalColumn";
            this.txtFinalColumn.Size = new System.Drawing.Size(41, 20);
            this.txtFinalColumn.TabIndex = 2;
            // 
            // txtFinalRow
            // 
            this.txtFinalRow.Location = new System.Drawing.Point(47, 24);
            this.txtFinalRow.Name = "txtFinalRow";
            this.txtFinalRow.Size = new System.Drawing.Size(41, 20);
            this.txtFinalRow.TabIndex = 1;
            // 
            // txtFinalLayer
            // 
            this.txtFinalLayer.Location = new System.Drawing.Point(0, 24);
            this.txtFinalLayer.Name = "txtFinalLayer";
            this.txtFinalLayer.Size = new System.Drawing.Size(41, 20);
            this.txtFinalLayer.TabIndex = 0;
            // 
            // panelInitialZone
            // 
            this.panelInitialZone.Controls.Add(this.lblInitialZoneValue);
            this.panelInitialZone.Controls.Add(this.txtInitialZone);
            this.panelInitialZone.Location = new System.Drawing.Point(203, 117);
            this.panelInitialZone.Name = "panelInitialZone";
            this.panelInitialZone.Size = new System.Drawing.Size(144, 44);
            this.panelInitialZone.TabIndex = 2;
            // 
            // lblInitialZoneValue
            // 
            this.lblInitialZoneValue.AutoSize = true;
            this.lblInitialZoneValue.Location = new System.Drawing.Point(0, 7);
            this.lblInitialZoneValue.Name = "lblInitialZoneValue";
            this.lblInitialZoneValue.Size = new System.Drawing.Size(32, 13);
            this.lblInitialZoneValue.TabIndex = 1;
            this.lblInitialZoneValue.Text = "Zone";
            // 
            // txtInitialZone
            // 
            this.txtInitialZone.Location = new System.Drawing.Point(0, 23);
            this.txtInitialZone.Name = "txtInitialZone";
            this.txtInitialZone.Size = new System.Drawing.Size(41, 20);
            this.txtInitialZone.TabIndex = 0;
            // 
            // panelFinalZone
            // 
            this.panelFinalZone.Controls.Add(this.lblFinalZoneValue);
            this.panelFinalZone.Controls.Add(this.txtFinalZone);
            this.panelFinalZone.Location = new System.Drawing.Point(203, 167);
            this.panelFinalZone.Name = "panelFinalZone";
            this.panelFinalZone.Size = new System.Drawing.Size(144, 44);
            this.panelFinalZone.TabIndex = 3;
            // 
            // lblFinalZoneValue
            // 
            this.lblFinalZoneValue.AutoSize = true;
            this.lblFinalZoneValue.Location = new System.Drawing.Point(0, 7);
            this.lblFinalZoneValue.Name = "lblFinalZoneValue";
            this.lblFinalZoneValue.Size = new System.Drawing.Size(32, 13);
            this.lblFinalZoneValue.TabIndex = 1;
            this.lblFinalZoneValue.Text = "Zone";
            // 
            // txtFinalZone
            // 
            this.txtFinalZone.Location = new System.Drawing.Point(0, 23);
            this.txtFinalZone.Name = "txtFinalZone";
            this.txtFinalZone.Size = new System.Drawing.Size(41, 20);
            this.txtFinalZone.TabIndex = 0;
            // 
            // panelTravelTime
            // 
            this.panelTravelTime.Controls.Add(this.lblTravelTimeValue);
            this.panelTravelTime.Controls.Add(this.txtTravelTime);
            this.panelTravelTime.Location = new System.Drawing.Point(203, 217);
            this.panelTravelTime.Name = "panelTravelTime";
            this.panelTravelTime.Size = new System.Drawing.Size(144, 44);
            this.panelTravelTime.TabIndex = 4;
            // 
            // lblTravelTimeValue
            // 
            this.lblTravelTimeValue.AutoSize = true;
            this.lblTravelTimeValue.Location = new System.Drawing.Point(0, 7);
            this.lblTravelTimeValue.Name = "lblTravelTimeValue";
            this.lblTravelTimeValue.Size = new System.Drawing.Size(59, 13);
            this.lblTravelTimeValue.TabIndex = 1;
            this.lblTravelTimeValue.Text = "Travel time";
            // 
            // txtTravelTime
            // 
            this.txtTravelTime.Location = new System.Drawing.Point(0, 23);
            this.txtTravelTime.Name = "txtTravelTime";
            this.txtTravelTime.Size = new System.Drawing.Size(88, 20);
            this.txtTravelTime.TabIndex = 0;
            // 
            // cboInitialCell
            // 
            this.cboInitialCell.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboInitialCell.FormattingEnabled = true;
            this.cboInitialCell.Location = new System.Drawing.Point(10, 39);
            this.cboInitialCell.Name = "cboInitialCell";
            this.cboInitialCell.Size = new System.Drawing.Size(188, 21);
            this.cboInitialCell.TabIndex = 5;
            this.cboInitialCell.SelectedIndexChanged += new System.EventHandler(this.cboInitialCell_SelectedIndexChanged);
            // 
            // cboFinalCell
            // 
            this.cboFinalCell.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboFinalCell.FormattingEnabled = true;
            this.cboFinalCell.Location = new System.Drawing.Point(10, 90);
            this.cboFinalCell.Name = "cboFinalCell";
            this.cboFinalCell.Size = new System.Drawing.Size(188, 21);
            this.cboFinalCell.TabIndex = 6;
            this.cboFinalCell.SelectedIndexChanged += new System.EventHandler(this.cboFinalCell_SelectedIndexChanged);
            // 
            // cboInitialZone
            // 
            this.cboInitialZone.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboInitialZone.FormattingEnabled = true;
            this.cboInitialZone.Location = new System.Drawing.Point(10, 140);
            this.cboInitialZone.Name = "cboInitialZone";
            this.cboInitialZone.Size = new System.Drawing.Size(188, 21);
            this.cboInitialZone.TabIndex = 7;
            this.cboInitialZone.SelectedIndexChanged += new System.EventHandler(this.cboInitialZone_SelectedIndexChanged);
            // 
            // cboFinalZone
            // 
            this.cboFinalZone.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboFinalZone.FormattingEnabled = true;
            this.cboFinalZone.Location = new System.Drawing.Point(10, 190);
            this.cboFinalZone.Name = "cboFinalZone";
            this.cboFinalZone.Size = new System.Drawing.Size(188, 21);
            this.cboFinalZone.TabIndex = 8;
            this.cboFinalZone.SelectedIndexChanged += new System.EventHandler(this.cboFinalZone_SelectedIndexChanged);
            // 
            // cboTimeFilter
            // 
            this.cboTimeFilter.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboTimeFilter.FormattingEnabled = true;
            this.cboTimeFilter.Location = new System.Drawing.Point(10, 240);
            this.cboTimeFilter.Name = "cboTimeFilter";
            this.cboTimeFilter.Size = new System.Drawing.Size(188, 21);
            this.cboTimeFilter.TabIndex = 9;
            this.cboTimeFilter.SelectedIndexChanged += new System.EventHandler(this.cboTimeFilter_SelectedIndexChanged);
            // 
            // lblTravelTime
            // 
            this.lblTravelTime.AutoSize = true;
            this.lblTravelTime.Location = new System.Drawing.Point(7, 224);
            this.lblTravelTime.Name = "lblTravelTime";
            this.lblTravelTime.Size = new System.Drawing.Size(106, 13);
            this.lblTravelTime.TabIndex = 10;
            this.lblTravelTime.Text = "Particle travel time is:";
            // 
            // lblFinalZone
            // 
            this.lblFinalZone.AutoSize = true;
            this.lblFinalZone.Location = new System.Drawing.Point(6, 174);
            this.lblFinalZone.Name = "lblFinalZone";
            this.lblFinalZone.Size = new System.Drawing.Size(107, 13);
            this.lblFinalZone.TabIndex = 11;
            this.lblFinalZone.Text = "Particle terminates in:";
            // 
            // lblInitialZone
            // 
            this.lblInitialZone.AutoSize = true;
            this.lblInitialZone.Location = new System.Drawing.Point(7, 124);
            this.lblInitialZone.Name = "lblInitialZone";
            this.lblInitialZone.Size = new System.Drawing.Size(84, 13);
            this.lblInitialZone.TabIndex = 12;
            this.lblInitialZone.Text = "Particle starts in:";
            // 
            // lblFinalCell
            // 
            this.lblFinalCell.AutoSize = true;
            this.lblFinalCell.Location = new System.Drawing.Point(6, 74);
            this.lblFinalCell.Name = "lblFinalCell";
            this.lblFinalCell.Size = new System.Drawing.Size(107, 13);
            this.lblFinalCell.TabIndex = 13;
            this.lblFinalCell.Text = "Particle terminates in:";
            // 
            // lblInitialCell
            // 
            this.lblInitialCell.AutoSize = true;
            this.lblInitialCell.Location = new System.Drawing.Point(6, 23);
            this.lblInitialCell.Name = "lblInitialCell";
            this.lblInitialCell.Size = new System.Drawing.Size(84, 13);
            this.lblInitialCell.TabIndex = 14;
            this.lblInitialCell.Text = "Particle starts in:";
            // 
            // lblInitialTime
            // 
            this.lblInitialTime.AutoSize = true;
            this.lblInitialTime.Location = new System.Drawing.Point(7, 273);
            this.lblInitialTime.Name = "lblInitialTime";
            this.lblInitialTime.Size = new System.Drawing.Size(103, 13);
            this.lblInitialTime.TabIndex = 15;
            this.lblInitialTime.Text = "Particle initial time is:";
            // 
            // cboInitialTime
            // 
            this.cboInitialTime.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboInitialTime.FormattingEnabled = true;
            this.cboInitialTime.Location = new System.Drawing.Point(10, 289);
            this.cboInitialTime.Name = "cboInitialTime";
            this.cboInitialTime.Size = new System.Drawing.Size(188, 21);
            this.cboInitialTime.TabIndex = 16;
            this.cboInitialTime.SelectedIndexChanged += new System.EventHandler(this.cboInitialTime_SelectedIndexChanged);
            // 
            // panelInitialTime
            // 
            this.panelInitialTime.Controls.Add(this.cboInitialTimeValue);
            this.panelInitialTime.Controls.Add(this.lblInitialTimeValue);
            this.panelInitialTime.Location = new System.Drawing.Point(203, 267);
            this.panelInitialTime.Name = "panelInitialTime";
            this.panelInitialTime.Size = new System.Drawing.Size(144, 44);
            this.panelInitialTime.TabIndex = 17;
            // 
            // cboInitialTimeValue
            // 
            this.cboInitialTimeValue.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboInitialTimeValue.FormattingEnabled = true;
            this.cboInitialTimeValue.Location = new System.Drawing.Point(1, 22);
            this.cboInitialTimeValue.Name = "cboInitialTimeValue";
            this.cboInitialTimeValue.Size = new System.Drawing.Size(105, 21);
            this.cboInitialTimeValue.TabIndex = 2;
            // 
            // lblInitialTimeValue
            // 
            this.lblInitialTimeValue.AutoSize = true;
            this.lblInitialTimeValue.Location = new System.Drawing.Point(0, 6);
            this.lblInitialTimeValue.Name = "lblInitialTimeValue";
            this.lblInitialTimeValue.Size = new System.Drawing.Size(53, 13);
            this.lblInitialTimeValue.TabIndex = 1;
            this.lblInitialTimeValue.Text = "Initial time";
            // 
            // panelGroup
            // 
            this.panelGroup.Controls.Add(this.txtGroup);
            this.panelGroup.Controls.Add(this.lblGroupValue);
            this.panelGroup.Location = new System.Drawing.Point(203, 317);
            this.panelGroup.Name = "panelGroup";
            this.panelGroup.Size = new System.Drawing.Size(144, 44);
            this.panelGroup.TabIndex = 18;
            // 
            // txtGroup
            // 
            this.txtGroup.Location = new System.Drawing.Point(0, 23);
            this.txtGroup.Name = "txtGroup";
            this.txtGroup.Size = new System.Drawing.Size(41, 20);
            this.txtGroup.TabIndex = 1;
            // 
            // lblGroupValue
            // 
            this.lblGroupValue.AutoSize = true;
            this.lblGroupValue.Location = new System.Drawing.Point(0, 6);
            this.lblGroupValue.Name = "lblGroupValue";
            this.lblGroupValue.Size = new System.Drawing.Size(36, 13);
            this.lblGroupValue.TabIndex = 0;
            this.lblGroupValue.Text = "Group";
            // 
            // cboGroup
            // 
            this.cboGroup.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboGroup.FormattingEnabled = true;
            this.cboGroup.Location = new System.Drawing.Point(10, 339);
            this.cboGroup.Name = "cboGroup";
            this.cboGroup.Size = new System.Drawing.Size(188, 21);
            this.cboGroup.TabIndex = 19;
            this.cboGroup.SelectedIndexChanged += new System.EventHandler(this.cboGroup_SelectedIndexChanged);
            // 
            // lblGroup
            // 
            this.lblGroup.AutoSize = true;
            this.lblGroup.Location = new System.Drawing.Point(7, 323);
            this.lblGroup.Name = "lblGroup";
            this.lblGroup.Size = new System.Drawing.Size(85, 13);
            this.lblGroup.TabIndex = 20;
            this.lblGroup.Text = "Particle group is:";
            // 
            // EndpointQueryFilter
            // 
            this.ClientSize = new System.Drawing.Size(403, 467);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "EndpointQueryFilter";
            this.Text = "Endpoint Query Filter";
            this.gboxQueryDef.ResumeLayout(false);
            this.gboxQueryDef.PerformLayout();
            this.panelInitialCell.ResumeLayout(false);
            this.panelInitialCell.PerformLayout();
            this.panelFinalCell.ResumeLayout(false);
            this.panelFinalCell.PerformLayout();
            this.panelInitialZone.ResumeLayout(false);
            this.panelInitialZone.PerformLayout();
            this.panelFinalZone.ResumeLayout(false);
            this.panelFinalZone.PerformLayout();
            this.panelTravelTime.ResumeLayout(false);
            this.panelTravelTime.PerformLayout();
            this.panelInitialTime.ResumeLayout(false);
            this.panelInitialTime.PerformLayout();
            this.panelGroup.ResumeLayout(false);
            this.panelGroup.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Panel panelInitialCell;
        private System.Windows.Forms.Panel panelTravelTime;
        private System.Windows.Forms.Panel panelFinalZone;
        private System.Windows.Forms.Panel panelInitialZone;
        private System.Windows.Forms.Panel panelFinalCell;
        private System.Windows.Forms.ComboBox cboFinalCell;
        private System.Windows.Forms.ComboBox cboInitialCell;
        private System.Windows.Forms.ComboBox cboInitialZone;
        private System.Windows.Forms.ComboBox cboFinalZone;
        private System.Windows.Forms.ComboBox cboTimeFilter;
        private System.Windows.Forms.Label lblTravelTime;
        private System.Windows.Forms.Label lblFinalZone;
        private System.Windows.Forms.Label lblInitialZone;
        private System.Windows.Forms.Label lblFinalCell;
        private System.Windows.Forms.Label lblInitialCell;
        private System.Windows.Forms.TextBox txtInitialColumn;
        private System.Windows.Forms.TextBox txtInitialRow;
        private System.Windows.Forms.TextBox txtInitialLayer;
        private System.Windows.Forms.Label lblInitialRow;
        private System.Windows.Forms.Label lblInitialLayer;
        private System.Windows.Forms.Label lblInitialColumn;
        private System.Windows.Forms.TextBox txtFinalLayer;
        private System.Windows.Forms.TextBox txtFinalRow;
        private System.Windows.Forms.TextBox txtFinalColumn;
        private System.Windows.Forms.Label lblFinalRow;
        private System.Windows.Forms.Label lblFinalLayer;
        private System.Windows.Forms.Label lblFinalColumn;
        private System.Windows.Forms.TextBox txtTravelTime;
        private System.Windows.Forms.Label lblFinalZoneValue;
        private System.Windows.Forms.TextBox txtFinalZone;
        private System.Windows.Forms.Label lblInitialZoneValue;
        private System.Windows.Forms.TextBox txtInitialZone;
        private System.Windows.Forms.Label lblTravelTimeValue;
        private System.Windows.Forms.Panel panelInitialTime;
        private System.Windows.Forms.Label lblInitialTimeValue;
        private System.Windows.Forms.ComboBox cboInitialTime;
        private System.Windows.Forms.Label lblInitialTime;
        private System.Windows.Forms.ComboBox cboInitialTimeValue;
        private System.Windows.Forms.Panel panelGroup;
        private System.Windows.Forms.TextBox txtGroup;
        private System.Windows.Forms.Label lblGroupValue;
        private System.Windows.Forms.ComboBox cboGroup;
        private System.Windows.Forms.Label lblGroup;
    }
}
