namespace USGS.Puma.UI.Modpath
{
    partial class ParticleRecordTextView
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.rtxRecords = new System.Windows.Forms.RichTextBox();
            this.SuspendLayout();
            // 
            // rtxRecords
            // 
            this.rtxRecords.BackColor = System.Drawing.SystemColors.Window;
            this.rtxRecords.Dock = System.Windows.Forms.DockStyle.Fill;
            this.rtxRecords.Font = new System.Drawing.Font("Courier New", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtxRecords.Location = new System.Drawing.Point(0, 0);
            this.rtxRecords.Name = "rtxRecords";
            this.rtxRecords.ReadOnly = true;
            this.rtxRecords.Size = new System.Drawing.Size(693, 485);
            this.rtxRecords.TabIndex = 0;
            this.rtxRecords.Text = "";
            this.rtxRecords.WordWrap = false;
            this.rtxRecords.SelectionChanged += new System.EventHandler(this.rtxRecords_SelectionChanged);
            // 
            // ParticleRecordTextView
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.rtxRecords);
            this.Name = "ParticleRecordTextView";
            this.Size = new System.Drawing.Size(693, 485);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.RichTextBox rtxRecords;



    }
}
