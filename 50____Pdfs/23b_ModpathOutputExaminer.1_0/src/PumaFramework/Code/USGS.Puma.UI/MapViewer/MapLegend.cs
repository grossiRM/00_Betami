using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegend :System.Windows.Forms.Panel
    {
        #region Private Fields
        private System.Windows.Forms.Label labelLegendTitle;
        private System.Windows.Forms.Panel panelHeader;
        private System.Windows.Forms.Panel panelBody;
        private MapLegendItemCollection _Items;
        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public MapLegend()
        {
            // Initialized component controls
            InitializeComponent();

            _Items = new MapLegendItemCollection();
            this.DoubleBuffered = true;

        }
        #endregion

        #region Events and Event Dispatchers
        /// <summary>
        /// 
        /// </summary>
        public event EventHandler<EventArgs> LayerVisibilityChanged;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="e"></param>
        protected virtual void OnLayerVisibilityChanged(EventArgs e)
        {
            EventHandler<EventArgs> handler = LayerVisibilityChanged;
            if (handler != null)
            {
                LayerVisibilityChanged(this, e);
            }
        }
        #endregion

        #region Event Handlers
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void MapLegend_LayerVisibilityChanged(object sender, EventArgs e)
        {
            OnLayerVisibilityChanged(e);
        }

        #endregion

        #region Private Methods
        /// <summary>
        /// 
        /// </summary>
        private void InitializeComponent()
        {
            this.labelLegendTitle = new System.Windows.Forms.Label();
            this.panelHeader = new System.Windows.Forms.Panel();
            this.panelBody = new System.Windows.Forms.Panel();
            this.panelHeader.SuspendLayout();
            this.SuspendLayout();
            // 
            // labelLegendTitle
            // 
            this.labelLegendTitle.AutoSize = true;
            this.labelLegendTitle.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelLegendTitle.Location = new System.Drawing.Point(0, 0);
            this.labelLegendTitle.Name = "labelLegendTitle";
            this.labelLegendTitle.Padding = new System.Windows.Forms.Padding(0, 3, 0, 0);
            this.labelLegendTitle.Size = new System.Drawing.Size(0, 16);
            this.labelLegendTitle.TabIndex = 0;
            // 
            // panelHeader
            // 
            this.panelHeader.BackColor = System.Drawing.Color.White;
            this.panelHeader.Controls.Add(this.labelLegendTitle);
            this.panelHeader.Location = new System.Drawing.Point(0, 0);
            this.panelHeader.Name = "panelHeader";
            this.panelHeader.Size = new System.Drawing.Size(200, 20);
            this.panelHeader.TabIndex = 0;
            // 
            // panelBody
            // 
            this.panelBody.BackColor = System.Drawing.Color.White;
            this.panelBody.Location = new System.Drawing.Point(0, 20);
            this.panelBody.Name = "panelBody";
            this.panelBody.Size = new System.Drawing.Size(200, 1500);
            this.panelBody.TabIndex = 0;
            // 
            // MapLegend
            // 
            this.Controls.Add(this.panelHeader);
            this.panelHeader.ResumeLayout(false);
            this.panelHeader.PerformLayout();
            this.ResumeLayout(false);

        }
        #endregion

        #region Public Properties
        /// <summary>
        /// 
        /// </summary>
        public string LegendTitle
        {
            get { return labelLegendTitle.Text; }
            set { labelLegendTitle.Text = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        public MapLegendItemCollection Items
        {
            get { return _Items; }
        }
        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="items"></param>
        public void AddItems(Collection<GraphicLayer> items)
        {
            MapLegendItemCollection legendItems = new MapLegendItemCollection();
            if (items != null)
            {
                for (int i = 0; i < items.Count; i++)
                {
                    if (items[i] is FeatureLayer)
                    {
                        MapLegendItem item = new MapLegendItem(items[i]);
                        legendItems.Add(item);
                    }
                }
            }
            AddItems(legendItems);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="items"></param>
        public void AddItems(MapLegendItemCollection items)
        {
            _Items.Clear();
            if (items != null)
            {
                this.SuspendLayout();
                this.panelBody.SuspendLayout();
                this.panelBody.Controls.Clear();

                for (int i = 0; i < items.Count; i++)
                {
                    _Items.Add(items[i]);
                }

                int y = 0;
                for (int i = _Items.Count - 1; i >= 0; i--)
                {
                    _Items[i].Dock = System.Windows.Forms.DockStyle.Top;
                    panelBody.Controls.Add(_Items[i]);
                    _Items[i].LayerVisibilityChanged += new EventHandler<EventArgs>(MapLegend_LayerVisibilityChanged);
                    y += _Items[i].Height;
                }
                panelBody.Height = y;

                this.Controls.Remove(panelBody);
                this.Controls.Add(panelBody);
                this.panelBody.ResumeLayout(false);
                this.panelBody.PerformLayout();
                this.ResumeLayout(false);
                this.PerformLayout();
                
               
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public void Clear()
        {
            this.Clear(false);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="resetTitle"></param>
        public void Clear(bool resetTitle)
        {
            this.Controls.Clear();
            _Items.Clear();
            this.Controls.Add(this.labelLegendTitle);
            if (resetTitle)
            {
                labelLegendTitle.Text = "";
            }
        }
        #endregion
    }
}
