using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class MapLegendItem : System.Windows.Forms.Panel
    {
        #region Private Fields
        private System.Windows.Forms.Panel panelExpandOrCollapse;
        private System.Windows.Forms.Panel panelHeader;
        private System.Windows.Forms.Label labelText;
        private System.Windows.Forms.CheckBox checkBoxVisible;
        private FeatureLayer _FeatureLayer;

        #endregion

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public MapLegendItem() : this(null) { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mapLayer"></param>
        public MapLegendItem(GraphicLayer mapLayer)
        {
            InitializeComponent();

            this.LayerVisible = true;
            this.IsExpanded = true;

            if (mapLayer != null)
            {
                this.MapLayer = mapLayer;
            }

            SetHeight();

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
        private void panelExpandOrCollapse_Click(object sender, EventArgs e)
        {
            if (IsExpanded)
            {
                Collapse();
            }
            else
            {
                Expand();
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void checkBoxVisible_CheckedChanged(object sender, EventArgs e)
        {
            if (MapLayer != null)
            {
                MapLayer.Visible = checkBoxVisible.Checked;
                this.Refresh();
                OnLayerVisibilityChanged(e);
            }
        }

        #endregion

        #region Public Properties
        private GraphicLayer _MapLayer;
        /// <summary>
        /// 
        /// </summary>
        public GraphicLayer MapLayer
        {
            get 
            {
                return (_MapLayer as GraphicLayer);
            }
            set 
            {
                if (value == null)
                {
                    _MapLayer = null;
                    _FeatureLayer = null;
                    Symbology = null;
                }
                else if (value is FeatureLayer)
                {
                    _FeatureLayer = value as FeatureLayer;

                    if (_FeatureLayer.Renderer is SingleSymbolRenderer)
                    {
                        Symbology = new MapLegendSingleSymbol(_FeatureLayer.Renderer as SingleSymbolRenderer);
                    }
                    else if (_FeatureLayer.Renderer is NumericValueRenderer)
                    {
                        Symbology = new MapLegendUniqueValues(_FeatureLayer.Renderer as NumericValueRenderer);
                    }
                    else if (_FeatureLayer.Renderer is ColorRampRenderer)
                    {
                        Symbology = new MapLegendContinuousValues(_FeatureLayer.Renderer as ColorRampRenderer, 75);
                    }
                    else
                    {
                        throw new ArgumentException("The specified renderer type is not supported.");
                    }

                    Symbology.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
                                       | System.Windows.Forms.AnchorStyles.Right)));

                    // Initialize Visible and LayerName properties
                    _MapLayer = null;
                    LayerVisible = _FeatureLayer.Visible;
                    labelText.Text = _FeatureLayer.LayerName;
                    this.Height = this.panelHeader.Height + Symbology.Height;
                    _MapLayer = value;

                }
                else
                {
                    throw new ArgumentException("Specified map layer type is not supported.");
                }
            }
        }

        private MapLegendItemData _Symbology;
        /// <summary>
        /// 
        /// </summary>
        public MapLegendItemData Symbology
        {
            get { return _Symbology; }
            private set
            {
                if (_Symbology != null)
                {
                    this.Controls.Remove(_Symbology);
                }
                _Symbology = value;
                if (_Symbology != null)
                {
                    _Symbology.Location = new System.Drawing.Point(45, this.panelHeader.Height);
                    _Symbology.Width = this.Width - 50;
                    _Symbology.Visible = true;
                    this.Controls.Add(_Symbology);
                    this.Controls.Add(this.panelHeader);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool LayerVisible
        {
            get { return checkBoxVisible.Checked; }
            set 
            { 
                checkBoxVisible.Checked = value;

                if (_MapLayer != null)
                {
                    _MapLayer.Visible = checkBoxVisible.Checked;

                    // Add code to send notification event

                }

            }
        }

        private bool _IsExpanded;
        /// <summary>
        /// 
        /// </summary>
        public bool IsExpanded
        {
            get { return _IsExpanded; }
            private set { _IsExpanded = value; }
        }

        #endregion

        #region Public Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="refresh"></param>
        public void Update(bool refresh)
        {
            // Add code
        }
        /// <summary>
        /// 
        /// </summary>
        public void Collapse()
        {
            panelExpandOrCollapse.BackgroundImage = global::USGS.Puma.UI.Properties.Resources.ExpandLegendItem2;
            IsExpanded = false;
            SetHeight();

        }
        /// <summary>
        /// 
        /// </summary>
        public void Expand()
        {
            panelExpandOrCollapse.BackgroundImage = global::USGS.Puma.UI.Properties.Resources.CollapseLegendItem2;
            IsExpanded = true;
            SetHeight();

        }
        #endregion

        #region Private Methods
        /// <summary>
        /// 
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MapLegendItem));
            this.labelText = new System.Windows.Forms.Label();
            this.checkBoxVisible = new System.Windows.Forms.CheckBox();
            this.panelHeader = new System.Windows.Forms.Panel();
            this.panelExpandOrCollapse = new System.Windows.Forms.Panel();
            this.panelHeader.SuspendLayout();
            this.SuspendLayout();
            // 
            // labelText
            // 
            this.labelText.AutoSize = true;
            this.labelText.Location = new System.Drawing.Point(45, 3);
            this.labelText.Name = "labelText";
            this.labelText.Size = new System.Drawing.Size(105, 13);
            this.labelText.TabIndex = 0;
            this.labelText.Text = "Map legend item text";
            // 
            // checkBoxVisible
            // 
            this.checkBoxVisible.AutoSize = true;
            this.checkBoxVisible.Location = new System.Drawing.Point(23, 3);
            this.checkBoxVisible.Name = "checkBoxVisible";
            this.checkBoxVisible.Size = new System.Drawing.Size(15, 14);
            this.checkBoxVisible.TabIndex = 0;
            this.checkBoxVisible.UseVisualStyleBackColor = true;
            this.checkBoxVisible.CheckedChanged += new System.EventHandler(this.checkBoxVisible_CheckedChanged);
            // 
            // panelHeader
            // 
            this.panelHeader.BackColor = System.Drawing.Color.White;
            this.panelHeader.Controls.Add(this.labelText);
            this.panelHeader.Controls.Add(this.checkBoxVisible);
            this.panelHeader.Controls.Add(this.panelExpandOrCollapse);
            this.panelHeader.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelHeader.Location = new System.Drawing.Point(0, 0);
            this.panelHeader.Name = "panelHeader";
            this.panelHeader.Size = new System.Drawing.Size(200, 20);
            this.panelHeader.TabIndex = 0;
            // 
            // panelExpandOrCollapse
            // 
            this.panelExpandOrCollapse.BackColor = System.Drawing.Color.White;
            this.panelExpandOrCollapse.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("panelExpandOrCollapse.BackgroundImage")));
            this.panelExpandOrCollapse.Location = new System.Drawing.Point(3, 3);
            this.panelExpandOrCollapse.Name = "panelExpandOrCollapse";
            this.panelExpandOrCollapse.Size = new System.Drawing.Size(16, 16);
            this.panelExpandOrCollapse.TabIndex = 0;
            this.panelExpandOrCollapse.Click += new System.EventHandler(this.panelExpandOrCollapse_Click);
            // 
            // MapLegendItem
            // 
            this.BackColor = System.Drawing.Color.White;
            this.panelHeader.ResumeLayout(false);
            this.panelHeader.PerformLayout();
            this.ResumeLayout(false);

        }
        /// <summary>
        /// 
        /// </summary>
        private void SetHeight()
        {
            int y = 0;
            if (IsExpanded)
            {
                if (Symbology != null)
                { y = Symbology.Height; }
            }
            this.Height = panelHeader.Height + y;
        }
        #endregion

    }
}
