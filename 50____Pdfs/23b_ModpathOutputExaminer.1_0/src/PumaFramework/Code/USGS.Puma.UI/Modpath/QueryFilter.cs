using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace USGS.Puma.UI.Modpath
{
    public partial class QueryFilter : Form, IQueryFilter
    {
        public QueryFilter()
        {
            InitializeComponent();

            ResetData();

        }

        #region Event Handlers
        private void btnOK_Click(object sender, EventArgs e)
        {
            AcceptDialogData();
        }
        private void btnCancel_Click(object sender, EventArgs e)
        {
            CancelDialogData();
        }
        private void radioBtnFilteringIsOn_CheckedChanged(object sender, EventArgs e)
        {
            gboxQueryDef.Enabled = radioBtnFilteringIsOn.Checked;
        }
        #endregion

        #region Public Methods

        #endregion

        #region IQueryFilter members
        private bool _FilteringIsOn;
        public bool FilteringIsOn
        {
            get
            {
                return _FilteringIsOn;
            }
            set
            {
                _FilteringIsOn = value;
            }
        }
        public string Summary
        {
            get
            {
                return OutputQueryFilterSummary();
            }
            set
            {
                throw new NotImplementedException();
            }
        }
        public virtual bool ShowFilterDialog()
        {
            SetDialogData();
            if (this.ShowDialog() == DialogResult.OK)
            {
                GetDialogData();
                return true;
            }
            else
            { return false; }
        }

        #endregion

        #region Private and Protected Methods
        protected virtual void AcceptDialogData()
        {
            this.DialogResult = DialogResult.OK;
        }
        protected virtual void CancelDialogData()
        {
            this.DialogResult = DialogResult.Cancel;
        }
        protected virtual void GetDialogData()
        {
            FilteringIsOn = radioBtnFilteringIsOn.Checked;
        }
        protected virtual void SetDialogData()
        {
            radioBtnFilteringIsOn.Checked = FilteringIsOn;
            gboxQueryDef.Enabled = FilteringIsOn;
        }
        protected virtual void ResetData()
        {
            FilteringIsOn = true;
        }
        protected virtual string OutputQueryFilterSummary()
        {
            return "";
        }
        #endregion

    }
}
