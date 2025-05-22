using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class OcDataHeader
    {
        #region Private Fields
        private List<string> _Comments = new List<string>();
        #endregion

        #region Constructors
        public OcDataHeader()
        {
            _HeadPrintFormat = 0;
            _HeadSaveFormat = 0;
            _HeadSaveFormatLabel = false;
            _HeadSaveUnit = -1;

            _DrawdownPrintFormat = 0;
            _DrawdownSaveFormat = 0;
            _DrawdownSaveFormatLabel = false;
            _DrawdownSaveUnit = -1;

            _IboundSaveFormat = "";
            _IboundSaveFormatLabel = false;
            _IboundSaveUnit = -1;

            _CompactBudget = false;
            _CompactBudgetSaveAuxiliary=false;
        }
        public OcDataHeader(OcDataHeader header) : this()
        {
            if (header != null)
            {
                _HeadPrintFormat = header.HeadPrintFormat;
                _HeadSaveFormat = header.HeadSaveFormat;
                _HeadSaveFormatLabel = header.HeadSaveFormatLabel;
                _HeadSaveUnit = header.HeadSaveUnit;

                _DrawdownPrintFormat = header.DrawdownPrintFormat;
                _DrawdownSaveFormat = header.DrawdownSaveFormat;
                _DrawdownSaveFormatLabel = header.DrawdownSaveFormatLabel;
                _DrawdownSaveUnit = header.DrawdownSaveUnit;

                _IboundSaveFormat = header.IboundSaveFormat;
                _IboundSaveFormatLabel = header.IboundSaveFormatLabel;
                _IboundSaveUnit = header.IboundSaveUnit;

                _CompactBudget = header.CompactBudget;
                _CompactBudgetSaveAuxiliary = header.CompactBudgetSaveAuxiliary;
            }
        }
        #endregion

        #region Public Properties
        public List<string> Comments
        {
            get { return _Comments; }
        }
        private int _HeadPrintFormat;
        public int HeadPrintFormat
        {
            get { return _HeadPrintFormat; }
            set { _HeadPrintFormat = value; }
        }

        private int _HeadSaveFormat;
        public int HeadSaveFormat
        {
            get { return _HeadSaveFormat; }
            set { _HeadSaveFormat = value; }
        }

        private bool _HeadSaveFormatLabel;
        public bool HeadSaveFormatLabel
        {
            get { return _HeadSaveFormatLabel; }
            set { _HeadSaveFormatLabel = value; }
        }

        private int _HeadSaveUnit;
        public int HeadSaveUnit
        {
            get { return _HeadSaveUnit; }
            set { _HeadSaveUnit = value; }
        }

        private int _DrawdownPrintFormat;
        public int DrawdownPrintFormat
        {
            get { return _DrawdownPrintFormat; }
            set { _DrawdownPrintFormat = value; }
        }

        private int _DrawdownSaveFormat;
        public int DrawdownSaveFormat
        {
            get { return _DrawdownSaveFormat; }
            set { _DrawdownSaveFormat = value; }
        }

        private bool _DrawdownSaveFormatLabel;
        public bool DrawdownSaveFormatLabel
        {
            get { return _DrawdownSaveFormatLabel; }
            set { _DrawdownSaveFormatLabel = value; }
        }

        private int _DrawdownSaveUnit;
        public int DrawdownSaveUnit
        {
            get { return _DrawdownSaveUnit; }
            set { _DrawdownSaveUnit = value; }
        }

        private string _IboundSaveFormat;
        public string IboundSaveFormat
        {
            get { return _IboundSaveFormat; }
            set { _IboundSaveFormat = value; }
        }

        private bool _IboundSaveFormatLabel;
        public bool IboundSaveFormatLabel
        {
            get { return _IboundSaveFormatLabel; }
            set { _IboundSaveFormatLabel = value; }
        }

        private int _IboundSaveUnit;
        public int IboundSaveUnit
        {
            get { return _IboundSaveUnit; }
            set { _IboundSaveUnit = value; }
        }

        private bool _CompactBudget;
        public bool CompactBudget
        {
            get { return _CompactBudget; }
            set { _CompactBudget = value; }
        }

        private bool _CompactBudgetSaveAuxiliary;
        public bool CompactBudgetSaveAuxiliary
        {
            get { return _CompactBudgetSaveAuxiliary; }
            set { _CompactBudgetSaveAuxiliary = value; }
        }
        #endregion

    }
}
