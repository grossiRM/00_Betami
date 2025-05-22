using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.NTS.Features;

namespace USGS.Puma.UI.MapViewer
{
    public class Renderer
    {
        public Renderer()
        {
            _MaskField = "";
            _MaskValues = new List<double>();
            _IncludeMaskValues = false;
            _UseMask = false;
        }

        private string _MaskField;
        /// <summary>
        /// 
        /// </summary>
        public string MaskField
        {
            get { return _MaskField; }
            set { _MaskField = value; }
        }

        private List<double> _MaskValues;
        /// <summary>
        /// 
        /// </summary>
        public List<double> MaskValues
        {
            get { return _MaskValues; }
        }

        private bool _IncludeMaskValues;
        /// <summary>
        /// 
        /// </summary>
        public bool IncludeMaskValues
        {
            get { return _IncludeMaskValues; }
            set { _IncludeMaskValues = value; }
        }

        private bool _UseMask;
        /// <summary>
        /// 
        /// </summary>
        public bool UseMask
        {
            get { return _UseMask; }
            set { _UseMask = value; }
        }
        /// <summary>
        /// Determine if the attribute list contains a numeric field with name specified by MaskField
        /// </summary>
        /// <param name="attributes"></param>
        /// <returns></returns>
        public bool IsValidMaskField(IAttributesTable attributes)
        {
            if(string.IsNullOrEmpty(MaskField))
            { return false;}

            try
            {
                object a = attributes[MaskField];
                double v = Convert.ToDouble(a);
                return true;
            }
            catch
            {
                return false;
            }
        }


        #region IRenderer Members

        #endregion
    }
}
