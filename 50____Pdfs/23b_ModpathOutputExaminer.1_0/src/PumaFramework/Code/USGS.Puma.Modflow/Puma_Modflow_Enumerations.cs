using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    #region Public Enumerations
    public enum ArrayControlRecordType
    {
        Constant = 0,
        Internal = 1,
        External = 2,
        OpenClose = 3
    }
    public enum StressPeriodType
    {
        Undefined = 0,
        SteadyState = 1,
        Transient = 2
    }
    public enum OutputPrecisionType
    {
        Undefined = 0,
        Single = 1,
        Double = 2
    }
    #endregion

}
