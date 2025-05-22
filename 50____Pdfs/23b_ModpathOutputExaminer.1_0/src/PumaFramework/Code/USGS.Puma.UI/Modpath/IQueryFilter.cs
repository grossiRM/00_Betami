using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.Modpath
{
    public interface IQueryFilter
    {
        bool FilteringIsOn { get; set; }
        string Summary { get; set; }
        bool ShowFilterDialog();
    }
}
