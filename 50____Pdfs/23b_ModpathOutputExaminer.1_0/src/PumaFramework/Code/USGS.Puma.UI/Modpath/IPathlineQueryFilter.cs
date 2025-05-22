using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Modpath.IO;

namespace USGS.Puma.UI.Modpath
{
    public interface IPathlineQueryFilter : IQueryFilter
    {
        List<PathlineRecord> Execute();
    }
}
