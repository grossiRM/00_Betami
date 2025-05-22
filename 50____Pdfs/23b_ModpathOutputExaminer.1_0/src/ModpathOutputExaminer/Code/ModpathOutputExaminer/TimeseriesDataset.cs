using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Modpath.IO;
using USGS.Puma.UI.Modpath;

namespace ModpathOutputExaminer
{
    public class TimeseriesDataset
    {
        
        public TimeseriesDataset() { }

        private TimeseriesHeader _Header;
        /// <summary>
        /// 
        /// </summary>
        public TimeseriesHeader Header
        {
            get { return _Header; }
            set { _Header = value; }
        }

        private List<TimeseriesRecord> _TotalRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<TimeseriesRecord> TotalRecords
        {
            get { return _TotalRecords; }
            set { _TotalRecords = value; }
        }

        private List<TimeseriesRecord> _FilteredRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<TimeseriesRecord> FilteredRecords
        {
            get { return _FilteredRecords; }
            set { _FilteredRecords = value; }
        }

        private TimeseriesQueryFilter _QueryFilter;
        /// <summary>
        /// 
        /// </summary>
        public TimeseriesQueryFilter QueryFilter
        {
            get { return _QueryFilter; }
            set { _QueryFilter = value; }
        }

        public void LoadFromFile(string filename)
        {
            using (TimeseriesFileReader reader = new TimeseriesFileReader(filename))
            {
                Header = reader.Header;
                TotalRecords = reader.Read();
                FilteredRecords = TotalRecords;
                QueryFilter = new TimeseriesQueryFilter();
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Dictionary<int,double> GetAvailableTimePoints()
        {
            Dictionary<int, double> list = new Dictionary<int, double>();
            if (FilteredRecords != null)
            {
                for (int i = 0; i < FilteredRecords.Count; i++)
                {
                    if(!list.ContainsKey(FilteredRecords[i].TimePoint))
                    {
                        list.Add(FilteredRecords[i].TimePoint, FilteredRecords[i].Time);
                    }
                }
            }
            return list;
        }

        public void CloseDataset()
        {
            TotalRecords = null;
            Header = null;
            FilteredRecords = null;
            QueryFilter = null;
        }

    }
}
