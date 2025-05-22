using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Modpath.IO;
using USGS.Puma.UI.Modpath;

namespace ModpathOutputExaminer
{
    public class PathlineDataset
    {
        public PathlineDataset() { }

        private PathlineHeader _Header;
        /// <summary>
        /// 
        /// </summary>
        public PathlineHeader Header
        {
            get { return _Header; }
            set { _Header = value; }
        }

        private List<PathlineRecord> _TotalRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<PathlineRecord> TotalRecords
        {
            get { return _TotalRecords; }
            set { _TotalRecords = value; }
        }

        private List<PathlineRecord> _FilteredRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<PathlineRecord> FilteredRecords
        {
            get { return _FilteredRecords; }
            set { _FilteredRecords = value; }
        }

        private PathlineQueryFilter _QueryFilter;
        /// <summary>
        /// 
        /// </summary>
        public PathlineQueryFilter QueryFilter
        {
            get { return _QueryFilter; }
            set { _QueryFilter = value; }
        }

        public void LoadFromFile(string filename)
        {
            using (PathlineFileReader reader = new PathlineFileReader(filename))
            {
                Header = reader.Header;
                TotalRecords = reader.Read();
                FilteredRecords = TotalRecords;
                QueryFilter = new PathlineQueryFilter();
            }

        }

        public void CloseDataset()
        {
            Header = null;
            TotalRecords = null;
            FilteredRecords = null;
            QueryFilter = null;
        }


    }
}
