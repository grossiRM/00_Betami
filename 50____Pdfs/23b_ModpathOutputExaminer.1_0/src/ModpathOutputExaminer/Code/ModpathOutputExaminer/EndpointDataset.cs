using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Modpath.IO;
using USGS.Puma.UI.Modpath;

namespace ModpathOutputExaminer
{
    public class EndpointDataset
    {
        public EndpointDataset() { }

        private EndpointHeader _Header;
        /// <summary>
        /// 
        /// </summary>
        public EndpointHeader Header
        {
            get { return _Header; }
            set { _Header = value; }
        }

        private List<EndpointRecord> _TotalRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<EndpointRecord> TotalRecords
        {
            get { return _TotalRecords; }
            set { _TotalRecords = value; }
        }

        private List<EndpointRecord> _FilteredRecords;
        /// <summary>
        /// 
        /// </summary>
        public List<EndpointRecord> FilteredRecords
        {
            get { return _FilteredRecords; }
            set { _FilteredRecords = value; }
        }

        private EndpointQueryFilter _QueryFilter;
        /// <summary>
        /// 
        /// </summary>
        public EndpointQueryFilter QueryFilter
        {
            get { return _QueryFilter; }
            set { _QueryFilter = value; }
        }

        public void LoadFromFile(string filename)
        {
            using (EndpointFileReader reader = new EndpointFileReader(filename))
            {
                Header = reader.Header;
                TotalRecords = reader.Read();
                FilteredRecords = TotalRecords;
                QueryFilter = new EndpointQueryFilter();
            }

        }

        public void CloseDataset()
        {
            TotalRecords = null;
            FilteredRecords = null;
            Header = null;
            QueryFilter = null;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public List<int> GetAvailableFinalZones()
        {
            List<int> list = new List<int>();
            if (FilteredRecords != null)
            {
                for (int i = 0; i < FilteredRecords.Count; i++)
                {
                    if (!list.Contains(FilteredRecords[i].FinalZone))
                    {
                        list.Add(FilteredRecords[i].FinalZone);
                    }
                }
            }
            return list;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public List<int> GetAvailableInitialZones()
        {
            List<int> list = new List<int>();
            if (FilteredRecords != null)
            {
                for (int i = 0; i < FilteredRecords.Count; i++)
                {
                    if (!list.Contains(FilteredRecords[i].InitialZone))
                    {
                        list.Add(FilteredRecords[i].InitialZone);
                    }
                }
            }
            return list;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public List<int> GetAvailableGroups()
        {
            List<int> list = new List<int>();
            if (FilteredRecords != null)
            {
                for (int i = 0; i < FilteredRecords.Count; i++)
                {
                    if (!list.Contains(FilteredRecords[i].Group))
                    {
                        list.Add(FilteredRecords[i].Group);
                    }
                }
            }
            return list;
        }

    }
}
