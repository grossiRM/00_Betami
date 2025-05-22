using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class ModflowNameData
    {
        #region Private Fields
        private Dictionary<int, NameFileItem> _NfItems = new Dictionary<int, NameFileItem>();
        #endregion

        #region Public Properties
        private string _ParentDirectory = "";
        public string ParentDirectory
        {
            get { return _ParentDirectory; }
            set { _ParentDirectory = value; }
        }

        #endregion

        #region Public Methods
        public void AddItem(int fileUnit, string fileType, string fileName)
        {
            NameFileItem item = new NameFileItem(fileType, fileUnit, fileName);
            _NfItems.Add(fileUnit, item);
        }
        public void AddItem(int fileUnit, string fileType, string fileName, string fileStatus)
        {
            NameFileItem item = new NameFileItem(fileType, fileUnit, fileName, fileStatus);
            _NfItems.Add(fileUnit, item);
        }
        public void AddItem(NameFileItem item)
        {
            _NfItems.Add(item.FileUnit, item);
        }

        public INameFileItem RemoveItem(int fileUnit)
        {
            if (_NfItems.ContainsKey(fileUnit))
            {
                NameFileItem item = _NfItems[fileUnit];
                _NfItems.Remove(fileUnit);
                return item;
            }
            else
            { return null; }
        }

        public void Clear()
        { _NfItems.Clear(); }

        public bool FileUnitAvailable(int fileUnit)
        {
            if (_NfItems.ContainsKey(fileUnit) )
            { return false; }
            else
            { return true; }
        }

        public List<NameFileItem> GetItemsAsList()
        {
            return GetItemsAsList("");
        }

        public List<NameFileItem> GetItemsAsList(string fileType)
        {
            string fType = fileType.Trim().ToUpper();
            NameFileItem item = null;
            NameFileItem itemCopy = null;
            List<NameFileItem> list = new List<NameFileItem>();
            foreach (KeyValuePair<int, NameFileItem> itemPair in _NfItems)
            {
                if ( (itemPair.Value.FileType == fType) || (fType=="") )
                {
                    item = itemPair.Value;
                    itemCopy = new NameFileItem(item.FileType, item.FileUnit, item.FileName, item.FileStatus);
                    list.Add(itemCopy);
                }
            }

            return list;

        }

        public string GetFullFilename(string filename)
        {
            string fname = filename.Trim();
            if (string.IsNullOrEmpty(fname)) return "";

            if (System.IO.Path.IsPathRooted(fname))
            {
                return fname;
            }
            else
            {
                if (!string.IsNullOrEmpty(ParentDirectory))
                {
                    return System.IO.Path.Combine(ParentDirectory, fname);
                }
                else
                { return ""; }
            }

        }
        public string GetFullFilename(int fileUnit)
        {
            if (_NfItems.ContainsKey(fileUnit))
            {
                NameFileItem item = _NfItems[fileUnit];
                return GetFullFilename(item.FileName);
            }
            else
            { return ""; }
        }

        #endregion
    }
}
