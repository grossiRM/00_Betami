using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modflow
{
    public class ModflowNameFileReader
    {
        #region Static Members

        public static ModflowNameData Read(string filename)
        {
            try
            {
                ModflowNameData nameFileItems = new ModflowNameData();
                if (!Path.IsPathRooted(filename))
                    return null;

                nameFileItems.ParentDirectory = Path.GetDirectoryName(filename);

                using (StreamReader reader = new StreamReader(filename))
                {
                    string sLine = null;
                    NameFileItem item = null;
                    List<string> itemList = null;

                    while (!reader.EndOfStream)
                    {
                        sLine = reader.ReadLine().TrimEnd(' ');
                        if (!string.IsNullOrEmpty(sLine))
                        {
                            if (sLine[0] != '#')
                            {
                                itemList = Puma.Utilities.StringUtility.ParseAsFortranFreeFormat(sLine, false);
                                item = new NameFileItem();
                                if (itemList.Count > 0) item.FileType = itemList[0];
                                if (itemList.Count > 1) item.FileUnit = Int32.Parse(itemList[1]);
                                if (itemList.Count > 2) item.FileName = itemList[2];
                                if (itemList.Count > 3) item.FileStatus = itemList[3];
                                if (nameFileItems.FileUnitAvailable(item.FileUnit))
                                { nameFileItems.AddItem(item); }
                            }
                        }
                    }
                }

                return nameFileItems;

            }
            catch (Exception)
            {
                return null;
            }
        }

        public static void Write(string filename, List<NameFileItem> items)
        {
            // Add code to write a MODFLOW name file from a list of items.
        }

        #endregion
    }
}
