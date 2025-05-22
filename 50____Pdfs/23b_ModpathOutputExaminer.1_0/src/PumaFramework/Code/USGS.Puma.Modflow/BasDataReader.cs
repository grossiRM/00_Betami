using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;
using USGS.Puma.Modflow;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class BasDataReader
    {

        #region Public Methods
        /// <summary>
        /// Reads the specified name data.
        /// </summary>
        /// <param name="nameData">The name data.</param>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public BasFileData Read(ModflowNameData nameData, int layerCount, int rowCount, int columnCount)
        {
            List<string> tokens = null;
            IGenericNumberUtility<float> gnu = new GenericNumberUtility() as IGenericNumberUtility<float>;
            BasFileData basData = null;
            ModflowDataArrayReader<float> mdaReader = null;

            string filename = "";
            List<NameFileItem> nfItems = nameData.GetItemsAsList("BAS6");
            if (nfItems.Count == 0)
                throw new Exception("Name file data does not include a BAS6 file.");

            filename = nfItems[0].FileName;
            if (string.IsNullOrEmpty(filename))
                throw new Exception("The BAS6 filename is invalid.");

            if (!Path.IsPathRooted(filename))
                filename = Path.Combine(nameData.ParentDirectory, filename);

            // Open file and read data
            using (StreamReader reader = new StreamReader(filename))
            {
                mdaReader = new ModflowDataArrayReader<float>(reader, nameData);
                string line = null;

                basData = new BasFileData(layerCount, rowCount, columnCount);

                // Read the comment lines.
                List<string> comments = new List<string>();
                while (!reader.EndOfStream)
                {
                    line = reader.ReadLine();
                    if (line[0] != '#') break;
                    comments.Add(line);
                }
                // Add comments
                for (int i = 0; i < comments.Count; i++)
                { basData.Comments.Add(comments[i]); }

                // The current line should be the OPTIONS line, so process it.
                tokens = StringUtility.ParseAsFortranFreeFormat(line, false);
                if (tokens != null)
                {
                    foreach (string item in tokens)
                    {
                        switch (item)
                        {
                            case "FREE":
                                basData.UseFreeFormat = true;
                                break;
                            case "CHTOCH":
                                basData.AllowFlowBetweenConstantHeads = true;
                                break;
                            case "XSECTION":
                                basData.IsCrossSection = true;
                                break;
                            default:
                                break;
                        }
                    }
                }

                // Read IBOUND
                ModflowDataArrayReader<int> intReader = new ModflowDataArrayReader<int>(reader, nameData);
                for (int i = 1; i <= layerCount; i++)
                {
                    intReader.Read(basData.GetIBound(i));
                }
                intReader = null;

                // Read HNOFLO
                line = reader.ReadLine();
                tokens = StringUtility.ParseAsFortranFreeFormat(line, false);
                basData.HNoFlo = gnu.Parse(tokens[0]);

                // Read STRT
                // don't read the starting heads until the reader code
                // is fixed to handle binary files
                //for (int i = 1; i <= layerCount; i++)
                //{
                //    mdaReader.Read(basData.GetSHead(i));
                //}

                // Return the basData object
                return basData;

            }
        }
        /// <summary>
        /// Reads the specified modflow name file.
        /// </summary>
        /// <param name="modflowNameFile">The modflow name file.</param>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public BasFileData Read(string modflowNameFile, int layerCount, int rowCount, int columnCount)
        {
            // Read the MODFLOW name file
            ModflowNameData nfData = ModflowNameFileReader.Read(modflowNameFile);

            // If nfData is null it means something went wrong reading the name
            // file. If so, return null.
            if (nfData == null) return null;

            // The nfData has valid name file data, so read the DIS file.
            return Read(nfData, layerCount, rowCount, columnCount);

        }
        #endregion
    }
}
