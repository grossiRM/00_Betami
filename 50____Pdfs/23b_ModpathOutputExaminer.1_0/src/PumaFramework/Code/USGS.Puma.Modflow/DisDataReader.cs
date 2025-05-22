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
    public class DisDataReader
    {
        #region Constructors
        public DisDataReader()
        {
            // Use default constructor
        }
        #endregion

        #region Public Methods
        public DisFileData Read(ModflowNameData nameData)
        {
            IGenericNumberUtility<float> gnu = new GenericNumberUtility() as IGenericNumberUtility<float>;
            DisFileData disFileData = null;
            ModflowDataArrayReader<float> mdaReader = null;

            string disFilename = "";
            List<NameFileItem> nfItems = nameData.GetItemsAsList("DIS");
            if (nfItems.Count == 0)
                throw new Exception("Name file data does not include a DIS file.");

            disFilename = nfItems[0].FileName;
            if (string.IsNullOrEmpty(disFilename))
                throw new Exception("The DIS filename is invalid.");

            if (!Path.IsPathRooted(disFilename))
                disFilename = Path.Combine(nameData.ParentDirectory, disFilename);

            // Open file and read data
            using (StreamReader reader = new StreamReader(disFilename))
            {
                mdaReader = new ModflowDataArrayReader<float>(reader, nameData);
                string line = null;

                // Read the comment lines.
                List<string> comments = new List<string>();
                while (!reader.EndOfStream)
                {
                    line = reader.ReadLine();
                    if (line[0] != '#') break;
                    comments.Add(line);
                }

                // The current value of line should be the header line of the DIS file
                List<string> header = StringUtility.ParseAsFortranFreeFormat(line, false);
                int layerCount = int.Parse(header[0]);
                int rowCount = int.Parse(header[1]);
                int columnCount = int.Parse(header[2]);
                int stressPeriodCount = int.Parse(header[3]);
                int timeUnit = int.Parse(header[4]);
                int lengthUnit = int.Parse(header[5]);

                // Create dis data object
                disFileData = new DisFileData(layerCount, rowCount, columnCount, 0.0f, 0.0f);
                
                // Set NameData reference
                disFileData.NameData = nameData;
                
                // Add comments
                for (int i = 0; i < comments.Count; i++)
                { disFileData.Comments.Add(comments[i]); }

                // Read the layer type flags
                int[] cbFlags = new int[layerCount];
                TextArrayIO<int> intArrayReader = new TextArrayIO<int>();
                if (!intArrayReader.Read(cbFlags, reader))
                    throw new Exception("Error reading the confining bed flags in the DIS package file.");
                
                for (int i= 0; i< layerCount; i++)
                {
                    disFileData.SetLayerConfiningBedFlag(i + 1, cbFlags[i]);
                }

                // Read DELR
                mdaReader.Read(disFileData.DelR);

                // Read DELC
                mdaReader.Read(disFileData.DelC);

                // Read Top elevation
                mdaReader.Read(disFileData.Top);

                // Read Bottom elevation and confining bed bottom elevation
                for (int i = 1; i <= disFileData.LayerCount; i++)
                {
                    // Bottom elevation
                    mdaReader.Read(disFileData.GetBottom(i));

                    // Confining bed bottom elevation if present.
                    if (disFileData.GetLayerConfiningBedFlag(i) != 0)
                    { mdaReader.Read(disFileData.GetConfiningBedBottom(i)); }

                }

                // Read Stress Period Definitions
                float periodLength;
                int timeStepCount;
                float timeStepMultiplier;
                StressPeriodType spType = StressPeriodType.SteadyState;
                List<string> spDefItems = null;
                for (int i = 1; i <= stressPeriodCount; i++)
                {
                    line = reader.ReadLine();
                    spDefItems = StringUtility.ParseAsFortranFreeFormat(line, false);
                    periodLength = gnu.Parse(spDefItems[0]);
                    timeStepCount=int.Parse(spDefItems[1]);
                    timeStepMultiplier=gnu.Parse(spDefItems[2]);
                    spType=StressPeriodType.SteadyState;
                    if (spDefItems[3].Trim().ToLower() == "tr")
                    { spType = StressPeriodType.Transient; }
                    int timeStepIndex = disFileData.TimeDiscretization.AddStressPeriod(periodLength, timeStepCount, timeStepMultiplier, spType);
                }

            }

            return disFileData;
        }
        public DisFileData Read(string filename, string modflowFiletype)
        {
            string filetype = modflowFiletype.ToLower();
            ModflowNameData nfData = null;

            if (filetype == "nam")
            {
                // Read the MODFLOW name file
                nfData = ModflowNameFileReader.Read(filename);
            }
            else if (filetype == "dis")
            {
                // First, create a dummy ModflowNameData list and add a record
                // for the DIS file. Then pass that to the DisDataReader to process
                // the DIS file. This option will handle array data that is
                // constant, internal, or open/close external. It will fail to
                // handle external arrays connected by unit number because no
                // Modflow name file is provided. Also, the filename specified
                // must be a full pathname.
                nfData = new ModflowNameData();
                nfData.AddItem(10, "DIS", filename);
                nfData.ParentDirectory = Path.GetDirectoryName(filename);
            }

            // If nfData is null it means something went wrong reading the name
            // file. If so, return null.
            if (nfData == null) return null;

            // The nfData has valid name file data, so read the DIS file.
            return Read(nfData);

        }
        #endregion

    }


}
