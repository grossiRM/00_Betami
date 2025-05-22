using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modflow
{
    public class OcDataReader
    {
        #region Private Fields

        #endregion

        #region Public Methods
        public OcDataHeader ReadHeader(string filename)
        {
            try
            {
                OcDataHeader header = new OcDataHeader();
                string line = null;
                string[] s = null;
                char[] delimiters = new char[2] { ' ', ',' };


                using (StreamReader reader = new StreamReader(filename))
                {
                    while (!reader.EndOfStream)
                    {
                        line = reader.ReadLine().ToUpper().PadRight(132);

                        if (!string.IsNullOrEmpty(line))
                        {
                            // Check for the end of the header info and break if found.
                            if (line.Substring(0, 6) == "PERIOD") break;
                            
                            // Check for comment line.
                            if (line[0] == '#')
                            {
                                if (line.Length == 1)
                                { header.Comments.Add(""); }
                                else
                                { header.Comments.Add(line.Substring(1)); }
                            }
                            else
                            {
                                // Parse instruction
                                s = line.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

                                // Check to see if this is the numeric style OC
                                int hedfm;
                                if (int.TryParse(s[0], out hedfm))
                                {
                                    header.HeadPrintFormat = hedfm;
                                    header.DrawdownPrintFormat = int.Parse(s[1]);
                                    header.HeadSaveUnit = int.Parse(s[2]);
                                    header.DrawdownSaveUnit = int.Parse(s[3]);
                                    reader.Close();
                                    return header;
                                }

                                // Continue and assume the OC file uses word formatting
                                if (line.Substring(0, 17) == "HEAD PRINT FORMAT")
                                {
                                    if (s.Length > 3)
                                    { header.HeadPrintFormat = int.Parse(s[3]); }
                                }
                                else if (line.Substring(0, 16) == "HEAD SAVE FORMAT")
                                {
                                    if (s.Length > 3)
                                    {
                                        header.HeadSaveFormat = int.Parse(s[3]);
                                        header.HeadSaveFormatLabel = false;
                                        if (s.Length > 4)
                                        {
                                            if (s[4] == "LABEL") header.HeadSaveFormatLabel = true;
                                        }
                                    }
                                }
                                else if (line.Substring(0, 14) == "HEAD SAVE UNIT")
                                {
                                    if (s.Length > 3) header.HeadSaveUnit = int.Parse(s[3]);
                                }
                                else if (line.Substring(0, 21) == "DRAWDOWN PRINT FORMAT")
                                {
                                    if (s.Length > 3)
                                    { header.DrawdownPrintFormat = int.Parse(s[3]); }
                                }
                                else if (line.Substring(0, 20) == "DRAWDOWN SAVE FORMAT")
                                {
                                    if (s.Length > 3)
                                    {
                                        header.DrawdownSaveFormat = int.Parse(s[3]);
                                        header.DrawdownSaveFormatLabel = false;
                                        if (s.Length > 4)
                                        {
                                            if (s[4] == "LABEL") header.DrawdownSaveFormatLabel = true;
                                        }
                                    }
                                }
                                else if (line.Substring(0, 18) == "DRAWDOWN SAVE UNIT")
                                {
                                    if (s.Length > 3) header.DrawdownSaveUnit = int.Parse(s[3]);
                                }
                                else if (line.Substring(0, 18) == "IBOUND SAVE FORMAT")
                                {
                                    if (s.Length > 3)
                                    {
                                        header.IboundSaveFormat = s[3];
                                        header.IboundSaveFormatLabel = false;
                                        if (s.Length > 4)
                                        {
                                            if (s[4] == "LABEL") header.IboundSaveFormatLabel = true;
                                        }
                                    }
                                }
                                else if (line.Substring(0, 16) == "IBOUND SAVE UNIT")
                                {
                                    if (s.Length > 3) header.IboundSaveUnit = int.Parse(s[3]);
                                }
                                else if (line.Substring(0, 14) == "COMPACT BUDGET")
                                {
                                    header.CompactBudget = true;
                                    if (s.Length > 2)
                                    {
                                        if ((s[2] == "AUX") || (s[2] == "AUXILIARY"))
                                        {
                                            header.CompactBudgetSaveAuxiliary = true;
                                        }
                                    }
                                }
                            }
                        }

                    }

                    reader.Close();
                }

                return header;

            }
            catch (Exception)
            {
                throw;
            }
        }

        #endregion


    }
}
