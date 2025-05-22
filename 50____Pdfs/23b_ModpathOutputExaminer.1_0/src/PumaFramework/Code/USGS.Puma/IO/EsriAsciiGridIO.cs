using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.IO
{
    /// <summary>
    /// Provides support for reading and writing two-dimensional array data in
    /// ESRI AsciiGrid format.
    /// </summary>
    public class EsriAsciiGridIO
    {
        /// <summary>
        /// Writes an AsciiGrid file
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="buffer"></param>
        /// <param name="xllCorner"></param>
        /// <param name="yllCorner"></param>
        /// <param name="cellSize"></param>
        /// <param name="noDataValue"></param>
        public static void Write(string filename, Array2d<float> buffer, double xllCorner, double yllCorner, double cellSize, float noDataValue)
        {
            if (buffer == null)
                throw new NullReferenceException();

            using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filename))
            {
                char sp = ' ';
                
                writer.WriteLine("NCOLS " + buffer.ColumnCount.ToString());
                writer.WriteLine("NROWS " + buffer.RowCount.ToString());
                writer.WriteLine("XLLCORNER " + xllCorner.ToString());
                writer.WriteLine("YLLCORNER " + yllCorner.ToString());
                writer.WriteLine("CELLSIZE " + cellSize.ToString());
                writer.WriteLine("NODATA_VALUE " + noDataValue.ToString());

                for (int row = 1; row <= buffer.RowCount; row++)
                {
                    for (int column = 1; column <= buffer.ColumnCount; column++)
                    {
                        writer.Write(buffer[row, column]);
                        if (column < buffer.ColumnCount)
                            writer.Write(sp);
                        else
                            writer.Write(Environment.NewLine);
                    }
                }
            }

        }

        /// <summary>
        /// Reads a floating point ESRI AsciiGrid file and returns a float Array2d object
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static Array2d<float> ReadSingle(string filename)
        {
            using (System.IO.StreamReader reader = new System.IO.StreamReader(filename))
            {
                char[] delimiters = new char[1] {' '};
                string[] parts;
                string sLine;
                int rowCount;
                int columnCount;
                string sValue;

                // Read number of columns
                sLine = reader.ReadLine();
                parts = sLine.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2)
                    throw new Exception("Invalid AsciiGrid data.");
                if (parts[0].ToUpper() != "NCOLS")
                    throw new Exception("Invalid AsciiGrid data.");
                columnCount = Convert.ToInt32(parts[1]);

                // Read number of rows
                sLine = reader.ReadLine();
                parts = sLine.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2)
                    throw new Exception("Invalid AsciiGrid data.");
                if (parts[0].ToUpper() != "NROWS")
                    throw new Exception("Invalid AsciiGrid data.");
                rowCount = Convert.ToInt32(parts[1]);

                // Read the other AsciiGrid header lines, but no need to process
                sLine = reader.ReadLine(); //  xllCorner
                sLine = reader.ReadLine(); //  yllCorner
                sLine = reader.ReadLine(); //  CELLSIZE
                sLine = reader.ReadLine(); //  NODATA_VALUE

                // Read the values into the buffer array
                Array2d<float> buffer = new Array2d<float>(rowCount, columnCount);
                StringBuilder sb = new StringBuilder(50);

                for (int row = 1; row <= buffer.RowCount; row++)
                {
                    for (int column = 1; column <= buffer.ColumnCount; column++)
                    {
                        sValue = StringUtility.GetNextToken(reader, sb);
                        buffer[row, column] = Convert.ToSingle(sValue);
                    }
                }

                return buffer;

            }

        }
    }

}
