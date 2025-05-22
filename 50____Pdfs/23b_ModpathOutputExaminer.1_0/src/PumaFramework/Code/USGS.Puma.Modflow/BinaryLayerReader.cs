using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.IO;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// This class reads a MODFLOW binary head or drawdown output file.
    /// </summary>
    /// <remarks>
    /// The class provides methods that allow layer data to be read in random order by specifying layer index information.
    /// </remarks>
    /// <example>
    /// <code>
    /// <![CDATA[
    /// // Create a BinaryLayerReader instance connected to a MODFLOW head file. 
    /// // Then read the layer data record for layer 1, stress period 1, time step 5.
    /// // Get the head values for the layer from the DataArray property of the
    /// // layer data record.
    /// 
    /// string filename = @"C:\MyData\Simulation.head";
    /// BinaryLayerReader reader = new BinaryLayerReader(filename);
    /// 
    /// // Find the record index
    /// int recordIndex = BinaryLayerReader.FindRecordIndex(1, 1, 5, "HEAD");
    /// 
    /// // 
    /// if(recordIndex != -1)
    /// {
    ///    LayerDataRecord<float> record = reader.GetRecordAsSingle(recordIndex);
    ///    Array2d<float> head = record.DataArray;
    /// }
    /// else
    /// {
    ///   // The file does not contain the specified layer data.
    ///   //
    ///   // do something here ...
    /// }
    /// 
    /// // Close the file and exit the reader.
    /// reader.Close();
    /// 
    /// ]]>
    /// </code>
    /// </example>
    public class BinaryLayerReader : IDisposable
    {
        #region Fields
        /// <summary>
        /// 
        /// </summary>
        private int _BytesPerElement;
        /// <summary>
        /// 
        /// </summary>
        private long _HeaderOffset;
        /// <summary>
        /// 
        /// </summary>
        private int _ArrayLength;
        /// <summary>
        /// 
        /// </summary>
        private long _ArrayOffset;
        /// <summary>
        /// 
        /// </summary>
        private long _RecordOffset;
        /// <summary>
        /// 
        /// </summary>
        private BinaryReader _reader = null;
        /// <summary>
        /// 
        /// </summary>
        private bool _disposed = false;
        /// <summary>
        /// 
        /// </summary>
        private BinaryArrayIO _bfio = null;
        /// <summary>
        /// 
        /// </summary>
        private FileStream _fs = null;
        /// <summary>
        /// 
        /// </summary>
        private const string _HEAD =     "            HEAD";
        /// <summary>
        /// 
        /// </summary>
        private const string _DRAWDOWN = "        DRAWDOWN";
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryLayerReader"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <remarks></remarks>
        public BinaryLayerReader(string filename)
        {
            _Filename = filename;
            _fs = new FileStream(_Filename, FileMode.Open, FileAccess.Read, FileShare.Read);
            _reader = new BinaryReader(_fs);

            _Valid = true;
            if (!ValidateFile(_reader))
            {
                Close();
                _Valid = false;
            }

            // Create a BinaryArrayFileIO helper object.
            _bfio = new BinaryArrayIO();

        }
        #endregion

        #region Public Members

        /// <summary>
        /// 
        /// </summary>
        private string _Filename = "";
        /// <summary>
        /// Gets the filename currently connected to the reader.
        /// </summary>
        /// <remarks></remarks>
        public string Filename
        {
            get { return _Filename; }
        }

        /// <summary>
        /// 
        /// </summary>
        private bool _Valid;
        /// <summary>
        /// Gets a boolean value indicating whether the reader is connected correctly to a valid MODFLOW binary output file.
        /// </summary>
        /// <remarks></remarks>
        public bool Valid
        {
            get { return _Valid; }
        }

        /// <summary>
        /// 
        /// </summary>
        private OutputPrecisionType _OutputPrecision;
        /// <summary>
        /// Gets a value indicating the precision of the connected MODFLOW output file.
        /// </summary>
        /// <value>The output precision.</value>
        /// <remarks></remarks>
        public OutputPrecisionType OutputPrecision
        {
            get { return _OutputPrecision; }
            set { _OutputPrecision = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _RowCount;
        /// <summary>
        /// Gets the number of rows in the MODFLOW dataset.
        /// </summary>
        /// <value>The row count.</value>
        /// <remarks></remarks>
        public int RowCount
        {
            get { return _RowCount; }
            set { _RowCount = value; }
        }

        /// <summary>
        /// Gets the number of columns in the MODFLOW dataset.
        /// </summary>
        private int _ColumnCount;
        /// <summary>
        /// Gets or sets the column count.
        /// </summary>
        /// <value>The column count.</value>
        /// <remarks></remarks>
        public int ColumnCount
        {
            get { return _ColumnCount; }
            set { _ColumnCount = value; }
        }

        /// <summary>
        /// Gets the number of layer data records in the file.
        /// </summary>
        private int _RecordCount;
        /// <summary>
        /// Gets or sets the record count.
        /// </summary>
        /// <value>The record count.</value>
        /// <remarks></remarks>
        public int RecordCount
        {
            get { return _RecordCount; }
            set { _RecordCount = value; }
        }

        /// <summary>
        /// Retrieve a LayerDataRecordHeader for the specified record index.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns>A LayerDataRecordHeader object</returns>
        /// <remarks>The LayerDataRecordHeader is identical for both single precision and double precision data. In both cases, the floating point properties PeriodTime and TotalTime are represented as singe precision even though a double precision MODFLOW output file would contain double precision values.</remarks>
        public LayerDataRecordHeader ReadRecordHeader(int recordIndex)
        {
            try
            {
                if (recordIndex > -1 && recordIndex < RecordCount)
                {
                    // Compute the offset from the start of the file to the record
                    long n = (long)recordIndex * _RecordOffset;
                    // Move the file pointer to the beginning of the record
                    _fs.Seek(n, SeekOrigin.Begin);

                    // Read the record header
                    LayerDataRecordHeader header = ReadRecordHeader(_reader, OutputPrecision);

                    if (header == null) return null;

                    return header;

                }
                else
                { return null; }

            }
            catch (Exception)
            {
                return null;
            }
        }
        /// <summary>
        /// Returns a list of layer data record headers for all the records in the file.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public LayerDataRecordHeaderCollection ReadAllRecordHeaders()
        {
            LayerDataRecordHeader header = null;
            LayerDataRecordHeaderCollection list = new LayerDataRecordHeaderCollection();
            for (int i = 0; i < RecordCount; i++)
            {
                header = ReadRecordHeader(i);
                if (header != null) list.Add(header);
            }
            return list;
        }
        /// <summary>
        /// Returns a list of strings indicating all of the MODFLOW layer data types
        /// present in the file (HEAD, DRAWDOWN, CONCENTRATION, etc.).
        /// </summary>
        /// <returns>A collection of strings.</returns>
        /// <remarks>The string represent the 16-character TEXT field saved by MODFLOW. By convention, MODFLOW saves all text as right-justified strings.</remarks>
        public Collection<string> GetDataTypes()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            Collection<string> list = new Collection<string>();
            LayerDataRecordHeaderCollection headers = ReadAllRecordHeaders();

            foreach (LayerDataRecordHeader header in headers)
            {
                if (!dict.ContainsKey(header.Text))
                    dict.Add(header.Text, header.Text);
            }

            foreach (KeyValuePair<string, string> item in dict)
            {
                list.Add(item.Key);
            }

            return list;

        }
        /// <summary>
        /// Returns a LayerDataRecord object corresponding to the specified record index.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public object GetRecord(int recordIndex)
        {
            if (OutputPrecision == OutputPrecisionType.Single)
            {
                return ReadRecordSingle(recordIndex);
            }
            else if (OutputPrecision == OutputPrecisionType.Double)
            {
                return ReadRecordDouble(recordIndex);
            }
            else
            {
                throw new InvalidOperationException("Invalid output precision: OutputPrecision = " + this.OutputPrecision.ToString());
            }
        }
        /// <summary>
        /// Retrieve the specified record as a single precision data array.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns>A double precision LayerDataRecord object</returns>
        /// <remarks>This method returns a single precision record even if the file contains double precision data.</remarks>
        public LayerDataRecord<float> GetRecordAsSingle(int recordIndex)
        {
            if (recordIndex > -1 && recordIndex < this.RecordCount)
            {
                object recObject = GetRecord(recordIndex);

                if (recObject is LayerDataRecord<float>)
                {
                    return recObject as LayerDataRecord<float>;
                }
                else if (recObject is LayerDataRecord<double>)
                {
                    // Convert double precision record to single precision record
                    LayerDataRecord<double> rec = recObject as LayerDataRecord<double>;
                    return rec.CopyAsSingle();
                }
                else
                {
                    throw new InvalidOperationException("The LayerDataRecord object either is not set or is an invalid type.");
                }
            }
            else
            {
                throw new ArgumentOutOfRangeException("recordIndex");
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="header"></param>
        /// <returns></returns>
        public LayerDataRecord<float> GetRecordAsSingle(LayerDataRecordHeader header)
        {
            int recordIndex = FindRecordIndex(header);
            return GetRecordAsSingle(recordIndex);
        }
        /// <summary>
        /// Retrieve the specified record as a double precision data array.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns>A double precision LayerDataRecord object</returns>
        /// <remarks>This method returns a double precision record even if the file contains single precision data.</remarks>
        public LayerDataRecord<double> GetRecordAsDouble(int recordIndex)
        {
            if (recordIndex > -1 && recordIndex < this.RecordCount)
            {
                object recObject = GetRecord(recordIndex);

                if (recObject is LayerDataRecord<double>)
                {
                    return recObject as LayerDataRecord<double>;
                }
                else if (recObject is LayerDataRecord<float>)
                {
                    // Convert single precision record to double precision record
                    LayerDataRecord<float> rec = recObject as LayerDataRecord<float>;
                    return rec.CopyAsDouble();
                }
                else
                {
                    throw new InvalidOperationException("The LayerDataRecord object either is not set or is an invalid type.");
                }
            }
            else
            {
                throw new ArgumentOutOfRangeException("recordIndex");
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="header"></param>
        /// <returns></returns>
        public LayerDataRecord<double> GetRecordAsDouble(LayerDataRecordHeader header)
        {
            int recordIndex = FindRecordIndex(header);
            return GetRecordAsDouble(recordIndex);
        }
        /// <summary>
        /// Returns the record index corresponding to the specified layer, stress period,
        /// time step, and MODFLOW data type (HEAD, DRAWDOWN, etc.)
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <param name="stressPeriod">The stress period.</param>
        /// <param name="timeStep">The time step.</param>
        /// <param name="text">The text.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public int FindRecordIndex(int layer, int stressPeriod, int timeStep, string text)
        {
            try
            {
                LayerDataRecordHeader h;
                LayerDataRecordHeaderCollection headers = RecordHeadersCache;
                for (int i = 0; i < RecordCount; i++)
                {
                    h = headers[i];
                    string s = text.Trim().ToUpper();
                    if (layer == h.Layer && stressPeriod == h.StressPeriod && timeStep == h.TimeStep && s == h.Text.Trim().ToUpper())
                    {
                        return i;
                    }
                }
                return -1;
            }
            catch (Exception)
            {
                return -1;
            }
        }
        /// <summary>
        /// Returns the record index corresponding to the specified record key.
        /// The record key is a comma-delimited string of the form: a,b,c,d
        /// where a= layer, b = stress period, c = time step, and d = MODFLOW data type.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public int FindRecordIndex(string key)
        {
            try
            {
                string[] s = key.Split(',');
                if (s.Length != 4) return -1;
                int stressPeriod = Int32.Parse(s[0]);
                int timeStep = Int32.Parse(s[1]);
                int layer = Int32.Parse(s[2]);
                string text = s[3].Trim();
                return FindRecordIndex(layer, stressPeriod, timeStep, text);
            }
            catch (Exception)
            {
                return -1;
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="header"></param>
        /// <returns></returns>
        public int FindRecordIndex(LayerDataRecordHeader header)
        {
            if (header == null)
            {
                return -1;
            }
            else
            {
                return FindRecordIndex(header.Layer, header.StressPeriod, header.TimeStep, header.Text);
            }
        }
        /// <summary>
        /// Get an array of all the stress period numbers that are present in the file.
        /// </summary>
        /// <returns>An array of integer values.</returns>
        /// <remarks></remarks>
        public int[] GetStressPeriods()
        {
            List<int> periods = new List<int>();
            if (RecordHeadersCache != null)
            {
                foreach (LayerDataRecordHeader header in RecordHeadersCache)
                {
                    if (!periods.Contains(header.StressPeriod))
                    {
                        periods.Add(header.StressPeriod);
                    }
                }
            }

            return periods.ToArray<int>();

        }
        /// <summary>
        /// Get a collection of all the distinct time steps present in the file.
        /// </summary>
        /// <returns>A TimeStepCollection object.</returns>
        /// <remarks></remarks>
        public TimeStepCollection GetTimeSteps()
        {
            TimeStepCollection tsCollection = new TimeStepCollection();
            if (RecordHeadersCache != null)
            {
                foreach (LayerDataRecordHeader header in RecordHeadersCache)
                {
                    if (!tsCollection.ContainsTimeStep(header.StressPeriod,header.TimeStep))
                    {
                        TimeStep ts = new TimeStep(header.StressPeriod, header.TimeStep);
                        tsCollection.Add(ts);
                    }
                }
            }

            return tsCollection;

        }
        /// <summary>
        /// Close the data file and release resources.
        /// </summary>
        /// <remarks></remarks>
        public void Close()
        {
            Dispose();
        }
        /// <summary>
        /// Release resources.
        /// </summary>
        /// <remarks></remarks>
        public void Dispose()
        {
            // Delegate to the protected Dispose member
            Dispose(true);
            // Tell the garbage collector it does not need to call the finalizer method
            GC.SuppressFinalize(this);
        }
        #endregion

        #region Private Members

        /// <summary>
        /// 
        /// </summary>
        private LayerDataRecordHeaderCollection _RecordHeadersCache = null;
        /// <summary>
        /// Gets the record headers cache.
        /// </summary>
        /// <remarks></remarks>
        private LayerDataRecordHeaderCollection RecordHeadersCache
        {
            get 
            {
                if (_RecordHeadersCache == null)
                {
                    _RecordHeadersCache = ReadAllRecordHeaders();
                }
                return _RecordHeadersCache; 
            }
        }

        /// <summary>
        /// Read and return a record header.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="precision">The precision.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private LayerDataRecordHeader ReadRecordHeader(BinaryReader reader, OutputPrecisionType precision)
        {
            try
            {
                LayerDataRecordHeader header = new LayerDataRecordHeader();
                header.TimeStep = reader.ReadInt32();
                header.StressPeriod = reader.ReadInt32();
                if (precision == OutputPrecisionType.Single)
                {
                    header.PeriodTime = reader.ReadSingle();
                    header.TotalTime = reader.ReadSingle();
                }
                else if (precision == OutputPrecisionType.Double)
                {
                    header.PeriodTime = Convert.ToSingle(reader.ReadDouble());
                    header.TotalTime = Convert.ToSingle(reader.ReadDouble());
                }
                else
                {
                    return null;
                }
                header.Text = StringUtility.CharArrayToString(reader.ReadChars(16));
                header.ColumnCount = reader.ReadInt32();
                header.RowCount = reader.ReadInt32();
                header.Layer = reader.ReadInt32();
                return header;
            }
            catch (Exception e)
            {
                return null;
            }
        }
        /// <summary>
        /// Reads the record single.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private LayerDataRecord<float> ReadRecordSingle(int recordIndex)
        {
            if (recordIndex > -1 && recordIndex < RecordCount)
            {
                // Read the record header
                LayerDataRecordHeader header = ReadRecordHeader(recordIndex);
                if (header == null) return null;

                // Create a new record object
                LayerDataRecord<float> record = new LayerDataRecord<float>();

                // Process header
                record.TimeStep = header.TimeStep;
                record.StressPeriod = header.StressPeriod;
                record.Layer = header.Layer;
                record.Text = header.Text;
                record.PeriodTime = header.PeriodTime;
                record.TotalTime = header.TotalTime;

                // Read array values into a buffer and attach it to record.DataArray
                Array2d<float> buffer = new Array2d<float>(this.RowCount, this.ColumnCount);
                _bfio.LoadFromStream(buffer, _reader);
                record.DataArray = buffer;

                // Return the record as the result.
                return record;

            }
            else
            {
                throw new ArgumentOutOfRangeException("recordIndex");
            }

        }
        /// <summary>
        /// Reads the record double.
        /// </summary>
        /// <param name="recordIndex">Index of the record.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private LayerDataRecord<double> ReadRecordDouble(int recordIndex)
        {
            if (recordIndex > -1 && recordIndex < RecordCount)
            {
                // Read the record header
                LayerDataRecordHeader header = ReadRecordHeader(recordIndex);
                if (header == null) return null;

                // Create a new record object
                LayerDataRecord<double> record = new LayerDataRecord<double>();

                // Process header
                record.TimeStep = header.TimeStep;
                record.StressPeriod = header.StressPeriod;
                record.Layer = header.Layer;
                record.Text = header.Text;
                record.PeriodTime = header.PeriodTime;
                record.TotalTime = header.TotalTime;

                // Read array values into a buffer and attach it to record.DataArray
                Array2d<double> buffer = new Array2d<double>(this.RowCount, this.ColumnCount);
                _bfio.LoadFromStream(buffer, _reader);
                record.DataArray = buffer;

                // Return the record as the result.
                return record;

            }
            else
            {
                throw new ArgumentOutOfRangeException("recordIndex");
            }

        }
        /// <summary>
        /// Determine if the data reader is hooked up to a valid MODFLOW layer data file.
        /// If the file is valid it returns true
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool ValidateFile(BinaryReader reader)
        {
            // Try to read the header of the first record. If the header cannot
            // be read as either a single or double precision file, return false. 
            // If the header can be read, proceed and compute the size of each 
            // data record and check to see if the file size is a multiple of 
            // that value. If not, then it indicates a problem, so return 
            // false. If everything checks out. return true.
            try
            {
                long length = reader.BaseStream.Length;
                _BytesPerElement = 0;
                _ArrayLength = 0;
                _ArrayOffset = 0;
                _RecordOffset = 0;
                _RecordCount = 0;
                _OutputPrecision = OutputPrecisionType.Undefined;

                // No need to check further if the length is less than the
                // length of a single precision file header.
                if (length < 44)
                { return false; }

                LayerDataRecordHeader header = ValidateHeader(reader, OutputPrecisionType.Single);
                if (header != null)
                {
                    _BytesPerElement = 4;
                    _HeaderOffset = 44;
                    _OutputPrecision = OutputPrecisionType.Single;
                }
                else
                {
                    header = ValidateHeader(reader, OutputPrecisionType.Double);
                    if (header != null)
                    {
                        _BytesPerElement = 8;
                        _HeaderOffset = 52;
                        _OutputPrecision = OutputPrecisionType.Double;
                    }
                    else
                    {
                        return false;
                    }
                }

                if (header.RowCount == 0 || header.ColumnCount == 0) return false;
                RowCount = header.RowCount;
                ColumnCount = header.ColumnCount;

                // Check to see if the file size is consistent with the array
                // record size by making sure the file length is an even
                // multiple of the record length.

                _ArrayLength = RowCount * ColumnCount;
                _ArrayOffset = (long)(_BytesPerElement * _ArrayLength);
                _RecordOffset = _HeaderOffset + _ArrayOffset;
                long n = length % _RecordOffset;
                if (n != 0) return false;

                // Otherwise, return true
                n = length / _RecordOffset;
                RecordCount = (int)n;
                return true;

            }
            catch (Exception)
            {
                return false;
            }

        }
        /// <summary>
        /// Validates the header.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="precision">The precision.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private LayerDataRecordHeader ValidateHeader(BinaryReader reader, OutputPrecisionType precision)
        {
            try
            {
                reader.BaseStream.Seek(0, SeekOrigin.Begin);
                LayerDataRecordHeader header = ReadRecordHeader(reader, precision);
                
                if (header != null)
                {
                    if ((header.Text == _HEAD) || (header.Text == _DRAWDOWN))
                    {
                        return header;
                    }
                }

                // Otherwise, return false
                return null;
                
            }
            catch (Exception e)
            {
                return null;
            }
        }
        
        
        #endregion

        #region Protected Members
        /// <summary>
        /// Close file, release resources, and terminate the binary reader object.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        /// <remarks></remarks>
        protected void Dispose(bool disposing)
        {
            try
            {
                if (_disposed == false && disposing == true)
                {
                    if (_reader != null)
                    { 
                        _reader.Close();
                        _reader = null;
                    }

                    if (_fs != null)
                    {
                        _fs.Close();
                        _fs = null;
                    }

                    // Indicate this has already been done
                    _disposed = true;

                }
            }
            catch (Exception ex)
            {
                // do nothing for now.
            }
        }
        #endregion


    }
}
