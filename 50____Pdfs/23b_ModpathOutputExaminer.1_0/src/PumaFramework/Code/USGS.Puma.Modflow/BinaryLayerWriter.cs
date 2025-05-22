using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class BinaryLayerWriter : IDisposable
    {
        #region Static Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        /// <param name="filename"></param>
        public static void WriteFile(BinaryLayerReader reader, string filename, OutputPrecisionType precision)
        {
            BinaryLayerWriter writer = null;

            if (reader == null)
            {
                throw new ArgumentNullException("reader");
            }
            if (reader.OutputPrecision == OutputPrecisionType.Undefined)
            {
                throw new InvalidOperationException("The reader output precision is undefined.");
            }
            if (precision == OutputPrecisionType.Undefined)
            {
                throw new ArgumentException("The output precision is undefined.");
            }
            if (reader.Filename.ToLower() == filename.ToLower())
            {
                throw new InvalidOperationException("The output file cannot be the same file specified in the file reader.");
            }

            try
            {
                writer = new BinaryLayerWriter(reader.RowCount, reader.ColumnCount, filename);
                if (precision == OutputPrecisionType.Single)
                {
                    for (int i = 0; i < reader.RecordCount; i++)
                    {
                        LayerDataRecord<float> rec = reader.GetRecordAsSingle(i);
                        writer.Write(rec);
                    }
                }
                else if (precision == OutputPrecisionType.Double)
                {
                    for (int i = 0; i < reader.RecordCount; i++)
                    {
                        LayerDataRecord<double> rec = reader.GetRecordAsDouble(i);
                        writer.Write(rec);
                    }
                }
                writer.Close();
                writer = null;
            }
            finally
            {
                if (writer != null)
                {
                    writer.Close();
                }
            }
            

        }
        #endregion

        #region Private Fields
        private BinaryWriter _writer = null;
        private bool _disposed = false;
        private BinaryArrayIO _bfio = null;
        private FileStream _fs = null;
        #endregion

        #region Constructors
        public BinaryLayerWriter(int rowCount, int columnCount, string filename)
        {
            if (rowCount < 1 || columnCount < 1)
            { throw new ArgumentException("Row and column dimensions must be greater than zero."); }

            RowCount = rowCount;
            ColumnCount = columnCount;

            _fs = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None);
            _writer = new BinaryWriter(_fs);
            _bfio = new BinaryArrayIO();

        }
        #endregion

        #region Public Members
        private int _RowCount;
        /// <summary>
        /// 
        /// </summary>
        public int RowCount
        {
            get { return _RowCount; }
            set { _RowCount = value; }
        }
        private int _ColumnCount;
        /// <summary>
        /// 
        /// </summary>
        public int ColumnCount
        {
            get { return _ColumnCount; }
            set { _ColumnCount = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="record"></param>
        public void Write(LayerDataRecord<float> record)
        {
            if (record == null)
            { throw new ArgumentNullException("record"); }

            if (record.DataArray == null)
            { throw new ArgumentNullException("DataArray"); }

            Array2d<float> dataArray = record.DataArray;
            if (dataArray.RowCount != RowCount || dataArray.ColumnCount != ColumnCount)
            { throw new ArgumentException("The dimensions of the input array do not match the specified dimensions for this file."); }

            // Write header
            char[] charLabel = record.Text.ToCharArray();
            if (charLabel.Length != 16)
            {
                throw new Exception("The Text value for the LayerDataRecordHeader is not 16 characters in length.");
            }
            _writer.Write(record.TimeStep);
            _writer.Write(record.StressPeriod);
            _writer.Write(record.PeriodTime);
            _writer.Write(record.TotalTime);
            _writer.Write(charLabel);
            _writer.Write(ColumnCount);
            _writer.Write(RowCount);
            _writer.Write(record.Layer);

            // Write the array
            float[] buffer = dataArray.GetValues();
            _bfio.SaveInStream(buffer, _writer);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="record"></param>
        public void Write(LayerDataRecord<double> record)
        {
            if (record == null)
            { throw new ArgumentNullException("record"); }

            if (record.DataArray == null)
            { throw new ArgumentNullException("DataArray"); }

            Array2d<double> dataArray = record.DataArray;
            if (dataArray.RowCount != RowCount || dataArray.ColumnCount != ColumnCount)
            { throw new ArgumentException("The dimensions of the input array do not match the specified dimensions for this file."); }

            // Write header
            double periodTime = Convert.ToDouble(record.PeriodTime);
            double totalTime = Convert.ToDouble(record.TotalTime);
            char[] charLabel = record.Text.ToCharArray();
            if (charLabel.Length != 16)
            {
                throw new Exception("The Text value for the LayerDataRecordHeader is not 16 characters in length.");
            }
            _writer.Write(record.TimeStep);
            _writer.Write(record.StressPeriod);
            _writer.Write(periodTime);
            _writer.Write(totalTime);
            _writer.Write(charLabel);
            _writer.Write(ColumnCount);
            _writer.Write(RowCount);
            _writer.Write(record.Layer);

            // Write the array
            double[] buffer = dataArray.GetValues();
            _bfio.SaveInStream(buffer, _writer);
        }
        /// <summary>
        /// 
        /// </summary>
        public void Close()
        {
            Dispose();
        }
        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            // Delegate to the protected Dispose member
            Dispose(true);
            // Tell the garbage collector it does not need to call the finalizer method
            GC.SuppressFinalize(this);
        }

        #endregion

        #region Protected Members
        protected void Dispose(bool disposing)
        {
            try
            {
                if (_disposed == false && disposing == true)
                {
                    if (_writer != null)
                    { 
                        _writer.Close();
                        _writer = null;
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

        #region Finalizer
        ~BinaryLayerWriter()
        {
            Dispose(false);
        }
        #endregion

    }
}
