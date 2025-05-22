using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class LayerDataRecord<T>
    {
        public LayerDataRecord()
        {
            Layer = 0;
            StressPeriod = 0;
            TimeStep = 0;
            Text = "";
            DataArray = null;
            PeriodTime = 0f;
            TotalTime = 0f;
        }
        public LayerDataRecord(int layer, int stressPeriod, int timeStep, string text, float periodTime, float totalTime, Array2d<T> dataArray)
        {
            Layer = layer;
            StressPeriod = stressPeriod;
            TimeStep = timeStep;
            Text = text;
            DataArray = dataArray;
            PeriodTime = periodTime;
            TotalTime = totalTime;
        }

        private int _Layer = 0;
        /// <summary>
        /// 
        /// </summary>
        public int Layer
        {
            get { return _Layer; }
            set { _Layer = value; }
        }

        private int _StressPeriod = 0;
        /// <summary>
        /// 
        /// </summary>
        public int StressPeriod
        {
            get { return _StressPeriod; }
            set { _StressPeriod = value; }
        }

        private int _TimeStep = 0;
        /// <summary>
        /// 
        /// </summary>
        public int TimeStep
        {
            get { return _TimeStep; }
            set { _TimeStep = value; }
        }

        private float _PeriodTime = 0;
        /// <summary>
        /// 
        /// </summary>
        public float PeriodTime
        {
            get { return _PeriodTime; }
            set { _PeriodTime = value; }
        }

        private float _TotalTime = 0;
        /// <summary>
        /// 
        /// </summary>
        public float TotalTime
        {
            get { return _TotalTime; }
            set { _TotalTime = value; }
        }

        private string _Text;
        /// <summary>
        /// 
        /// </summary>
        public string Text
        {
            get { return _Text; }
            set
            {
                _Text = "";
                if ( !string.IsNullOrEmpty(value)) _Text = value;
                if (_Text.Length > 16) _Text = _Text.Substring(0, 16);
            }
        }

        private Array2d<T> _DataArray = null;
        /// <summary>
        /// 
        /// </summary>
        public Array2d<T> DataArray
        {
            get { return _DataArray; }
            set { _DataArray = value; }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="writer"></param>
        public void Write(System.Xml.XmlWriter writer)
        {
            Write(writer, "LayerDataRecord");
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="elementName"></param>
        public void Write(System.Xml.XmlWriter writer, string elementName)
        {
            // Write start tag of root element
            writer.WriteStartElement(elementName);

            // Write layer, stress period, time step, and text as attributes
            writer.WriteElementString("Text", this.Text);
            writer.WriteElementString("Layer", this.Layer.ToString());
            writer.WriteElementString("StressPeriod", this.StressPeriod.ToString());
            writer.WriteElementString("TimeStep", this.TimeStep.ToString());
            writer.WriteElementString("PeriodTime", this.PeriodTime.ToString());
            writer.WriteElementString("TotalTime", this.TotalTime.ToString());
            writer.WriteElementString("Rows", this.DataArray.RowCount.ToString());
            writer.WriteElementString("Columns", this.DataArray.ColumnCount.ToString());

            // Write the data array
            this.DataArray.Write(writer, "Buffer", false);
            
            // Close the root element
            writer.WriteEndElement();

        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public LayerDataRecord<float> CopyAsSingle()
        {
            LayerDataRecord<float> rec = new LayerDataRecord<float>();
            rec.Layer = this.Layer;
            rec.StressPeriod = this.StressPeriod;
            rec.PeriodTime = this.PeriodTime;
            rec.TimeStep = this.TimeStep;
            rec.TotalTime = this.TotalTime;
            rec.Text = this.Text;

            if (this.GetType() == typeof(LayerDataRecord<float>))
            {
                Array2d<float> buffer = new Array2d<float>(this.DataArray.RowCount, this.DataArray.ColumnCount);
                Array2d<float> dataArray = this.DataArray as Array2d<float>;
                for (int row = 1; row <= this.DataArray.RowCount; row++)
                {
                    for (int column = 1; column <= this.DataArray.ColumnCount; column++)
                    {
                        buffer[row, column] = dataArray[row, column];
                    }
                }
                rec.DataArray = buffer;
                return rec;
            }
            else if (this.GetType() == typeof(LayerDataRecord<double>))
            {
                Array2d<float> buffer = new Array2d<float>(this.DataArray.RowCount, this.DataArray.ColumnCount);
                Array2d<double> dataArray = this.DataArray as Array2d<double>;
                for (int row = 1; row <= this.DataArray.RowCount; row++)
                {
                    for (int column = 1; column <= this.DataArray.ColumnCount; column++)
                    {
                        buffer[row, column] = Convert.ToSingle(dataArray[row, column]);
                    }
                }
                rec.DataArray = buffer;
                return rec;
            }
            else
            {
                // This condition should not be possible
                return null;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public LayerDataRecord<double> CopyAsDouble()
        {
            LayerDataRecord<double> rec = new LayerDataRecord<double>();
            rec.Layer = this.Layer;
            rec.StressPeriod = this.StressPeriod;
            rec.PeriodTime = this.PeriodTime;
            rec.TimeStep = this.TimeStep;
            rec.TotalTime = this.TotalTime;
            rec.Text = this.Text;

            if (this.GetType() == typeof(LayerDataRecord<double>))
            {
                Array2d<double> buffer = new Array2d<double>(this.DataArray.RowCount, this.DataArray.ColumnCount);
                Array2d<double> dataArray = this.DataArray as Array2d<double>;
                for (int row = 1; row <= this.DataArray.RowCount; row++)
                {
                    for (int column = 1; column <= this.DataArray.ColumnCount; column++)
                    {
                        buffer[row, column] = dataArray[row, column];
                    }
                }
                rec.DataArray = buffer;
                return rec;
            }
            else if (this.GetType() == typeof(LayerDataRecord<float>))
            {
                Array2d<double> buffer = new Array2d<double>(this.DataArray.RowCount, this.DataArray.ColumnCount);
                Array2d<float> dataArray = this.DataArray as Array2d<float>;
                for (int row = 1; row <= this.DataArray.RowCount; row++)
                {
                    for (int column = 1; column <= this.DataArray.ColumnCount; column++)
                    {
                        buffer[row, column] = Convert.ToDouble(dataArray[row, column]);
                    }
                }
                rec.DataArray = buffer;
                return rec;
            }
            else
            {
                // This condition should not be possible
                return null;
            }

        }
    }
}
