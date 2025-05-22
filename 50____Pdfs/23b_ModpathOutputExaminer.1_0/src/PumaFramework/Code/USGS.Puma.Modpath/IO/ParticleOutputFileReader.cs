using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public class ParticleOutputFileReader : IDisposable
    {
        #region Private Fields
        protected StreamReader _Reader = null;
        protected bool _IsDisposed = false;
        #endregion

        #region Public Methods
        protected bool _Valid = false;
        public bool Valid
        {
            get { return _Valid; }
        }

        public void Close()
        {
            Dispose();
        }

        #endregion

        #region Private and Protected Methods
        protected void Dispose(bool disposeManagedObjs)
        {
            if (!_IsDisposed)
            {
                try
                {
                    if (disposeManagedObjs)
                    {
                        if (_Reader != null)
                        {
                            _Reader.Close();
                        }
                        GC.SuppressFinalize(this);
                    }
                }
                catch (Exception)
                {
                    _IsDisposed = false;
                    throw;
                }

                _IsDisposed = true;
            }
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            Dispose(true);
        }

        #endregion
    }
}
