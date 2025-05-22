
using System;
using System.Collections;
using System.Data;
using System.Diagnostics;
using USGS.Puma.Core;

namespace USGS.Puma.Utilities
{
    public class MathUtility  
    { 
        public static void ComputeCosSin( double Angle, ref double AngleCos, ref double AngleSin, bool IsRadians ) 
        { 
            double rAngle = 0; 
            
            if ( IsRadians ) 
            { 
                rAngle = Angle; 
            } 
            else 
            { 
                rAngle = ToRadians( Angle ); 
            } 
           
            AngleCos = System.Math.Cos( rAngle ); 
            AngleSin = System.Math.Sin( rAngle ); 
            
        } 
        public static void FastForwardRotate( double AngleCos, double AngleSin, double XOrigin, double YOrigin, double X1, double Y1, ref double X2, ref double Y2 ) 
        { 
            X2 = ( X1 - XOrigin ) * AngleCos - ( Y1 - YOrigin ) * AngleSin + XOrigin; 
            Y2 = ( X1 - XOrigin ) * AngleSin + ( Y1 - YOrigin ) * AngleCos + YOrigin; 
        } 
        public static void FastBackwardRotate( double AngleCos, double AngleSin, double XOrigin, double YOrigin, double X1, double Y1, ref double X2, ref double Y2 ) 
        { 
            X2 = ( X1 - XOrigin ) * AngleCos + ( Y1 - YOrigin ) * AngleSin + XOrigin; 
            Y2 = -( X1 - XOrigin ) * AngleSin + ( Y1 - YOrigin ) * AngleCos + YOrigin; 
        } 
        public static double ToRadians( double AngleDeg ) 
        { 
            return AngleDeg * 0.0174532925199433;
        } 
        public static double ToDegrees( double AngleRad ) 
        { 
            return AngleRad * 57.2957795130823;
        } 
    } 
    
    
} 
