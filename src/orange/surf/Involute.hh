//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/InvoluteSolver.hh" 

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Involute:
 * 
 * \f[
    x = r_b * (cos(t+a) + tsin(t+a)) + x_0
    y = r_b * (sin(t+a) - tcos(t+a)) + y_0   
   \f]

 * where \em t is the normal angle of the tangent to the circle of involute with
 * radius \em r_b from a starting angle of \em a (\f$r/h\f$ for a finite cone.
 */
class Involute
{
  public:
     //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 3>;
    using StorageSpan = Span<real_type const, 4>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::inv;
    }

    // Construct with square of tangent for simplification
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////
    // Construct at (0,0,z)
    static Involute at0origin(Real3 const& origin, real_type radius, 
                              real_type a, real_type sign, 
                              real_type tmin, real_type tmax);

    // Construct from origin and radius of circle of involute
    inline CELER_FUNCTION Involute(Real3 const& origin, real_type radius, 
                                   real_type a, real_type sign, 
                                   real_type tmin, real_type tmax);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Involute(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Get the origin position
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Get involute parameters
    CELER_FUNCTION real_type r_b() const { return r_b_; }
    CELER_FUNCTION real_type a() const { return a_; }
    CELER_FUNCTION real_type sign() const { return sign_; }

    //! Get bounds of the involute
    CELER_FUNCTION real_type tmin() const { return tmin_; }
    CELER_FUNCTION real_type tmax() const { return tmax_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_[0], 4}; }

    // // Helper function to get the origin as a 3-vector
    // Real3 calc_origin() const;

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Location of the center of circle of involute
    Real3 origin_;

    // Involute parameters
    real_type r_b_;
    real_type a_;
    real_type sign_;

    // Bounds
    real_type tmin_;
    real_type tmax_;


    //! Private default constructor for manual construction
    Involute() = default;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct from origin and radius.
 */
CELER_FUNCTION Involute::Involute(Real3 const& origin, real_type radius, 
                                  real_type a, real_type sign, 
                                  real_type tmin, real_type tmax)
    : origin_(origin), r_b_(radius), a_(a) , sign_(sign) , tmin_(tmin) , tmax_(tmax)
{
    const double pi = 3.14159265358979323846;
    CELER_EXPECT(radius > 0);
    CELER_EXPECT(a > 0);
    CELER_EXPECT(abs(tmax) < 2*pi+abs(tmin));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION Involute::Involute(Span<R, StorageSpan::extent> data)
    : origin_{data[0], data[1], data[2]}, r_b_{data[3]}, a_{data[4]}, 
      sign_{data[5]}, tmin_{data[6]}, tmax_{data[7]} 
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense Involute::calc_sense(Real3 const& pos) const
{   
    const double pi = 3.14159265358979323846;

    /*
     * Define tolerances.
     */
    real_type const tol = 1e-7;

    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];
    
    /*
     * Calculate distance to origin and obtain t value for disance.
     */
    real_type const rxy2 = pow(2.0,x) + pow(2.0,y);
    real_type const tPoint2 = (rxy2/pow(2.0,r_b_))-1;
    real_type const tPoint = sqrt(tPoint2);

     /*
      * Check if Point is on involute.
      */
    real_type const angle = tPoint + a_;
    real_type const xInv = r_b_ * (std::cos(angle) + tPoint * std::sin(angle));
    real_type const yInv = r_b_ * (std::sin(angle) - tPoint * std::cos(angle));

    if (abs(x-xInv) < tol && abs(y-yInv) < tol) { 
       return real_to_sense(0);
     }

    /*
     * Check if point is in defined bounds. 
     */
    if (abs(tPoint2) < abs(pow(2.0,tmin_)) - tol) {
       return real_to_sense(1);
    }
    if (abs(tPoint2) > abs(pow(2.0,tmax_)) + tol) {
       return real_to_sense(1);
    }

    

    /*
     * Check if point is inside involute
     */

    // Calculate tangents
    real_type a = rxy2;
    real_type b;
    real_type c;
    real_type xa, xb;
    real_type ya, yb, yc, yd;
    real_type yalpha, ybeta;

    b = -2 * pow(2.0,r_b_) * x;
    c = pow(4.0,r_b_) - pow(2.0,y) * pow(2.0,r_b_);

    xa = (-b + sqrt(pow(2.0,b)-4*a*c))/(2*a);
    xb = (-b - sqrt(pow(2.0,b)-4*a*c))/(2*a);
    
    ya = sqrt(pow(2.0,r_b_)-pow(2.0,xa));
    yb = -sqrt(pow(2.0,r_b_)-pow(2.0,xa));
    yc = sqrt(pow(2.0,r_b_)-pow(2.0,xb));
    yd = -sqrt(pow(2.0,r_b_)-pow(2.0,xb));

    b =  -2 * pow(2.0,r_b_) * y;
    c = pow(4.0,r_b_) - pow(2.0,x) * pow(2.0,r_b_);

    yalpha = (-b + sqrt(pow(2.0,b)-4*a*c))/(2*a);
    ybeta = (-b - sqrt(pow(2.0,b)-4*a*c))/(2*a);

    Array<real_type, 2> point1;
    Array<real_type, 2> point2;

    // First tangent
    if (abs(ya-yalpha) <= tol) {
        point1 = { xa, ya };
    } else {
        point1 = { xb, yc };
    }
    // Second tangent
    if (abs(yb-ybeta) <= tol) {
        point1 = { xa, yb };
    } else {
        point1 = { xb, yd };
    }

    // Determine which tangent
    Array<real_type, 2> point;
    Array<real_type, 2> norm = { -y, x };
    real_type dot = point1[0] * norm[0] + point1[1] * norm[1];
    if ( ((0<dot) - (dot<0)) == sign_) {
        point = point1;
    } else {
        point = point2;
    }

    // Calculate angle of tangent
    real_type theta = std::acos(point[0]/sqrt(pow(2.0,point[0])+pow(2.0,point[1])));
    if (point[1] < 0) {
        theta = (pi - theta) + pi;
    }

    while (abs(theta) < (abs(tmin_ + a_) - tol)) {
        theta += pi*2*sign_;
    }

    if (abs(theta) < abs(tmax_ + a_) + tol) {
        return real_to_sense(-1);
    }

    return real_to_sense(1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto Involute::calc_intersections(Real3 const& pos,
                                              Real3 const& dir,
                                              SurfaceState on_surface) const
    -> Intersections
{
     // Expand translated positions into 'xyz' coordinate system
    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];
    real_type const z = pos[2] - origin_[2];

    real_type const u = dir[0];
    real_type const v = dir[1];
    real_type const w = dir[2];

    detail::InvoluteSolver solve(r_b_, a_, sign_, tmin_, tmax_);

    return solve(x, y, z, u, v, w);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position on the surface.
 */
CELER_FORCEINLINE_FUNCTION Real3 Involute::calc_normal(Real3 const& pos) const
{
    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];

    /*
     * Calculate distance to origin and obtain t value for disance.
     */
    real_type const rxy2 = pow(2.0,x) + pow(2.0,y);
    real_type const tPoint = sqrt((rxy2/pow(2.0,r_b_))-1) * sign_;

    /*
     * Calculate normal
     */ 
    real_type const angle = tPoint + a_;
    Real3 normal_ = {std::sin(angle), -std::cos(angle), 0};

    return normal_;
}
}