//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagatorHandler.hh
//---------------------------------------------------------------------------//
/**
 * @brief Interface between the scheduler and the field integrator.
 *
 * Originated from GeantV
 */
//===----------------------------------------------------------------------===//
#pragma once

#include "base/SystemOfUnits.hh"
#include "geometry/GeoTrackView.hh"
//#include "geometry/track/GeoTrackState.hh"

namespace celeritas {

class FieldLookup 
{
public:
  //using ThreeVector_t            = vecgeom::Vector3D<double>;
  using ThreeVector_t = Real3;

  static constexpr double _bmag          = 3 * units::tesla;
  static constexpr ThreeVector_t _bfield = {0.0, 0.0, _bmag};

  // CELER_FUNCTION
  // static void GetFieldValue(const ThreeVector_t &pos, ThreeVector_t &magFld, double &bmag) {
  //   bmag   = _bmag;
  //   magFld = _bfield;
  // }

  CELER_FORCEINLINE_FUNCTION
  static void GetFieldValue(const ThreeVector_t &pos, ThreeVector_t &magFld, double &bmag) {
    bmag   = _bmag;
    magFld = _bfield; //.x(), _bfield.y(), _bfield.z()};
  }
};

class FieldPropagationHandler
{
public:
   //using ThreeVector_t     = vecgeom::Vector3D<double>;
   using ThreeVector_t       = Real3;

   FieldPropagationHandler() = default;
   ~FieldPropagationHandler() = default;


   CELER_FUNCTION
   double Curvature(const GeoTrackView &track) const;

   CELER_FUNCTION
   double Curvature(const GeoTrackView &track, const ThreeVector_t &magFld, double bmag) const;

   CELER_FUNCTION
   bool Propagate(GeoTrackView &track) const;

   CELER_FUNCTION
   void PropagateInVolume(GeoTrackView &track, double crtstep, const ThreeVector_t &BfieldInitial, 
                          double bmag) const;

   // CELER_FUNCTION
   // bool IsSameLocation(GeoTrackView &track) const;

   CELER_FORCEINLINE_FUNCTION
   double SafeLength(const GeoTrackView &track, double eps, const ThreeVector_t &magFld, double bmag) const
   {
      // Returns the propagation length in field such that the propagated point is
      // shifted less than eps with respect to the linear propagation.
      // OLD: return 2. * sqrt(eps / track.Curvature(Bz));
      double c   = Curvature(track, magFld, bmag); //, td);
      double val = 0.0;
      // if (c < 1.E-10) { val= 1.E50; } else
      val = 2. * sqrt(eps / c);
      return val;
   }


// #ifndef __NVCC__
//    void PrintStats() const;
// #endif

   CELER_FUNCTION
   void CheckTrack(GeoTrackView &track, const char *msg, double epsilon = 1.0e-5) const;
};

} // namespace celeritas

#include "FieldPropagationHandler.i.hh"
