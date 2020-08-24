//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef geometry_Types_hh
#define geometry_Types_hh

#include "base/OpaqueId.hh"

namespace celeritas
{
class Geometry;
//---------------------------------------------------------------------------//
using VolumeId = OpaqueId<Geometry, unsigned int>;

enum class Boundary : bool
{
  // Note: VecGeom has three boundary states defined: inside, onBoundary, outside.
  // Most times particle is inside NavState.Top() volume, and boundaries are treated
  // more carefully, therefore we set true=onBoundary, false otherwise.
  // onBoundary (within tolerance) with Outside.
  No  = false, // inside or outside
  Yes = true
};

enum class GeoTrackStatus : short
{
    Alive,
    Killed,
    InFlight,
    Boundary,
    ExitingSetup,
    Physics,
    Postponed,
    New
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include <ostream>

namespace std
{

  inline std::ostream&
  operator<<(std::ostream& os, const celeritas::Boundary& boundary)
  {
    using celeritas::Boundary;
    switch(boundary)
    {
       case Boundary::Yes: os << "ONBoundary";  break;
       case Boundary::No:  os << "offBoundary"; break;
    }
    return os;
  }

  inline std::ostream&
  operator<<(std::ostream& os, const celeritas::GeoTrackStatus& status)
  {
    using celeritas::GeoTrackStatus;
    switch(status)
    {
      case GeoTrackStatus::Alive: os << "Alive"; break;
      case GeoTrackStatus::Killed: os << "Killed"; break;
      case GeoTrackStatus::InFlight: os << "InFlight"; break;
      case GeoTrackStatus::Boundary: os << "Boundary"; break;
      case GeoTrackStatus::ExitingSetup: os << "ExitingSetup"; break;
      case GeoTrackStatus::Physics: os << "Physics"; break;
      case GeoTrackStatus::Postponed: os << "Postponed"; break;
      case GeoTrackStatus::New: os << "New"; break;
      default: os << "Unknown"; break;
    }
    return os;
  }
}  // namespace std

#endif // geometry_Types_hh
