//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "geocel/Types.hh"
#include "geocel/detail/LengthUnits.hh"

class G4VPhysicalVolume;

namespace celeritas
{
class GeoParamsInterface;
class VolumeParams;

namespace test
{
//---------------------------------------------------------------------------//
struct GenericGeoTrackingResult;
struct GenericGeoTrackingTolerance;
struct GenericGeoVolumeStackResult;

//---------------------------------------------------------------------------//
/*!
 * Access capabilities from any templated GenericGeo test.
 *
 * The volume/instance offsets are usually used with Geant4, which has volume
 * IDs that may not start with zero if the problem has been reinitaialized.
 * (This is because geant4 uses global static integers for counting.) It can
 * also be used in other circumstances (vecgeom internal construction) where
 * fake volumes/instances are inserted before the "real" volumes/instances.
 */
class GenericGeoTestInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TrackingResult = GenericGeoTrackingResult;
    using VolumeStackResult = GenericGeoVolumeStackResult;
    //!@}

  public:
    //! Generate a track
    virtual TrackingResult track(Real3 const& pos_cm, Real3 const& dir) = 0;

    //! Get the safety tolerance (defaults to SoftEq tol) for tracking result
    virtual GenericGeoTrackingTolerance tracking_tol() const;

    //!@{
    //! Obtain the "touchable history" at a point
    virtual VolumeStackResult volume_stack(Real3 const& pos_cm) = 0;
    //!@}

    //! Get the label for this geometry: Geant4, VecGeom, ORANGE
    virtual std::string_view geometry_type() const = 0;

    //! Access the geometry interface
    virtual GeoParamsInterface const& geometry_interface() const = 0;

    // Get the basename or unique geometry key
    virtual std::string_view gdml_basename() const = 0;

    // Whether surface normals work for the current geometry/test
    virtual bool supports_surface_normal() const;

    // Get the threshold in "unit lengths" for a movement being a "bump"
    virtual real_type bump_tol() const;

    //! Unit length for "track" testing and other results
    virtual Constant unit_length() const { return lengthunits::centimeter; }

  protected:
    // Virtual interface only
    ~GenericGeoTestInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
