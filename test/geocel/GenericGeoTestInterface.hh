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

#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"
#include "geocel/detail/LengthUnits.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct GenericGeoTrackingResult;
struct GenericGeoVolumeStackResult;
struct GenericGeoModelInp;

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
    using ModelInpResult = GenericGeoModelInp;
    using SPConstGeoInterface = std::shared_ptr<GeoParamsInterface const>;
    //!@}

  public:
    //!@{
    // Generate a track
    virtual TrackingResult track(Real3 const& pos_cm, Real3 const& dir) = 0;
    virtual TrackingResult
    track(Real3 const& pos_cm, Real3 const& dir, int max_step)
        = 0;
    //!@}

    //!@{
    //! Obtain the "touchable history" at a point
    virtual VolumeStackResult volume_stack(Real3 const& pos_cm) = 0;
    //!@}

    //! Get the model input from the geometry
    virtual ModelInpResult model_inp() const = 0;

    //! Get the label for this geometry: Geant4, VecGeom, ORANGE
    virtual std::string_view geometry_type() const = 0;

    //! Access the geometry interface, building if needed
    virtual SPConstGeoInterface geometry_interface() const = 0;

    // Get the basename or unique geometry key (defaults to suite name)
    virtual std::string geometry_basename() const;

    // Get the safety tolerance (defaults to SoftEq tol)
    virtual real_type safety_tol() const;

    // Get the threshold in "unit lengths" for a movement being a "bump"
    virtual real_type bump_tol() const;

    //! Unit length for "track" testing and other results
    virtual Constant unit_length() const { return lengthunits::centimeter; }

    //! Access the loaded geant4 world (if one exists)
    virtual G4VPhysicalVolume const* g4world() const { return nullptr; }

    // Get all logical volume names
    std::vector<std::string> get_volume_labels() const;

    // Get all physical volume names
    std::vector<std::string> get_volume_instance_labels() const;

    // Get mapped Geant4 physical volume names
    std::vector<std::string> get_g4pv_labels() const;

    // Get the volume name, adjusting for offsets from loading multiple geo
    std::string_view get_volume_name(VolumeId i) const;

  protected:
    // Virtual interface only
    ~GenericGeoTestInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
