//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "geocel/GeoTrackInterface.hh"
#include "geocel/Types.hh"

#include "CheckedGeoTrackView.hh"
#include "LazyGeantGeoManager.hh"
#include "UnitUtils.hh"

class G4VPhysicalVolume;

namespace celeritas
{
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
 * \todo This is being refactored into GenericGeoTestBase so that we can just
 * use the GeoTrackInterface and GeoParamsInterface wrappers.
 */
class GenericGeoTestInterface : public LazyGeantGeoManager
{
  public:
    //!@{
    //! \name Type aliases
    using TrackingResult = GenericGeoTrackingResult;
    using TrackingTol = GenericGeoTrackingTolerance;
    using VolumeStackResult = GenericGeoVolumeStackResult;
    using GeoTrackView = GeoTrackInterface<real_type>;
    using UPGeoTrack = std::unique_ptr<GeoTrackView>;
    //!@}

  public:
    //// TESTS ////

    // Track until exiting the geometry
    TrackingResult track(Real3 const& pos_cm,
                         Real3 const& dir,
                         TrackingTol const& tol,
                         int max_steps);

    // Track until exiting the geometry (default test tol)
    TrackingResult
    track(Real3 const& pos_cm, Real3 const& dir, int max_steps = 50);

    // Obtain the "touchable history" at a point
    VolumeStackResult volume_stack(Real3 const& pos_cm);

    //// BASE INTERFACE ////

    // Default to using test suite name
    std::string_view gdml_basename() const override;

    //// PURE INTERFACE ////

    //! Get the label for this geometry: Geant4, VecGeom, ORANGE
    virtual std::string_view geometry_type() const = 0;

    //! Access the geometry interface
    virtual SPConstGeoI geometry_interface() const = 0;

    //! Create a track view (TODO: replace geo test base view)
    virtual UPGeoTrack make_geo_track_view_interface() = 0;

    // Create a checked track view
    CheckedGeoTrackView make_checked_track_view();

    //// CONFIGURABLE INTERFACE ////

    // Unit length for "track" testing and other results (defaults to cm)
    virtual UnitLength unit_length() const;

    // Maximum number of local track slots
    virtual size_type num_track_slots() const;

    // Whether surface normals work for the current geometry/test
    virtual bool supports_surface_normal() const;

    // Get the safety tolerance (defaults to SoftEq tol) for tracking result
    virtual GenericGeoTrackingTolerance tracking_tol() const;

    //// UTILITIES ////

    // Construct an initializer with correct scaling/normalization
    GeoTrackInitializer
    make_initializer(Real3 const& pos_unit, Real3 const& dir) const;
    //! Get the name of the current volume
    std::string volume_name(GeoTrackView const& geo) const;
    //! Get the stack of volume instances
    std::string unique_volume_name(GeoTrackView const& geo) const;

  protected:
    // Virtual interface only
    ~GenericGeoTestInterface() = default;

  private:
    // Volume params, possibly not from G4
    SPConstVolumes const& get_test_volumes() const;
    // Lazily constructed volumes, possibly from non-G4 model
    mutable SPConstVolumes volumes_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
