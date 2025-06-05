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

// DEPRECATED: remove in v0.7
#define EXPECT_RESULT_EQ(EXPECTED, ACTUAL) EXPECT_REF_EQ(EXPECTED, ACTUAL)
#define EXPECT_RESULT_NEAR(EXPECTED, ACTUAL, TOL) \
    EXPECT_REF_NEAR(EXPECTED, ACTUAL, TOL)

namespace celeritas
{
namespace test
{

class GenericGeoTestInterface;

//---------------------------------------------------------------------------//
struct GenericGeoTrackingResult
{
    std::vector<std::string> volumes;
    std::vector<std::string> volume_instances;
    std::vector<real_type> distances;  //!< [cm]
    std::vector<real_type> halfway_safeties;  //!< [cm]
    // Locations the particle had a very tiny distance in a volume
    std::vector<real_type> bumps;  //!< [cm * 3]

    void print_expected();
};

struct GenericGeoTrackingTolerance
{
    real_type distance{0};
    real_type safety{0};

    static GenericGeoTrackingTolerance
    from_test(GenericGeoTestInterface const&);
};

//---------------------------------------------------------------------------//
struct GenericGeoVolumeStackResult
{
    std::vector<std::string> volume_instances;
    std::vector<int> replicas;

    void print_expected();
};

//---------------------------------------------------------------------------//
::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   char const* tol_expr,
                                   GenericGeoTrackingResult const& expected,
                                   GenericGeoTrackingResult const& actual,
                                   GenericGeoTrackingTolerance const& tol);

//---------------------------------------------------------------------------//
inline ::testing::AssertionResult
IsRefEq(char const* expected_expr,
        char const* actual_expr,
        GenericGeoTrackingResult const& expected,
        GenericGeoTrackingResult const& actual)
{
    return IsRefEq(expected_expr, actual_expr, "default", expected, actual, {});
}

//---------------------------------------------------------------------------//
::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   GenericGeoVolumeStackResult const& expected,
                                   GenericGeoVolumeStackResult const& actual);

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
    // Obtain the "touchable history" at a point
    virtual VolumeStackResult volume_stack(Real3 const& pos_cm) = 0;
    //!@}

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
