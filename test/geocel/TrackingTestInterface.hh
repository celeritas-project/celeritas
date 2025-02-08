//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TrackingTestInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct GenericGeoTrackingResult
{
    std::vector<std::string> volumes;
    std::vector<std::string> volume_instances;
    std::vector<real_type> distances;  //!< [cm]
    std::vector<real_type> halfway_safeties;  //!< [cm]

    void print_expected();
};

//---------------------------------------------------------------------------//
/*!
 * Access capabilities from any templated GenericGeo test.
 */
class GenericGeoTestInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TrackingResult = GenericGeoTrackingResult;
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

    //! Access the geometry interface, building if needed
    virtual SPConstGeoInterface geometry_interface() = 0;

  protected:
    // Virtual interface only
    ~GenericGeoTestInterface() = default;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
