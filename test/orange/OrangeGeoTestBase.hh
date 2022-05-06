//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

// Source dependencies
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "orange/Data.hh"
#include "orange/OrangeParams.hh"
#include "celeritas/Types.hh"

// Test dependencies
#include "gtest/Test.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test base for loading geometry.
 */
class OrangeGeoTestBase : public celeritas::Test
{
  public:
    //!@{
    //! Type aliases
    using real_type = celeritas::real_type;
    using size_type = celeritas::size_type;
    using Real3     = celeritas::Real3;
    using Sense     = celeritas::Sense;
    using VolumeId  = celeritas::VolumeId;
    using FaceId    = celeritas::FaceId;
    using SurfaceId = celeritas::SurfaceId;
    using Params    = celeritas::OrangeParams;

    using HostStateRef
        = celeritas::OrangeStateData<celeritas::Ownership::reference,
                                     celeritas::MemSpace::host>;
    //!@}

    //!@{
    //! On-the-fly construction inputs
    struct OneVolInput
    {
        bool complex_tracking = true;
    };

    struct TwoVolInput
    {
        real_type radius = 1;
    };
    //!@}

  public:
    // Convert a vector of senses to a string
    static std::string senses_to_string(celeritas::Span<const Sense> senses);

    // Default constructor
    OrangeGeoTestBase() = default;

    // Destructor
    ~OrangeGeoTestBase();

    // Load `test/orange/data/{filename}` JSON input
    void build_geometry(const char* filename);

    // Load geometry with one infinite volume
    void build_geometry(OneVolInput);

    // Load geometry with two volumes separated by a spherical surface
    void build_geometry(TwoVolInput);

    //! Get the data after loading
    const Params& params() const
    {
        CELER_EXPECT(params_);
        return *params_;
    }

    // Lazily create and get a single-serving host state
    const HostStateRef& host_state();

    //// QUERYING ////

    // Find the volume from its label (nullptr allowed)
    VolumeId find_volume(const char* label) const;

    // Find the surface from its label (NULL pointer allowed)
    SurfaceId find_surface(const char* label) const;

    // Surface name (or sentinel if no surface);
    std::string id_to_label(SurfaceId) const;

    // Cell name (or sentinel if no surface);
    std::string id_to_label(VolumeId) const;

    // Print geometry description
    void describe(std::ostream& os) const;

    //! Number of volumes
    VolumeId::size_type num_volumes() const
    {
        CELER_EXPECT(params_);
        return params_->num_volumes();
    }

  private:
    //// TYPES ////

    using HostStateStore
        = celeritas::CollectionStateStore<celeritas::OrangeStateData,
                                          celeritas::MemSpace::host>;

    //// DATA ////

    // Param data
    std::unique_ptr<Params> params_;

    // State data
    HostStateStore host_state_;

    //// HELPER FUNCTIONS ////

    void resize_state_storage();
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
