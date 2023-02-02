//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "celeritas/Types.hh"

// Test dependencies
#include "Test.hh"

namespace celeritas
{
struct UnitInput;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test base for loading geometry.
 */
class OrangeGeoTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using HostStateRef = OrangeStateData<Ownership::reference, MemSpace::host>;
    using Params = OrangeParams;
    //!@}

    //!@{
    //! On-the-fly construction inputs
    struct OneVolInput
    {
        bool complex_tracking = false;
    };

    struct TwoVolInput
    {
        real_type radius = 1;
    };
    //!@}

  public:
    // Convert a vector of senses to a string
    static std::string senses_to_string(Span<Sense const> senses);

    // Convert a string to a sense vector
    static std::vector<Sense> string_to_senses(std::string const& s);

    // Default constructor
    OrangeGeoTestBase() = default;

    // Destructor
    ~OrangeGeoTestBase();

    // Load `test/orange/data/{filename}` JSON input
    void build_geometry(char const* filename);

    // Load geometry with one infinite volume
    void build_geometry(OneVolInput);

    // Load geometry with two volumes separated by a spherical surface
    void build_geometry(TwoVolInput);

    // Load geometry from a single unit
    void build_geometry(UnitInput);

    //! Get the data after loading
    Params const& params() const
    {
        CELER_EXPECT(params_);
        return *params_;
    }

    // Lazily create and get a single-serving host state
    HostStateRef const& host_state();

    //// QUERYING ////

    // Find the volume from its label (nullptr allowed)
    VolumeId find_volume(char const* label) const;

    // Find the surface from its label (NULL pointer allowed)
    SurfaceId find_surface(char const* label) const;

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
        = CollectionStateStore<OrangeStateData, MemSpace::host>;

    //// DATA ////

    // Param data
    std::unique_ptr<Params> params_;

    // State data
    HostStateStore host_state_;

    //// HELPER FUNCTIONS ////

    void resize_state_storage();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
