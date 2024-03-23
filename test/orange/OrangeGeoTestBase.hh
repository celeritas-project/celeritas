//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "orange/OrangeData.hh"
#include "celeritas/Types.hh"

#include "OrangeTestBase.hh"
#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct UnitInput;
class OrangeParams;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test base for loading geometry.
 */
class OrangeGeoTestBase : public OrangeTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using HostStateRef = HostRef<OrangeStateData>;
    using HostParamsRef = HostCRef<OrangeParamsData>;
    using Params = OrangeParams;
    using SPConstParams = std::shared_ptr<OrangeParams const>;
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

    // Load `test/orange/data/{filename}` JSON input
    void build_geometry(std::string const& filename);

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

    // Access the host data
    HostParamsRef const& host_params() const;

    //// QUERYING ////

    // Find the volume from its label (nullptr allowed)
    VolumeId find_volume(std::string const& label) const;

    // Find the surface from its label (NULL pointer allowed)
    SurfaceId find_surface(std::string const& label) const;

    // Surface name (or sentinel if no surface)
    std::string id_to_label(UniverseId uid, LocalSurfaceId surfid) const;

    // Surface name (or sentinel if no surface) within UniverseId{0}
    std::string id_to_label(LocalSurfaceId surfid) const;

    // Cell name (or sentinel if no surface)
    std::string id_to_label(UniverseId uid, LocalVolumeId volid) const;

    // Cell name (or sentinel if no surface) within UniverseId{0}
    std::string id_to_label(LocalVolumeId volid) const;

    // Print geometry description
    void describe(std::ostream& os) const;

    // Number of volumes
    VolumeId::size_type num_volumes() const;

    //// GenericGeoTestBase ////

    // Return the geometry that was created
    SPConstGeo build_geometry() final;

  private:
    //// TYPES ////

    using HostStateStore
        = CollectionStateStore<OrangeStateData, MemSpace::host>;

    //// DATA ////

    // Param data
    SPConstParams params_;

    // State data
    HostStateStore host_state_;

    //// HELPER FUNCTIONS ////

    void resize_state_storage();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
