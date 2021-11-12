//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

// Source dependencies
#include "base/CollectionMirror.hh"
#include "base/Span.hh"
#include "orange/Data.hh"

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
    using Sense     = celeritas::Sense;
    using VolumeId  = celeritas::VolumeId;
    using SurfaceId = celeritas::SurfaceId;
    using ParamsHostRef
        = celeritas::OrangeParamsData<celeritas::Ownership::const_reference,
                                      celeritas::MemSpace::host>;
    using ParamsDeviceRef
        = celeritas::OrangeParamsData<celeritas::Ownership::const_reference,
                                      celeritas::MemSpace::device>;
    //!@}

    //!@{
    //! On-the-fly construction inputs
    struct OneVolInput
    {
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
    const ParamsHostRef& params_host_ref() const
    {
        CELER_EXPECT(params_);
        return params_.host();
    }

    //! Get device data after loading
    const ParamsDeviceRef& params_device_ref() const
    {
        CELER_EXPECT(params_);
        return params_.device();
    }

    //! Access the shared CPU storage space for senses
    celeritas::Span<Sense> sense_storage()
    {
        return celeritas::make_span(sense_storage_);
    }

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

  private:
    //// TYPES ////
    using ParamsHostValue
        = celeritas::OrangeParamsData<celeritas::Ownership::value,
                                      celeritas::MemSpace::host>;

    //// DATA ////
    celeritas::CollectionMirror<celeritas::OrangeParamsData> params_;

    std::vector<Sense>                         sense_storage_;
    std::vector<std::string>                   surf_names_;
    std::vector<std::string>                   vol_names_;
    std::unordered_map<std::string, VolumeId>  vol_ids_;
    std::unordered_map<std::string, SurfaceId> surf_ids_;

    //// METHODS ////
    void build_impl(ParamsHostValue&& params);
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
