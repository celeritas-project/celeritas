//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

// Source dependencies
#include "base/CollectionMirror.hh"
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
    using ParamsHostRef
        = celeritas::OrangeParamsData<celeritas::Ownership::const_reference,
                                      celeritas::MemSpace::host>;
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

  private:
    celeritas::CollectionMirror<celeritas::OrangeParamsData> params_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
