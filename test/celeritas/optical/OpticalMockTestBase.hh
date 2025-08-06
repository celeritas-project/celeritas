//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/GlobalTestBase.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Class containing mock test data for optical physics.
 */
class OpticalMockTestBase : public GlobalTestBase
{
  public:
    // Construct optical material parameters from mock data
    SPConstOpticalMaterial build_optical_material() override;

    // Construct (core) material parameters from mock data
    SPConstMaterial build_material() override;

    // Access mock imported data
    ImportData const& imported_data() const;

    // Retrieve imported optical model data by class
    ImportOpticalModel const& import_model_by_class(ImportModelClass) const;

    //! Number of mock optical materials
    OptMatId::size_type num_optical_materials() const
    {
        return this->imported_data().optical_materials.size();
    }

    std::string_view gdml_basename() const override
    {
        CELER_ASSERT_UNREACHABLE();
    }

    //!@{
    //! \name Unsupported params builders
    SPConstCoreGeo build_geometry() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstParticle build_particle() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCutoff build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstSim build_sim() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstWentzelOKVI build_wentzel() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCherenkov build_cherenkov() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstOpticalPhysics build_optical_physics() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstScintillation build_scintillation() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    //!@}

  private:
    // Construct mock import data in place
    void build_import_data(ImportData&) const;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
