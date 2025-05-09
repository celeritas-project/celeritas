//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ImportedDataTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GlobalGeoTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ImportData;
struct PhysicsOptions;
struct ProcessBuilderOptions;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas tests using imported data.
 *
 * This is an implementation detail of GeantTestBase and RootTestBase.
 */
class ImportedDataTestBase : virtual public GlobalGeoTestBase
{
  public:
    //! Access lazily loaded problem-dependent data
    virtual ImportData const& imported_data() const = 0;

  protected:
    // Set up options for loading processes
    virtual ProcessBuilderOptions build_process_options() const;

    // Set up options for physics
    virtual PhysicsOptions build_physics_options() const;

    // Implemented overrides that load from import data
    SPConstMaterial build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle build_particle() override;
    SPConstCutoff build_cutoff() override;
    SPConstPhysics build_physics() override;
    SPConstSim build_sim() override;
    SPConstWentzelOKVI build_wentzel() override;
    SPConstCherenkov build_cherenkov() override;
    SPConstOpticalMaterial build_optical_material() override;
    SPConstOpticalPhysics build_optical_physics() override;
    SPConstScintillation build_scintillation() override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
