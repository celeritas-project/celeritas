//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
struct PhysicsParamsOptions;
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
    //!@{
    //! \name Type aliases
    using PhysicsOptions = PhysicsParamsOptions;
    //!@}

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
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
