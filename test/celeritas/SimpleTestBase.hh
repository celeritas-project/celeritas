//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/SimpleTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "GlobalTestBase.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Compton scattering with gammas in mock aluminum in a box in hard vacuum.
 *
 * The aluminum has a compton scattering cross section of 1.0 at 1 MeV, 100 at
 * 1e-4 MeV, and 1/E higher than 1 MeV. The detector is only 10 cm on a side.
 */
class SimpleTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! Type aliases
    using real_type = celeritas::real_type;
    //!@}

  protected:
    const char* geometry_basename() const override { return "two-boxes"; }

    virtual real_type secondary_stack_factor() const { return 2.0; }

    SPConstMaterial    build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle    build_particle() override;
    SPConstCutoff      build_cutoff() override;
    SPConstPhysics     build_physics() override;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
