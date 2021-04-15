//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include <functional>
#include <vector>
#include "physics/base/Model.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Mock process.
 *
 * Multiple instances of this process can be created to test out the physics.
 * The value grids are all parameterized:
 * - Cross section is scaled by the material's atomic number density, and is
 *   constant with energy.
 * - Energy loss rate is also constant with energy and scales with the number
 *   density.
 * - Range is determined by the energy loss rate -- constant energy loss rate k
 *   means linear range of E/k.
 *
 * The given applicability vector has one element per model that it will
 * create. Each model can have a different particle type and/or energy range.
 */
class MockProcess : public celeritas::Process
{
  public:
    //!@{
    //! Type aliases
    using real_type        = celeritas::real_type;
    using BarnMicroXs      = celeritas::Quantity<celeritas::units::Barn>;
    using Applicability    = celeritas::Applicability;
    using ModelIdGenerator = celeritas::ModelIdGenerator;
    using VecApplicability = std::vector<Applicability>;
    using VecMicroXs       = std::vector<BarnMicroXs>;
    using SPConstMaterials = std::shared_ptr<const celeritas::MaterialParams>;
    using ModelCallback    = std::function<void(celeritas::ModelId)>;
    //!@}

    struct Input
    {
        SPConstMaterials materials;
        std::string      label;
        VecApplicability applic;        //!< Applicablity per model
        ModelCallback    interact;      //!< MockModel::interact callback
        VecMicroXs       xs;            //!< Constant per atom [bn]
        real_type        energy_loss{}; //!< Constant per atom [MeV/cm / cm^-3]
    };

  public:
    explicit MockProcess(Input data);

    VecModel          build_models(ModelIdGenerator next_id) const final;
    StepLimitBuilders step_limits(Applicability range) const final;
    std::string       label() const final;

  private:
    Input data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
