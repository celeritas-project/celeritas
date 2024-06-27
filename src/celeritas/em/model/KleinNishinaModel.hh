//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/KleinNishinaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Klein-Nishina model interaction.
 */
class KleinNishinaModel final : public Model, public ConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    KleinNishinaModel(ActionId id, ParticleParams const& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    KleinNishinaData const& host_ref() const { return data_; }
    KleinNishinaData const& device_ref() const { return data_; }
    //!@}

  private:
    KleinNishinaData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
