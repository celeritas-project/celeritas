//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/ICRU73QOModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/BraggICRU73QOData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the ICRU73QO ionization model interaction.
 */
class ICRU73QOModel final : public Model, public ConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    ICRU73QOModel(ActionId id, ParticleParams const& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    BraggICRU73QOData const& host_ref() const { return data_; }
    BraggICRU73QOData const& device_ref() const { return data_; }
    //!@}

  private:
    BraggICRU73QOData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
