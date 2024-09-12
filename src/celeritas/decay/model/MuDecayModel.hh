//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/model/MuDecayModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the muon decay interaction.
 */
class MuDecayModel : public Model, public ConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    inline MuDecayModel(ActionId id, ParticleParams const& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    MuDecayData const& host_ref() const { return data_; }
    MuDecayData const& device_ref() const { return data_; }
    //!@}

  private:
    MuDecayData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
