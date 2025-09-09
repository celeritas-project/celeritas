//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MollerBhabhaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Moller-Bhabha model interaction.
 */
class MollerBhabhaModel final : public Model, public StaticConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    MollerBhabhaModel(ActionId id, ParticleParams const& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    MollerBhabhaData const& host_ref() const { return data_; }
    MollerBhabhaData const& device_ref() const { return data_; }
    //!@}

  private:
    MollerBhabhaData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
