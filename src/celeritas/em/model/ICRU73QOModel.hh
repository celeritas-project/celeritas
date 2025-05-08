//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/ICRU73QOModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/MuHadIonizationData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the ICRU73QO ionization model interaction.
 */
class ICRU73QOModel final : public Model, public StaticConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    ICRU73QOModel(ActionId, ParticleParams const&, SetApplicability);

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
    MuHadIonizationData const& host_ref() const { return data_; }
    MuHadIonizationData const& device_ref() const { return data_; }
    //!@}

  private:
    // Particle types and energy ranges that this model applies to
    SetApplicability applicability_;
    // Model data
    MuHadIonizationData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
