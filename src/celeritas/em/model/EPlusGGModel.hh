//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/EPlusGGModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/executor/EPlusGGExecutor.hh"  // IWYU pragma: associated
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch two-gamma positron annihiliation.
 */
class EPlusGGModel final : public Model, public StaticConcreteAction
{
  public:
    // Construct from model ID and other necessary data
    EPlusGGModel(ActionId id, ParticleParams const& particles);

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
    EPlusGGData const& host_ref() const { return data_; }
    EPlusGGData const& device_ref() const { return data_; }
    //!@}

  private:
    EPlusGGData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
