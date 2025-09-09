//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheHeitlerModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/em/data/BetheHeitlerData.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Bethe-Heitler model interaction.
 */
class BetheHeitlerModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    BetheHeitlerModel(ActionId id,
                      ParticleParams const& particles,
                      SPConstImported data,
                      bool enable_lpm);

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
    BetheHeitlerData const& host_ref() const { return data_; }
    BetheHeitlerData const& device_ref() const { return data_; }
    //!@}

  private:
    BetheHeitlerData data_;
    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
