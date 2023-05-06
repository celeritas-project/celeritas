//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBremsstrahlungModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Muon Bremsstrahlung model interaction.
 */
class MuBremsstrahlungModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    MuBremsstrahlungModel(ActionId id,
                          ParticleParams const& particles,
                          SPConstImported data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void execute(CoreParams const&, StateHostRef&) const final;

    // Apply the interaction kernel on device
    void execute(CoreParams const&, StateDeviceRef&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "brems-muon"; }

    //! Name of the model, for user interaction
    std::string description() const final { return "Muon bremsstrahlung"; }

    //!@{
    //! Access model data
    MuBremsstrahlungData const& host_ref() const { return data_; }
    MuBremsstrahlungData const& device_ref() const { return data_; }
    //!@}

  private:
    MuBremsstrahlungData data_;
    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
