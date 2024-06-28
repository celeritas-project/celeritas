//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CoulombScatteringModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;
class IsotopeView;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Wentzel Coulomb scattering model interaction.
 */
class CoulombScatteringModel final : public Model, public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    CoulombScatteringModel(ActionId id,
                           ParticleParams const& particles,
                           MaterialParams const& materials,
                           SPConstImported data);

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
    CoulombScatteringData const& host_ref() const { return data_; }
    CoulombScatteringData const& device_ref() const { return data_; }
    //!@}

  private:
    CoulombScatteringData data_;
    ImportedModelAdapter imported_;
    ImportedModelAdapter::EnergyBounds energy_limit_;
};

//---------------------------------------------------------------------------//
}  //  namespace celeritas
