//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/RelativisticBremModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the relativistic Bremsstrahlung model for high-energy
 * electrons and positrons with the Landau-Pomeranchuk-Migdal (LPM) effect
 */
class RelativisticBremModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using HostRef = HostCRef<RelativisticBremData>;
    using DeviceRef = DeviceCRef<RelativisticBremData>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //@}

  public:
    // Construct from model ID and other necessary data
    RelativisticBremModel(ActionId id,
                          ParticleParams const& particles,
                          MaterialParams const& materials,
                          SPConstImported data,
                          bool enable_lpm);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel to host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "brems-rel"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by relativistic bremsstrahlung";
    }

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<RelativisticBremData> data_;

    ImportedModelAdapter imported_;

    //// TYPES ////

    using HostValue = HostVal<RelativisticBremData>;

    using FormFactor = RelBremFormFactor;
    using ElementData = RelBremElementData;

    //// HELPER FUNCTIONS ////

    void build_data(HostValue* host_data,
                    MaterialParams const& materials,
                    real_type particle_mass);

    static FormFactor const& get_form_factor(AtomicNumber index);
    ElementData
    compute_element_data(ElementView const& elem, real_type particle_mass);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
