//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
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
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "brems-rel"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "Relativistic bremsstrahlung";
    }

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device(); }

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
