//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CombinedBremModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/io/ImportSBTable.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

#include "RelativisticBremModel.hh"
#include "SeltzerBergerModel.hh"

namespace celeritas
{
struct ImportSBTable;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch a combined model of SeltzerBergerModel at the low energy
 * and RelativisticBremModel at the hight energy for e+/e- Bremsstrahlung.
 */
class CombinedBremModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using ReadData = std::function<ImportSBTable(AtomicNumber)>;
    using HostRef = HostCRef<CombinedBremData>;
    using DeviceRef = DeviceCRef<CombinedBremData>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //@}

  public:
    // Construct from model ID and other necessary data
    CombinedBremModel(ActionId id,
                      ParticleParams const& particles,
                      MaterialParams const& materials,
                      SPConstImported data,
                      ReadData load_sb_table,
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
    std::string label() const final { return "brems-combined"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "SB+relativistic electron+positron bremsstrahlung";
    }

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<CombinedBremData> data_;

    // Associated models
    std::shared_ptr<SeltzerBergerModel> sb_model_;
    std::shared_ptr<RelativisticBremModel> rb_model_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
