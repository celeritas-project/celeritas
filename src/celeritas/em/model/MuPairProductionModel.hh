//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuPairProductionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuPairProductionData.hh"
#include "celeritas/io/ImportMuPairProductionTable.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the muon pair production model.
 */
class MuPairProductionModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using HostRef = HostCRef<MuPairProductionData>;
    using DeviceRef = DeviceCRef<MuPairProductionData>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    MuPairProductionModel(ActionId,
                          ParticleParams const&,
                          SPConstImported,
                          ImportMuPairProductionTable const&);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    CollectionMirror<MuPairProductionData> data_;
    ImportedModelAdapter imported_;

    using HostTable = HostVal<MuPairProductionTableData>;
    void build_table(ImportMuPairProductionTable const&, HostTable*) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
