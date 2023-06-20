//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/WokviModel.cc
//---------------------------------------------------------------------------//
#include "WokviModel.hh"

#include "celeritas_config.h"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/data/WokviData.hh"
#include "celeritas/em/executor/WokviExecutor.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
WokviModel::WokviModel(ActionId id,
                       ParticleParams const& particles,
                       MaterialParams const& materials,
                       SPConstImported data)
    : imported_(data,
                particles,
                ImportProcessClass::coulomb_scat,  // TODO: Check the
                                                   // ImportProcessClass tags
                ImportModelClass::e_coulomb_scattering,  // TODO: Check the
                                                         // ImportModelClass
                                                         // tags
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);

    ScopedMem record_mem("WokviModel.construct");
    HostVal<WokviData> host_data;

    // This is where the data is built and transfered to the device

    data_ = CollectionMirror<WokviData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
}

auto WokviModel::applicability() const -> SetApplicability
{
    Applicability electron_applic;
    electron_applic.particle = this->host_ref().ids.electron;
    // TODO: construct actual energy range
    electron_applic.lower = zero_quantity();
    electron_applic.upper = max_quantity();

    Applicability positron_applic;
    positron_applic.particle = this->host_ref().ids.positron;
    positron_applic.lower = zero_quantity();
    positron_applic.upper = max_quantity();

    return {electron_applic, positron_applic};
}

auto WokviModel::micro_xs(Applicability applic) const -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

void WokviModel::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{WokviExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

#if !CELER_USE_DEVICE
void WokviModel::execute(CoreParams const&, coreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

ActionId WokviModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
