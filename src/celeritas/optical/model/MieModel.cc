//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieModel.cc
//---------------------------------------------------------------------------//
#include "MieModel.hh"

#include <limits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/inp/Grid.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/MieData.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"
#include "celeritas/optical/model/MieExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from input data.
 */
MieModel::MieModel(ActionId id,
                   inp::OpticalBulkMie input,
                   SPConstMaterials const& materials)
    : Model(id, "optical-mie", "interact by optical Mie scattering")
    , input_(std::move(input))
{
    HostVal<MieData> data;
    CollectionBuilder builder{&data.mie_record};

    for (auto mat_id : range(OptMatId(materials->num_materials())))
    {
        auto iter = input_.materials.find(mat_id);
        if (iter == input_.materials.end())
        {
            // No Mie data for this material
            builder.push_back({});
            continue;
        }

        MieMaterialData record;
        record.backward_g = iter->second.backward_g;
        record.forward_g = iter->second.forward_g;
        record.forward_ratio = iter->second.forward_ratio;
        CELER_VALIDATE(record || !iter->second.mfp,
                       << "Mie parameters out of bounds for material "
                       << mat_id.get() << ": forward_g=" << record.forward_g
                       << ", backward_g=" << record.backward_g
                       << ", forward_ratio=" << record.forward_ratio);

        builder.push_back(record);
    }
    CELER_ASSERT(data.mie_record.size() == materials->num_materials());

    data_ = ParamsDataStore<MieData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void MieModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    if (auto iter = input_.materials.find(mat); iter != input_.materials.end())
    {
        build(iter->second.mfp);
    }
    else
    {
        build();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void MieModel::step(CoreParams const& params, CoreStateHost& state) const
{
    launch_action(state,
                  make_action_thread_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      this->action_id(),
                      InteractionApplier{MieExecutor{this->host_ref()}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void MieModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
