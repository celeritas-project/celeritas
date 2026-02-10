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
 * Create a model builder from imported data.
 */
auto MieModel::make_builder(SPConstImported imported, Input input)
    -> ModelBuilder
{
    CELER_EXPECT(imported);
    return [imported = std::move(imported),
            input = std::move(input)](ActionId id) {
        return std::make_shared<MieModel>(id, imported, input);
    };
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data and imported material parameters.
 */
MieModel::MieModel(ActionId id, SPConstImported imported, Input input)
    : Model(id, "optical-mie", "interact by optical Mie scattering")
    , imported_(ImportModelClass::mie, std::move(imported))
{
    HostVal<MieData> data;
    CollectionBuilder builder{&data.mie_record};

    for (auto const& mie : input.data)
    {
        MieMaterialData record;
        record.backward_g = mie.backward_g;
        record.forward_g = mie.forward_g;
        record.forward_ratio = mie.forward_ratio;

        builder.push_back(record);
    }

    data_ = ParamsDataStore<MieData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void MieModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());

    if (auto const& mfp = imported_.mfp(mat))
    {
        auto const& mie_data = this->host_ref().mie_record[mat];
        CELER_VALIDATE(mie_data,
                       << "Mie parameters out of bounds for material "
                       << mat.get() << ": forward_g=" << mie_data.forward_g
                       << ", backward_g=" << mie_data.backward_g
                       << ", forward_ratio=" << mie_data.forward_ratio);

        build(mfp);
    }
    else
    {
        // Cross sections are not available: disable for this material
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
