//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftModel.cc
//---------------------------------------------------------------------------//
#include "WavelengthShiftModel.hh"

#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/math/PdfUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "WavelengthShiftExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Create a model builder from imported data.
 */
auto WavelengthShiftModel::make_builder(SPConstImported imported, Input input)
    -> ModelBuilder
{
    CELER_EXPECT(imported);
    return [imported = std::move(imported),
            input = std::move(input)](ActionId id) {
        return std::make_shared<WavelengthShiftModel>(id, imported, input);
    };
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data and imported material parameters.
 */
WavelengthShiftModel::WavelengthShiftModel(ActionId id,
                                           SPConstImported imported,
                                           Input input)
    : Model(id, to_cstring(input.model), "interact by WLS")
    , imported_(input.model, std::move(imported))
{
    CELER_EXPECT(input.data.size() == imported_.num_materials());

    CELER_VALIDATE(input.model == ImportModelClass::wls
                       || input.model == ImportModelClass::wls2,
                   << "Invalid model '" << input.model
                   << "' for optical wavelength shifting");
    CELER_VALIDATE(input.time_profile != WlsTimeProfile::size_,
                   << "Invalid time profile for model '" << input.model << "'");

    SegmentIntegrator integrate_emission{TrapezoidSegmentIntegrator{}};

    HostVal<WavelengthShiftData> data;
    data.time_profile = input.time_profile;
    CollectionBuilder wls_record{&data.wls_record};
    NonuniformGridInserter insert_energy_cdf(&data.reals, &data.energy_cdf);
    for (auto const& wls : input.data)
    {
        if (!wls)
        {
            // No WLS data for this material
            wls_record.push_back({});
            insert_energy_cdf();
            continue;
        }

        // WLS material properties
        WlsMaterialRecord record;
        record.mean_num_photons = wls.mean_num_photons;
        record.time_constant = wls.time_constant;
        wls_record.push_back(record);

        // Calculate the WLS cumulative probability of the emission spectrum
        inp::Grid grid;
        grid.x = wls.component.x;
        grid.y.resize(grid.x.size());
        integrate_emission(make_span(wls.component.x),
                           make_span(wls.component.y),
                           make_span(grid.y));
        normalize_cdf(make_span(grid.y));

        // Insert energy -> CDF grid
        insert_energy_cdf(grid);
    }
    CELER_ASSERT(data.energy_cdf.size() == input.data.size());
    CELER_ASSERT(data.wls_record.size() == data.energy_cdf.size());

    data_ = CollectionMirror<WavelengthShiftData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void WavelengthShiftModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());
    build(imported_.mfp(mat));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void WavelengthShiftModel::step(CoreParams const& params,
                                CoreStateHost& state) const
{
    launch_action(
        state,
        make_action_thread_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            InteractionApplier{WavelengthShiftExecutor{this->host_ref()}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void WavelengthShiftModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
