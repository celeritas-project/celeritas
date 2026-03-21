//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftModel.cc
//---------------------------------------------------------------------------//
#include "WavelengthShiftModel.hh"

#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/StreamUtils.hh"
#include "corecel/math/PdfUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MaterialParams.hh"
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
 * Construct the model from imported data and imported material parameters.
 */
WavelengthShiftModel::WavelengthShiftModel(ActionId id,
                                           inp::OpticalBulkWavelengthShift input,
                                           SPConstMaterials const& materials,
                                           std::string label)
    : Model(id, label, "interact by WLS"), input_(std::move(input))
{
    CELER_EXPECT(materials);

    CELER_VALIDATE(input_, << "invalid input for optical WLS model");

    SegmentIntegrator integrate_emission{TrapezoidSegmentIntegrator{}};

    HostVal<WavelengthShiftData> data;
    data.time_profile = input_.time_profile;
    CollectionBuilder wls_record{&data.wls_record};
    NonuniformGridInserter insert_energy_cdf(&data.reals, &data.energy_cdf);
    for (auto mat_id : range(OptMatId(materials->num_materials())))
    {
        auto iter = input_.materials.find(mat_id);
        if (iter == input_.materials.end())
        {
            // No WLS data for this material
            wls_record.push_back({});
            insert_energy_cdf();
            continue;
        }
        CELER_VALIDATE(iter->second,
                       << "invalid optical WLS properties in material "
                       << mat_id.get());

        // WLS material properties
        WlsMaterialRecord record;
        record.mean_num_photons = iter->second.mean_num_photons;
        record.time_constant = iter->second.time_constant;
        wls_record.push_back(record);

        // Calculate the WLS cumulative probability of the emission spectrum
        inp::Grid grid;
        grid.x = iter->second.component.x;
        grid.y.resize(grid.x.size());
        integrate_emission(make_span(std::as_const(iter->second.component.x)),
                           make_span(std::as_const(iter->second.component.y)),
                           make_span(grid.y));
        normalize_cdf(make_span(grid.y));

        // Insert energy -> CDF grid
        insert_energy_cdf(grid);
    }
    CELER_ASSERT(data.energy_cdf.size() == materials->num_materials());
    CELER_ASSERT(data.wls_record.size() == data.energy_cdf.size());

    data_ = ParamsDataStore<WavelengthShiftData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void WavelengthShiftModel::build_mfps(OptMatId mat, MfpBuilder& build) const
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
