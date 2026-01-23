//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "HeuristicGeoTestBase.hh"

#include <iomanip>
#include <iostream>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/ScopedStreamFormat.hh"
#include "corecel/random/params/RngParams.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "HeuristicGeoExecutor.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run tracks on host and device, and compare the resulting path length.
 */
void HeuristicGeoTestBase::run(size_type num_states,
                               size_type num_steps,
                               real_type tolerance)
{
    auto host_path = this->run_impl<MemSpace::host>(num_states, num_steps);

    auto ref_path = this->reference_avg_path();
    if (ref_path.empty())
    {
        ScopedStreamFormat save_fmt(&std::cout);

        ADD_FAILURE() << "Implement the following as "
                         "TestCase::reference_avg_path() const";

        int precision_digits
            = static_cast<int>(std::ceil(-std::log10(tolerance) + 0.5));

        std::cout << "/* REFERENCE PATH LENGTHS */\n"
                     "static real_type const paths[] = {"
                  << std::setprecision(precision_digits)
                  << join(host_path.begin(), host_path.end(), ", ")
                  << "};\n"
                     "/* END REFERENCE PATH LENGTHS */\n";
    }
    else if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
    {
        EXPECT_VEC_NEAR(ref_path, host_path, tolerance)
            << "Host results differ from reference";
    }
    else
    {
        std::cout << "Skipping reference comparison: non-default RNG\n";
    }

    if (celeritas::device())
    {
        device().create_streams(1);

        auto dvc_path = this->run_impl<MemSpace::device>(num_states, num_steps);
        EXPECT_VEC_SOFT_EQ(host_path, dvc_path)
            << R"(GPU and CPU produced different results)";
    }
    else
    {
        std::cout << "Skipping device comparison: device not active\n";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Run tracks on host *or* device and return the resulting path lengths.
 */
template<MemSpace M>
auto HeuristicGeoTestBase::run_impl(size_type num_states, size_type num_steps)
    -> VecReal
{
    auto host_params = this->build_test_params<MemSpace::host>();
    StateStore<M> state{host_params, num_states};

    auto params = this->build_test_params<M>();

    CELER_LOG(status) << "Running heuristic test on " << to_cstring(M);
    if constexpr (M == MemSpace::host)
    {
        HeuristicGeoExecutor execute{make_observer(&params),
                                     make_observer(&state.ref())};
        for (auto tid : range(TrackSlotId{num_states}))
        {
            for ([[maybe_unused]] auto step : range(num_steps))
            {
                execute.state_ptr->step = step;
                execute(tid);
            }
        }
    }
    else
    {
        // Create device vectors to hold params and state refs
        DeviceVector<DeviceCRef<HeuristicGeoParamsData>> params_vec(1);
        DeviceVector<DeviceRef<HeuristicGeoStateData>> state_vec(1);

        params_vec.copy_to_device({&params, 1});

        auto state_ref = state.ref();
        for ([[maybe_unused]] auto step : range(num_steps))
        {
            state_ref.step = step;
            state_vec.copy_to_device({&state_ref, 1});
            heuristic_test_execute(make_observer(params_vec),
                                   make_observer(state_vec),
                                   num_states);
        }
    }

    VecReal host_accum_path(state.ref().accum_path.size());
    copy_to_host(state.ref().accum_path, make_span(host_accum_path));
    return this->get_avg_path_impl(host_accum_path, num_states);
}

//---------------------------------------------------------------------------//

template<MemSpace M>
auto HeuristicGeoTestBase::build_test_params()
    -> HeuristicGeoParamsData<Ownership::const_reference, M>
{
    auto const& geo = *this->geometry();

    HeuristicGeoParamsData<Ownership::const_reference, M> result;
    result.s = this->build_scalars();
    result.s.num_volumes = geo.impl_volumes().size();
    result.s.ignore_zero_safety = geo.supports_safety();
    CELER_ASSERT(result.s);

    result.geometry = get_ref<M>(geo);
    result.rng = get_ref<M>(*this->rng());
    return result;
}

//---------------------------------------------------------------------------//

auto HeuristicGeoTestBase::get_avg_path_impl(VecReal const& path,
                                             size_type num_states) const
    -> VecReal
{
    CELER_EXPECT(path.size() == this->geometry()->impl_volumes().size());

    auto const& geo = *this->geometry();

    std::vector<std::string> temp_labels;
    SpanConstStr ref_vol_labels = this->reference_volumes();
    if (ref_vol_labels.empty())
    {
        temp_labels.reserve(geo.impl_volumes().size());
        for (auto vid : range(ImplVolumeId{geo.impl_volumes().size()}))
        {
            std::string const& vol_name = geo.impl_volumes().at(vid).name;
            if (vol_name != "[EXTERIOR]")
            {
                temp_labels.push_back(vol_name);
            }
        }

        ADD_FAILURE() << "Implement the following as "
                         "TestCase::reference_volumes() const";
        std::cout << "/* REFERENCE VOLUMES */\n"
                     "static std::string const vols[] = "
                  << repr(temp_labels)
                  << ";\n"
                     "/* END REFERENCE VOLUMES */\n";
        ref_vol_labels = make_span(temp_labels);
    }

    VecReal result(ref_vol_labels.size());

    real_type const norm = 1 / real_type(num_states);
    for (auto i : range(ref_vol_labels.size()))
    {
        auto vol_id = geo.impl_volumes().find_unique(ref_vol_labels[i]);
        if (vol_id)
        {
            result[i] = path[vol_id.unchecked_get()] * norm;
        }
        else
        {
            ADD_FAILURE() << "reference volume '" << ref_vol_labels[i]
                          << "' is not in the geometry";
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
// DEVICE KERNEL EXECUTION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void heuristic_test_execute(HeuristicGeoParamsPtr<MemSpace::device>,
                            HeuristicGeoStatePtr<MemSpace::device>,
                            size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
