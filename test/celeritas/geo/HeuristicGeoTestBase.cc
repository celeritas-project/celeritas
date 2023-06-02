//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "HeuristicGeoTestBase.hh"

#include <iomanip>
#include <iostream>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/ScopedStreamFormat.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/random/RngParams.hh"

#include "HeuristicGeoExecutor.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run tracks on host and compare the resulting path length.
 */
void HeuristicGeoTestBase::run_host(size_type num_states, real_type tolerance)
{
    const size_type num_steps = this->num_steps();
    auto params = this->build_test_params<MemSpace::host>();
    StateStore<MemSpace::host> state{params, num_states};

    HeuristicGeoExecutor execute{params, state.ref()};
    for (auto tid : range(TrackSlotId{num_states}))
    {
        for ([[maybe_unused]] auto step : range(num_steps))
        {
            execute(tid);
        }
    }

    auto avg_path = this->get_avg_path(state.ref().accum_path, num_states);
    auto ref_path = this->reference_avg_path();

    if (ref_path.empty())
    {
        ScopedStreamFormat save_fmt(&std::cout);

        ADD_FAILURE() << "Implement the following as "
                         "TestCase::reference_avg_path() const";

        int precision_digits = std::ceil(-std::log10(tolerance) + 0.5);

        std::cout << "/* REFERENCE PATH LENGTHS */\n"
                     "static const real_type paths[] = {"
                  << std::setprecision(precision_digits)
                  << join(avg_path.begin(), avg_path.end(), ", ")
                  << "};\n"
                     "/* END REFERENCE PATH LENGTHS */\n";

        return;
    }

    EXPECT_VEC_NEAR(ref_path, avg_path, tolerance);
}

//---------------------------------------------------------------------------//
/*!
 * Run tracks on device and compare the resulting path length.
 */
void HeuristicGeoTestBase::run_device(size_type num_states, real_type tolerance)
{
    const size_type num_steps = this->num_steps();

    auto params = this->build_test_params<MemSpace::device>();
    StateStore<MemSpace::device> state{
        this->build_test_params<MemSpace::host>(), num_states};

    for ([[maybe_unused]] auto step : range(num_steps))
    {
        heuristic_test_execute(params, state.ref());
    }

    auto avg_path = this->get_avg_path(state.ref().accum_path, num_states);
    EXPECT_VEC_NEAR(this->reference_avg_path(), avg_path, tolerance);
}

//---------------------------------------------------------------------------//

auto HeuristicGeoTestBase::reference_volumes() const -> SpanConstStr
{
    GeoParams const& geo = *this->geometry();
    temp_str_.reserve(geo.num_volumes());
    for (auto vid : range(VolumeId{geo.num_volumes()}))
    {
        std::string const& vol_name = geo.id_to_label(vid).name;
        if (vol_name != "[EXTERIOR]")
        {
            temp_str_.push_back(vol_name);
        }
    }

    ADD_FAILURE() << "Implement the following as "
                     "TestCase::reference_volumes() const";
    std::cout << "/* REFERENCE VOLUMES */\n"
                 "static const std::string vols[] = "
              << repr(temp_str_)
              << ";\n"
                 "/* END REFERENCE VOLUMES */\n";
    return make_span(temp_str_);
}

//---------------------------------------------------------------------------//

auto HeuristicGeoTestBase::reference_avg_path() const -> SpanConstReal
{
    return {};
}

//---------------------------------------------------------------------------//

template<MemSpace M>
auto HeuristicGeoTestBase::build_test_params()
    -> HeuristicGeoParamsData<Ownership::const_reference, M>
{
    auto const& geo = *this->geometry();

    HeuristicGeoParamsData<Ownership::const_reference, M> result;
    result.s = this->build_scalars();
    result.s.num_volumes = geo.num_volumes();
    result.s.ignore_zero_safety = geo.supports_safety();
    CELER_ASSERT(result.s);

    result.geometry = get_ref<M>(geo);
    result.rng = get_ref<M>(*this->rng());
    return result;
}

//---------------------------------------------------------------------------//

template<MemSpace M>
auto HeuristicGeoTestBase::get_avg_path(PathLengthRef<M> path,
                                        size_type num_states) const
    -> std::vector<real_type>
{
    std::vector<real_type> result(path.size());
    Copier<real_type, MemSpace::host> copy_to_host{make_span(result)};
    copy_to_host(M, path[AllItems<real_type, M>{}]);

    return this->get_avg_path_impl(result, num_states);
}

//---------------------------------------------------------------------------//

auto HeuristicGeoTestBase::get_avg_path_impl(std::vector<real_type> const& path,
                                             size_type num_states) const
    -> std::vector<real_type>
{
    CELER_EXPECT(path.size() == this->geometry()->num_volumes());

    SpanConstStr ref_vol_labels = this->reference_volumes();
    std::vector<real_type> result(ref_vol_labels.size());

    auto const& geo = *this->geometry();
    const real_type norm = 1 / real_type(num_states);
    for (auto i : range(ref_vol_labels.size()))
    {
        auto vol_id = geo.find_volume(ref_vol_labels[i]);
        CELER_VALIDATE(vol_id,
                       << "reference volme '" << ref_vol_labels[i]
                       << "' is not in the geometry");
        result[i] = path[vol_id.unchecked_get()] * norm;
    }
    return result;
}

//---------------------------------------------------------------------------//
// DEVICE KERNEL EXECUTION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void heuristic_test_execute(DeviceCRef<HeuristicGeoParamsData> const&,
                            DeviceRef<HeuristicGeoStateData> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
