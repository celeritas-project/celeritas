//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnostic.cc
//---------------------------------------------------------------------------//
#include "FieldDiagnostic.hh"

#include <algorithm>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Label.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/FieldDiagnosticExecutor.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
char const* to_cstring(FieldTally value)
{
    static EnumStringMapper<FieldTally> const to_cstring_impl{
        "steps",
        "boundary",
        "looping",
        "stuck",
        "substeps",
        "accepted",
        "bumped",
        "intercepted",
        "shortened",
        "integrations",
        "first_substep_integrations",
        "quick_advances",
        "chord_iterations",
        "accurate_advances",
        "accurate_iterations",
        "good_step_rejects",
        "intersects",
        "intersect_hits",
        "skippable_intersects",
        "safeties",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
char const* to_cstring(FieldStepHist value)
{
    static EnumStringMapper<FieldStepHist> const to_cstring_impl{
        "substeps",
        "integrations",
        "first_substep_integrations",
        "intersects",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
char const* to_cstring(FieldTrackHist value)
{
    static EnumStringMapper<FieldTrackHist> const to_cstring_impl{
        "steps",
        "substeps",
        "integrations",
        "intersects",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
char const* to_cstring(FieldStepIter value)
{
    static EnumStringMapper<FieldStepIter> const to_cstring_impl{
        "tracks",
        "substeps",
        "integrations",
        "intersects",
        "max_substeps",
        "max_integrations",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
//! Integer value of a diagnostic enum
template<class E>
constexpr size_type index_of(E e)
{
    return static_cast<size_type>(e);
}

//---------------------------------------------------------------------------//
//! Copy a slice and remove trailing zeros
std::vector<ull_int> trimmed(
    std::vector<ull_int> const& values, size_type start, size_type size)
{
    auto first = values.begin() + start;
    auto last = first + size;
    while (last != first && *(last - 1) == 0)
    {
        --last;
    }
    return {first, last};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with particle and volume data.
 */
FieldDiagnostic::FieldDiagnostic(ActionId id,
                                 Input const& input,
                                 SPConstParticle particle,
                                 SPConstVolume volume,
                                 size_type num_streams)
    : id_{id}, particle_{std::move(particle)}, volume_{std::move(volume)}
{
    CELER_EXPECT(id_);
    CELER_EXPECT(particle_);
    CELER_EXPECT(volume_);
    CELER_EXPECT(num_streams > 0);
    CELER_VALIDATE(input.bins > 0,
                   << "nonpositive field diagnostic bins " << input.bins);

    HostVal<FieldDiagnosticParamsData> host_params;
    host_params.num_bins = input.bins + 1;
    host_params.num_particles = particle_->size();
    host_params.num_volumes = volume_->num_volumes();
    host_params.num_step_records = input.step_records;
    host_params.oracle_safety = input.oracle_safety;
    store_ = {std::move(host_params), num_streams};

    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
/*!
 * Description of the action.
 */
std::string_view FieldDiagnostic::description() const
{
    return "accumulate field propagation counters";
}

//---------------------------------------------------------------------------//
/*!
 * Bin the propagation records with host data.
 */
void FieldDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto& diag_state
        = this->state<MemSpace::native>(state.stream_id(), state.size());
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::FieldDiagnosticExecutor{this->params<MemSpace::native>(),
                                        diag_state});
    launch_action(*this, params, state, execute);
    ++diag_state.step_iter;
}

#if !CELER_USE_DEVICE
void FieldDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void FieldDiagnostic::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto const& p = store_.params<MemSpace::host>();

    // Sum a collection over all streams
    auto sum_streams = [this](auto get, size_type size) {
        std::vector<ull_int> result(size, 0);
        accumulate_over_streams(store_, get, &result);
        return result;
    };

    auto obj = json::object();

    // Particle names
    {
        auto labels = json::array();
        for (auto pid : range(ParticleId{particle_->size()}))
        {
            labels.push_back(particle_->id_to_label(pid));
        }
        obj["particles"] = std::move(labels);
    }

    // Number of count bins, the last of which is overflow
    obj["num_bins"] = p.num_bins;

    // Histograms per step and per track for each quantity and particle
    auto hist_obj = [&](auto get, auto size_enum) {
        using E = decltype(size_enum);
        size_type const num_q = index_of(E::size_);
        auto counts = sum_streams(get, p.num_particles * num_q * p.num_bins);
        auto result = json::object();
        for (auto q : range(E::size_))
        {
            auto per_particle = json::array();
            for (auto pid : range(p.num_particles))
            {
                per_particle.push_back(
                    trimmed(counts,
                            (pid * num_q + index_of(q)) * p.num_bins,
                            p.num_bins));
            }
            result[to_cstring(q)] = std::move(per_particle);
        }
        return result;
    };
    obj["step"]
        = hist_obj([](auto& s) { return s.step_hist; }, FieldStepHist::size_);
    obj["track"] = hist_obj([](auto& s) { return s.track_hist; },
                            FieldTrackHist::size_);

    // Tallies by energy and volume
    constexpr size_type num_tally = index_of(FieldTally::size_);
    {
        auto tally
            = sum_streams([](auto& s) { return s.energy; }, p.energy_size());
        auto per_q = json::object();
        for (auto q : range(FieldTally::size_))
        {
            std::vector<ull_int> values(p.num_energy_bins);
            for (auto bin : range(p.num_energy_bins))
            {
                values[bin] = tally[bin * num_tally + index_of(q)];
            }
            per_q[to_cstring(q)] = std::move(values);
        }
        obj["energy"] = {
            {"log_min", p.log_energy_min},
            {"bins_per_decade", p.energy_bins_per_decade},
            {"tally", std::move(per_q)},
        };
    }
    if (p.num_volumes > 0)
    {
        auto tally
            = sum_streams([](auto& s) { return s.volume; }, p.volume_size());
        // Only write volumes where propagation occurred
        std::vector<size_type> vols;
        for (auto vol : range(p.num_volumes))
        {
            if (tally[vol * num_tally + index_of(FieldTally::steps)] > 0)
            {
                vols.push_back(vol);
            }
        }
        auto labels = json::array();
        for (auto vol : vols)
        {
            labels.push_back(
                to_string(volume_->volume_labels().at(VolumeId(vol))));
        }
        auto per_q = json::object();
        for (auto q : range(FieldTally::size_))
        {
            std::vector<ull_int> values;
            values.reserve(vols.size());
            for (auto vol : vols)
            {
                values.push_back(tally[vol * num_tally + index_of(q)]);
            }
            per_q[to_cstring(q)] = std::move(values);
        }
        obj["volume"] = {
            {"label", std::move(labels)},
            {"tally", std::move(per_q)},
        };
    }

    if (p.oracle_safety)
    {
        auto counts
            = sum_streams([](auto& s) { return s.safety; }, p.safety_size());
        auto per_particle = json::array();
        for (auto pid : range(p.num_particles))
        {
            per_particle.push_back(
                trimmed(counts, pid * p.num_safety_bins, p.num_safety_bins));
        }
        obj["safety"] = {
            {"log_min", p.log_safety_min},
            {"bins_per_decade", p.safety_bins_per_decade},
            {"counts", std::move(per_particle)},
        };
    }

    // Step iteration records for each stream
    if (p.num_step_records > 0)
    {
        constexpr size_type num_iter_q = index_of(FieldStepIter::size_);
        auto streams = json::array();
        auto append_stream = [&](auto const* s) {
            if (!s)
            {
                return;
            }
            size_type num_iters = std::min(s->step_iter, p.num_step_records);
            std::vector<ull_int> values(s->step_iters.size());
            copy_to_host(s->step_iters, make_span(values));
            auto per_q = json::object();
            for (auto q : range(FieldStepIter::size_))
            {
                std::vector<ull_int> series(num_iters);
                for (auto i : range(num_iters))
                {
                    series[i] = values[i * num_iter_q + index_of(q)];
                }
                per_q[to_cstring(q)] = std::move(series);
            }
            if (s->step_iter > p.num_step_records)
            {
                per_q["_truncated"] = s->step_iter;
            }
            streams.push_back(std::move(per_q));
        };
        for (auto sid : range(StreamId{store_.num_streams()}))
        {
            append_stream(store_.state<MemSpace::host>(sid));
            append_stream(store_.state<MemSpace::device>(sid));
        }
        obj["step_iter"] = std::move(streams);
    }

    obj["_index"] = {
        {"step", {"quantity", "particle", "count"}},
        {"track", {"quantity", "particle", "count"}},
        {"energy", {"quantity", "energy_bin"}},
        {"volume", {"quantity", "volume"}},
        {"safety", {"particle", "safety_bin"}},
        {"step_iter", {"stream", "quantity", "step_iter"}},
    };

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
